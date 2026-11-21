"""
sistem.py — UÇTAN UCA OTOMATİK SİSTEM (projenin nihai hali)
==========================================================

    yayın karesi
       ├─► keypoint modeli (sinir ağı #2) → saha noktaları
       │       └─► RANSAC homografi (yanlış tahminleri ayıklar)
       ├─► YOLO (sinir ağı #1) → oyuncular → forma rengi → K-means → takımlar
       └─► ayak noktaları × homografi → kuş bakışı mini harita

Elle tıklama YOK. manuel_kalibrasyon.py'deki elle tıklama adımının yerini
keypoint modeli aldı; geri kalan her şey aynı kod.

VİDEO MODU (maç özeti → kuş bakışı video)
  Video kare kare okunur, her kare yukarıdaki boru hattından geçer, sonuç kareleri
  ffmpeg ile tarayıcıda oynayan bir H.264 videoya yazılır.
  * ZAMANSAL YUMUŞATMA: her kare için ayrı homografi hesaplanınca harita titrer.
    Çözüm üstel hareketli ortalama (EMA):  H_yeni = α · H_bu_kare + (1 − α) · H_önceki
    α küçük → daha yumuşak ama kamera hızlı dönünce geç kalır.
  * Bu karede homografi bulunamazsa önceki kullanılır (en fazla 'tolerans' kare).
  * SAHNE KESMESİ: yayın başka kameraya geçince (renk histogramı birden değişir) önceki homografi unutulur.
  * Videonun her karesi işlenmez: saniyede 'hedef_fps' kare (varsayılan 12) yeterince akıcı.
  * Saha görünmeyen karelerde (yakın plan, tribün, tekrar geçişi) harita yerine
    "saha görünmüyor" yazılır → video senkron kalır.

Kullanım:
    .venv\\Scripts\\python.exe src\\sistem.py --model models\\unet_kp_gercek_v1.pt --goruntu data\\ornekler\\ornek2.webp
    .venv\\Scripts\\python.exe src\\sistem.py --model models\\unet_kp_gercek_v1.pt --video data\\raw_videos\\12_gxsTs2qtV8A.mp4 --bas 60 --sure 20
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

import pitch
from homografi import HomografiSonuc, isareti_duzelt
from keypoint_tahmin import SahaModeli, model_yukle, oku, ransac_homografi
from manuel_kalibrasyon import sonuc_uret

PROJE = Path(__file__).resolve().parent.parent
CIKTI = PROJE / "results" / "sistem"
ARA = 10   # sonuc_uret'in görüntü ile harita arasına koyduğu beyaz şerit (px)


def kare_isle(model: SahaModeli, img: np.ndarray, kal_onceki=None, alfa: float = 1.0,
              yolo: str = "yolov8n.pt", noktalari_ciz: bool = True):
    noktalar = model.tahmin(img)
    kal = ransac_homografi(noktalar)
    if kal is not None and kal_onceki is not None and alfa < 1.0:
        # H'ler ölçek-belirsiz: karşılaştırmadan önce H[2,2] = 1 olacak şekilde normalize et
        H = alfa * kal.H / kal.H[2, 2] + (1 - alfa) * kal_onceki.H / kal_onceki.H[2, 2]
        # H[2,2]'ye bölmek işareti yine ters çevirebilir (ufuk görüntüdeyse) → düzelt
        H = isareti_duzelt(H, np.array([n["px"] for n in noktalar]))
        kal = HomografiSonuc(H, np.linalg.inv(H), kal.hata_metre, kal.hata_piksel, kal.nokta_hatalari)
    if kal is None:
        kal = kal_onceki
    if kal is None:
        return None, None, noktalar
    birlesik, sahadakiler, kal = sonuc_uret(img, noktalar, kal=kal, yolo=yolo)
    if noktalari_ciz:   # modelin bulduğu noktaları da göster
        for n in noktalar:
            cv2.circle(birlesik, tuple(np.round(n["px"]).astype(int)), 5, (0, 0, 255), -1)
    return birlesik, kal, noktalar


def renk_histogrami(img: np.ndarray) -> np.ndarray:
    """Küçültülmüş karenin renk tonu × doygunluk histogramı (sahne kesmesi için parmak izi)."""
    k = cv2.cvtColor(cv2.resize(img, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([k], [0, 1], None, [30, 32], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def bos_kare(img: np.ndarray, mesaj: str) -> np.ndarray:
    """Homografi yokken: görüntü + soluk boş harita + mesaj (birlesik ile aynı boyutta)."""
    harita = pitch.saha_ciz()
    olcek = img.shape[0] / harita.shape[0]
    harita = cv2.resize(harita, (int(harita.shape[1] * olcek), img.shape[0]))
    harita = cv2.addWeighted(harita, 0.35, np.full_like(harita, 40), 0.65, 0)
    s = img.shape[0] / 720
    (tw, th), _ = cv2.getTextSize(mesaj, cv2.FONT_HERSHEY_SIMPLEX, 1.0 * s, max(1, int(2 * s)))
    cv2.putText(harita, mesaj, ((harita.shape[1] - tw) // 2, (harita.shape[0] + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0 * s, (255, 255, 255), max(1, int(2 * s)), cv2.LINE_AA)
    return np.hstack([img, np.full((img.shape[0], ARA, 3), 255, np.uint8), harita])


class VideoYazici:
    """ffmpeg varsa tarayıcıda oynayan H.264 (yuv420p) yazar; yoksa OpenCV mp4v."""

    def __init__(self, yol: Path, fps: float, w: int, h: int):
        self.yol, self.w, self.h = yol, w, h
        ff = shutil.which("ffmpeg")
        if ff:
            self.p = subprocess.Popen(
                [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                 "-r", f"{fps:.3f}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(yol)],
                stdin=subprocess.PIPE)
            self.cv = None
        else:
            self.p = None
            self.cv = cv2.VideoWriter(str(yol), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def yaz(self, kare: np.ndarray):
        if kare.shape[1] != self.w or kare.shape[0] != self.h:
            kare = cv2.resize(kare, (self.w, self.h), interpolation=cv2.INTER_AREA)
        if self.p:
            self.p.stdin.write(np.ascontiguousarray(kare).tobytes())
        else:
            self.cv.write(kare)

    def kapat(self):
        if self.p:
            self.p.stdin.close()
            self.p.wait()
        else:
            self.cv.release()


def video_isle(model, video: str | Path, cikti: Path, bas: float = 0, sure: float | None = None,
               hedef_fps: float = 12, alfa: float = 0.3, tolerans: int = 6, kesme_esigi: float = 0.35,
               yolo: str = "yolov8n.pt", sadece_harita: bool = False, genislik: int = 1280,
               ilerleme=None) -> dict:
    """
    Videoyu kuş bakışı videoya çevirir.
      hedef_fps     : saniyede kaç kare işlensin (25/60 fps'lik videonun her karesi gerekmez;
                      12 → 60 fps videoda her 5. kare). Az → hızlı ama daha kesik.
      kesme_esigi   : ardışık iki karenin renk histogramı farkı (Bhattacharyya, 0..1) bunu aşarsa
                      sahne kesmesi. Ölçüm: aynı çekimde fark ≤ 0.26, gerçek kesmelerde ≥ 0.46.
      sadece_harita : True → sadece kuş bakışı harita, False → görüntü + harita yan yana
      ilerleme      : ilerleme(oran 0..1, mesaj) çağrılır (arayüzdeki ilerleme çubuğu için)
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Video açılamadı: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    adim = max(1, round(fps / hedef_fps))
    toplam = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 10 ** 9
    bas_kare = int(bas * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, bas_kare)
    son = min(toplam, bas_kare + int(sure * fps)) if sure else toplam
    n_hedef = max(1, (son - bas_kare + adim - 1) // adim)

    yazici, kal_onceki, kayip, onceki_hash = None, None, 0, None
    n_islenen = n_ok = n_kesme = 0
    t0 = time.time()
    for j in range(son - bas_kare):
        if j % adim:
            if not cap.grab():          # atlanan kareyi çözmeden geç (hızlı)
                break
            continue
        ok, img = cap.read()
        if not ok:
            break
        n_islenen += 1
        # SAHNE KESMESİ: ardışık iki karenin renk dağılımı çok farklıysa yayın başka
        # kameraya geçmiştir → önceki homografi artık geçersiz, unut.
        # (dHash denendi: yeşil sahada kararsız, aynı çekimde bile 10 sn'de 25 sahte kesme buldu.)
        h = renk_histogrami(img)
        if onceki_hash is not None and cv2.compareHist(onceki_hash, h, cv2.HISTCMP_BHATTACHARYYA) > kesme_esigi:
            kal_onceki, kayip = None, 0
            n_kesme += 1
        onceki_hash = h

        birlesik, kal, _ = kare_isle(model, img, kal_onceki, alfa, yolo, noktalari_ciz=False)
        if birlesik is not None:
            kayip = 0 if kal is not kal_onceki else kayip + 1
            if kayip > tolerans:
                kal_onceki, birlesik = None, None
        if birlesik is None:
            birlesik = bos_kare(img, "saha gorunmuyor")
        else:
            kal_onceki, n_ok = kal, n_ok + 1
        if sadece_harita:
            birlesik = birlesik[:, img.shape[1] + ARA:]

        if yazici is None:
            olcek = min(1.0, genislik / birlesik.shape[1])
            w, hh = int(birlesik.shape[1] * olcek) // 2 * 2, int(birlesik.shape[0] * olcek) // 2 * 2
            cikti.parent.mkdir(parents=True, exist_ok=True)
            yazici = VideoYazici(cikti, fps / adim, w, hh)
        yazici.yaz(birlesik)
        if ilerleme and (n_islenen % 5 == 0 or n_islenen == n_hedef):
            gecen = time.time() - t0
            kalan = gecen / n_islenen * (n_hedef - n_islenen)
            ilerleme(min(1.0, n_islenen / n_hedef), f"{n_islenen}/{n_hedef} kare · kalan ~{kalan:.0f} sn")
    cap.release()
    if yazici:
        yazici.kapat()
    return {"islenen": n_islenen, "haritalanan": n_ok, "kesme": n_kesme, "sure_sn": time.time() - t0,
            "video_sn": n_islenen * adim / fps, "yol": str(cikti) if yazici else None}


def main():
    ap = argparse.ArgumentParser(description="Uçtan uca otomatik kuş bakışı harita")
    ap.add_argument("--model", required=True, help="saha keypoint modeli (.pt)")
    ap.add_argument("--goruntu", nargs="*", default=[], help="bir veya daha fazla görüntü")
    ap.add_argument("--video", help="video dosyası")
    ap.add_argument("--bas", type=float, default=0, help="videoda başlangıç saniyesi")
    ap.add_argument("--sure", type=float, default=15, help="videodan kaç saniye işlensin")
    ap.add_argument("--fps", type=float, default=12, help="saniyede kaç kare işlensin")
    ap.add_argument("--alfa", type=float, default=0.3, help="video yumuşatma katsayısı (1 = yok)")
    ap.add_argument("--tolerans", type=int, default=6, help="homografi bulunamazsa kaç kare öncekini kullan")
    ap.add_argument("--kesme-esigi", type=float, default=0.35,
                    help="ardışık karelerin renk histogramı farkı bunu aşarsa sahne kesmesi say (0..1)")
    ap.add_argument("--yolo", default="yolov8n.pt", help="oyuncu tespit modeli")
    ap.add_argument("--sadece-harita", action="store_true", help="videoda sadece kuş bakışı harita")
    args = ap.parse_args()

    model = model_yukle(args.model)   # YOLO-pose ya da U-Net (dosya adına göre)
    CIKTI.mkdir(parents=True, exist_ok=True)

    for g in args.goruntu:
        img = oku(Path(g))
        birlesik, kal, noktalar = kare_isle(model, img, yolo=args.yolo)
        if birlesik is None:
            print(f"{Path(g).name}: homografi bulunamadı ({len(noktalar)} nokta)")
            continue
        yol = CIKTI / f"{Path(g).stem}.jpg"
        cv2.imwrite(str(yol), birlesik)
        ic = int(kal.icerdeki.sum()) if kal.icerdeki is not None else len(noktalar)
        print(f"{Path(g).name}: {len(noktalar)} nokta, RANSAC {ic} tanesini kabul etti, "
              f"bunlarda hata {kal.hata_metre:.2f} m / {kal.hata_piksel:.1f} px → {yol}")

    if args.video:
        yol = CIKTI / f"{Path(args.video).stem}_{int(args.bas)}s.mp4"
        s = video_isle(model, args.video, yol, args.bas, args.sure, args.fps, args.alfa, args.tolerans,
                       args.kesme_esigi, args.yolo, args.sadece_harita,
                       ilerleme=lambda o, m: print(f"\r{m}", end="", flush=True))
        print(f"\nVideo: {s['haritalanan']}/{s['islenen']} kare haritalandı, {s['kesme']} sahne kesmesi, "
              f"{s['sure_sn']:.0f} sn sürdü → {s['yol']}")


if __name__ == "__main__":
    main()
