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
from homografi import HomografiSonuc, cizgileri_goruntuye_ciz, isareti_duzelt, noktalari_donustur
from keypoint_tahmin import SahaModeli, model_yukle, oku, ransac_homografi
from manuel_kalibrasyon import sonuc_uret
from oyuncu import kesik_ayaklari_tahmin_et, oyunculari_takip_et
from takip import HomografiFiltresi, OyuncuTakipci, TakimAtayici

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


def video_kareleri(video: str | Path, bas: float = 0, sure: float | None = None, hedef_fps: float = 12,
                   kesme_esigi: float = 0.35):
    """
    Videodan saniyede ~hedef_fps kare okur. Üretir: (img, sahne_kesmesi_mi, bilgi).
    SAHNE KESMESİ: ardışık iki karenin renk dağılımı (HSV histogramı, Bhattacharyya uzaklığı)
    eşiği aşarsa yayın başka kameraya geçmiştir. Ölçüm: aynı çekimde ≤ 0.26, gerçek kesmede ≥ 0.46.
    (dHash denendi: yeşil sahada kararsız, aynı çekimde 10 sn'de 25 sahte kesme buldu.)
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
    bilgi = {"fps": fps, "adim": adim, "hedef": max(1, (son - bas_kare + adim - 1) // adim)}
    onceki = None
    try:
        for j in range(son - bas_kare):
            if j % adim:
                if not cap.grab():          # atlanan kareyi çözmeden geç (hızlı)
                    break
                continue
            ok, img = cap.read()
            if not ok:
                break
            h = renk_histogrami(img)
            kesme = onceki is not None and cv2.compareHist(onceki, h, cv2.HISTCMP_BHATTACHARYYA) > kesme_esigi
            onceki = h
            yield img, kesme, bilgi
    finally:
        cap.release()


def video_karesi_ciz(img: np.ndarray, kal, tespitler: list, oyuncular: list[dict], iz: bool = True) -> np.ndarray:
    """Sol: görüntü + oturtulmuş saha çizgileri + takım renkli kutular. Sağ: kuş bakışı harita (+ izler)."""
    sol = cizgileri_goruntuye_ciz(img, kal.H_ters)
    for t in tespitler:
        x1, y1, x2, y2 = map(int, t.kutu)
        renk = t.ek.get("takim_renk") or pitch.TAKIM_RENK.get(t.sinif, (0, 0, 255))
        cv2.rectangle(sol, (x1, y1), (x2, y2), tuple(int(c) for c in renk), 2)
        cv2.rectangle(sol, (x1, y1), (x2, y2), (0, 0, 0), 1)
    harita = pitch.saha_ciz()
    h, w = img.shape[:2]
    gorus = noktalari_donustur(kal.H, [[0, h], [w, h], [w, 0], [0, 0]])
    if not np.isnan(gorus).any():               # kameranın gördüğü alan
        poligon = np.array([pitch.metre_to_piksel(x, y) for x, y in
                            np.clip(gorus, [-50, -50], [155, 118])], np.int32)
        katman = harita.copy()
        cv2.fillPoly(katman, [poligon], (255, 255, 255))
        harita = cv2.addWeighted(katman, 0.15, harita, 0.85, 0)
    if iz:                                      # son ~1.5 sn'lik hareket izi
        for o in oyuncular:
            if o.get("sinif") not in pitch.HARITADA_GIZLI | pitch.HARITADA_SOLUK | {"top"} and len(o.get("iz", [])) > 1:
                p = np.array([pitch.metre_to_piksel(x, y) for x, y in o["iz"]], np.int32)
                renk = tuple(int(c) for c in (o.get("renk") or (200, 200, 200)))
                cv2.polylines(harita, [p], False, renk, 2, cv2.LINE_AA)
    harita = pitch.oyunculari_ciz(harita, oyuncular)
    olcek = sol.shape[0] / harita.shape[0]
    harita = cv2.resize(harita, (int(harita.shape[1] * olcek), sol.shape[0]))
    return np.hstack([sol, np.full((sol.shape[0], ARA, 3), 255, np.uint8), harita])


class VideoHatti:
    """
    Videoda kare kare: saha noktaları → RANSAC → (filtre) → tespit + takip → saha konumu →
    takım ataması → (konum yumuşatma). yumusat=False → her kare bağımsız (karşılaştırma için).
    """

    def __init__(self, model, yolo: str = "yolov8n.pt", yumusat: bool = True, top_goster: bool = False,
                 kararli_takim: bool = True, hiz_m: float | None = 1.0):
        # top_goster=False: top haritada gösterilmez. Hazır model topu karelerin ~%59'unda bulur, yanlış
        # tespitler olur ve havadaki top homografi yüzünden metrelerce kayar → haritada titrek görünür.
        self.model, self.yolo, self.yumusat, self.top_goster = model, yolo, yumusat, top_goster
        self.filtre, self.takipci = HomografiFiltresi(), OyuncuTakipci(hiz_m=hiz_m)
        self.takim = TakimAtayici(kararli=kararli_takim)
        self.ilk = True

    def isle(self, img: np.ndarray, kesme: bool):
        """Döner: (kal, tespitler (sahadakiler), harita oyuncuları). kal None → saha yok."""
        if kesme:
            self.filtre.sifirla(), self.takim.sifirla(), self.takipci.sifirla()
        kal_ham = ransac_homografi(self.model.tahmin(img))
        kal = self.filtre.guncelle(kal_ham, img.shape) if self.yumusat else kal_ham
        if self.yumusat and self.filtre.yeniden_kuruldu:
            # Homografi yeniden kuruldu / büyük kamera değişimi kabul edildi → eski saha konumları
            # geçersiz: oyuncu konum yumuşatmasını sıfırla (eski yerden yeni yere yavaşça kaymasın).
            self.takipci.sifirla()
        tespitler = oyunculari_takip_et(img, self.yolo, sifirla=self.ilk or kesme)
        self.ilk = False
        if kal is None:
            return None, [], []
        kesik_ayaklari_tahmin_et(tespitler, img.shape[0])        # alt kenarda kesilen oyuncular
        ayak = np.array([t.ayak for t in tespitler]).reshape(-1, 2)
        saha = noktalari_donustur(kal.H, ayak) if len(tespitler) else np.empty((0, 2))
        sahadakiler = []
        for t, (x, y) in zip(tespitler, saha):
            if np.isfinite(x) and -2 <= x <= pitch.SAHA_UZUNLUK + 2 and -2 <= y <= pitch.SAHA_GENISLIK + 2:
                t.saha = (float(x), float(y))
                sahadakiler.append(t)
        self.takim.ata(sahadakiler)
        if self.yumusat:
            oyuncular = self.takipci.guncelle(sahadakiler)
        else:
            oyuncular = [{"x": t.saha[0], "y": t.saha[1], "sinif": t.sinif, "renk": t.ek.get("takim_renk"),
                          "id": t.ek.get("id")} for t in sahadakiler]
        if not self.top_goster:
            oyuncular = [o for o in oyuncular if o["sinif"] != "top"]
        # Her yarı sahada en fazla bir kaleci (kısa süre görünmeyip son yerinde tutulan izler dahil):
        # kale çizgisine en yakın olan kalır, diğerleri gizlenir.
        for sol in (True, False):
            k = [o for o in oyuncular if o["sinif"] == "kaleci" and (o["x"] < pitch.SAHA_UZUNLUK / 2) == sol]
            for o in sorted(k, key=lambda o: o["x"] if sol else pitch.SAHA_UZUNLUK - o["x"])[1:]:
                o["sinif"] = "diger"
        return kal, sahadakiler, oyuncular


def video_isle(model, video: str | Path, cikti: Path, bas: float = 0, sure: float | None = None,
               hedef_fps: float = 12, kesme_esigi: float = 0.35, yolo: str = "yolov8n.pt",
               sadece_harita: bool = False, genislik: int = 1280, yumusat: bool = True,
               top_goster: bool = False, kirp: bool = True, en_kisa_sn: float = 1.0, ilerleme=None) -> dict:
    """
    Videoyu kuş bakışı videoya çevirir.
      hedef_fps     : saniyede kaç kare işlensin (12 → 60 fps videoda her 5. kare)
      yumusat       : homografi filtresi + oyuncu takibi/yumuşatma (False: her kare bağımsız)
      sadece_harita : True → sadece kuş bakışı harita, False → görüntü + harita yan yana
      kirp          : True → saha görünmeyen sahneler (yakın plan, tribün, tekrar) videodan ÇIKARILIR;
                      'en_kisa_sn'den kısa haritalı parçalar da atılır (yanıp sönmesin diye).
                      False → o karelerde "saha görünmüyor" yazılır, video tam uzunlukta kalır.
      ilerleme      : ilerleme(oran 0..1, mesaj) çağrılır (arayüzdeki ilerleme çubuğu için)
    """
    hat = VideoHatti(model, yolo, yumusat, top_goster)
    yazici, n_islenen, n_ok, n_kesme, n_yazilan, bilgi = None, 0, 0, 0, 0, {}
    parca: list[np.ndarray] = []       # kırpma modunda: haritalanan ardışık kareler (bir sahne parçası)
    t0 = time.time()

    def yaz(kare):
        nonlocal yazici, n_yazilan
        if yazici is None:
            olcek = min(1.0, genislik / kare.shape[1])
            w, hh = int(kare.shape[1] * olcek) // 2 * 2, int(kare.shape[0] * olcek) // 2 * 2
            cikti.parent.mkdir(parents=True, exist_ok=True)
            yazici = VideoYazici(cikti, bilgi["fps"] / bilgi["adim"], w, hh)
        yazici.yaz(kare)
        n_yazilan += 1

    def parcayi_bosalt():
        if len(parca) >= en_kisa_sn * hedef_fps:
            for k in parca:
                yaz(k)
        parca.clear()

    for img, kesme, bilgi in video_kareleri(video, bas, sure, hedef_fps, kesme_esigi):
        n_islenen += 1
        n_kesme += kesme
        kal, tespitler, oyuncular = hat.isle(img, kesme)
        if kesme and kirp:
            parcayi_bosalt()
        if kal is None:
            if kirp:
                parcayi_bosalt()
                birlesik = None
            else:
                birlesik = bos_kare(img, "saha gorunmuyor")
        else:
            birlesik, n_ok = video_karesi_ciz(img, kal, tespitler, oyuncular, iz=yumusat), n_ok + 1
        if birlesik is not None:
            if sadece_harita:
                birlesik = birlesik[:, img.shape[1] + ARA:]
            parca.append(birlesik) if kirp else yaz(birlesik)
        n_hedef = bilgi["hedef"]
        if ilerleme and (n_islenen % 5 == 0 or n_islenen == n_hedef):
            gecen = time.time() - t0
            kalan = gecen / n_islenen * max(0, n_hedef - n_islenen)
            ilerleme(min(1.0, n_islenen / n_hedef), f"{n_islenen}/{n_hedef} frames · ~{kalan:.0f} s left")
    if kirp:
        parcayi_bosalt()
    if yazici:
        yazici.kapat()
    return {"islenen": n_islenen, "haritalanan": n_ok, "kesme": int(n_kesme), "sure_sn": time.time() - t0,
            "video_sn": n_islenen * bilgi.get("adim", 1) / bilgi.get("fps", 25),
            "cikti_sn": n_yazilan / hedef_fps if n_islenen else 0,
            "reddedilen_sicrama": hat.filtre.reddedilen, "yol": str(cikti) if yazici else None}


def main():
    ap = argparse.ArgumentParser(description="Uçtan uca otomatik kuş bakışı harita")
    ap.add_argument("--model", required=True, help="saha keypoint modeli (.pt)")
    ap.add_argument("--goruntu", nargs="*", default=[], help="bir veya daha fazla görüntü")
    ap.add_argument("--video", help="video dosyası")
    ap.add_argument("--bas", type=float, default=0, help="videoda başlangıç saniyesi")
    ap.add_argument("--sure", type=float, default=15, help="videodan kaç saniye işlensin")
    ap.add_argument("--fps", type=float, default=12, help="saniyede kaç kare işlensin")
    ap.add_argument("--yumusatma-yok", action="store_true",
                    help="homografi filtresi ve oyuncu yumuşatmasını kapat (her kare bağımsız)")
    ap.add_argument("--top-goster", action="store_true", help="topu videodaki haritada göster")
    ap.add_argument("--kirpma", action="store_true",
                    help="saha görünmeyen sahneleri çıkarma; 'saha görünmüyor' yazarak tam uzunlukta yaz")
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
        s = video_isle(model, args.video, yol, args.bas, args.sure, args.fps, args.kesme_esigi, args.yolo,
                       args.sadece_harita, yumusat=not args.yumusatma_yok, top_goster=args.top_goster,
                       kirp=not args.kirpma,
                       ilerleme=lambda o, m: print(f"\r{m}", end="", flush=True))
        print(f"\nVideo: {s['haritalanan']}/{s['islenen']} kare haritalandı, çıktı {s['cikti_sn']:.0f} sn / "
              f"girdi {s['video_sn']:.0f} sn, {s['kesme']} sahne kesmesi, "
              f"{s['reddedilen_sicrama']} homografi sıçraması reddedildi, {s['sure_sn']:.0f} sn sürdü → {s['yol']}")


if __name__ == "__main__":
    main()
