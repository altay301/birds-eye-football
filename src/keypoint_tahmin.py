"""
keypoint_tahmin.py — Eğitilmiş saha keypoint modelini gerçek karelerde kullanır
==============================================================================

Üç iş yapar:
  --oneri       : data\\frames içindeki tüm karelere tahmin yap, etiketleme aracı için
                  öneri dosyaları yaz (data\\labels\\oneri_noktalar\\...). Etiketlerken
                  'A' ile kabul edip sadece yanlışları düzeltirsin.
  --gorsel N    : N rastgele kare için tahmin edilen noktalar + (RANSAC homografisiyle)
                  saha çizgileri çizilmiş bir pano üret → modelin gözle kontrolü.
  --degerlendir : Elle etiketlenmiş karelerde modelin hatasını ölç (piksel, PCK).

PCK (Percentage of Correct Keypoints): tahmin, gerçek noktaya E piksel'den yakınsa
"doğru" sayılır; doğru noktaların oranı. Keypoint modellerinin standart metriği.

Kullanım:
    .venv\\Scripts\\python.exe src\\keypoint_tahmin.py --model models\\saha_kp_sentetik_v1.pt --gorsel 15
    .venv\\Scripts\\python.exe src\\keypoint_tahmin.py --model models\\saha_kp_sentetik_v1.pt --oneri
    .venv\\Scripts\\python.exe src\\keypoint_tahmin.py --model models\\saha_kp_sentetik_v1.pt --degerlendir
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import pitch
from homografi import cizgileri_goruntuye_ciz, dejenere_mi, homografi_hesapla, noktalari_donustur

PROJE = Path(__file__).resolve().parent.parent
KARE_KLASOR = PROJE / "data" / "frames"
ETIKET_KLASOR = PROJE / "data" / "labels" / "noktalar"
ONERI_KLASOR = PROJE / "data" / "labels" / "oneri_noktalar"


def oku(yol: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(yol), np.uint8), cv2.IMREAD_COLOR)


class SahaModeli:
    def __init__(self, model_yolu: str, imgsz: int | None = None, esik: float = 0.5):
        from ultralytics import YOLO
        self.model = YOLO(model_yolu)
        # Tahmin, modelin EĞİTİLDİĞİ çözünürlükte yapılmalı (640'ta eğitilen model 960'ta
        # farklı ölçekte nesneler görür). Verilmezse modelin kendi eğitim ayarından okunur.
        if imgsz is None:
            imgsz = (getattr(self.model, "ckpt", None) or {}).get("train_args", {}).get("imgsz", 640)
        self.imgsz, self.esik = imgsz, esik

    def tahmin(self, img: np.ndarray) -> list[dict]:
        """Döner: [{"id", "px": [x, y], "guven"}] — sadece güveni eşiği geçen, görüntü içindekiler."""
        s = self.model.predict(img, imgsz=self.imgsz, verbose=False, conf=0.1)[0]
        if s.keypoints is None or len(s.keypoints) == 0:
            return []
        # Birden fazla "saha" bulunduysa en güvenlisini al
        en_iyi = int(s.boxes.conf.argmax())
        xy = s.keypoints.xy[en_iyi].cpu().numpy()
        guv = s.keypoints.conf[en_iyi].cpu().numpy()
        h, w = img.shape[:2]
        sonuc = []
        for i, ((x, y), g) in enumerate(zip(xy, guv)):
            if g >= self.esik and 0 <= x < w and 0 <= y < h:
                sonuc.append({"id": i, "px": [round(float(x), 1), round(float(y), 1)], "guven": round(float(g), 3)})
        return sonuc


def model_yukle(yol: str, imgsz: int | None = None, esik: float = 0.5):
    """Dosya adında 'unet' geçiyorsa kendi U-Net modelimiz, değilse YOLO-pose. Arayüz aynı."""
    if "unet" in Path(yol).name.lower():
        from unet_keypoint import UNetSahaModeli
        return UNetSahaModeli(yol, esik)
    return SahaModeli(yol, imgsz, esik)


def ransac_homografi(noktalar: list[dict]):
    """Model tahminlerinden homografi (yanlış noktaları RANSAC ayıklar). Olmazsa None."""
    if len(noktalar) < 4:
        return None
    try:
        return homografi_hesapla(np.array([n["px"] for n in noktalar]),
                                 pitch.NOKTA_KOORD[[n["id"] for n in noktalar]], ransac=True)
    except ValueError:
        return None


def ciz(img: np.ndarray, noktalar: list[dict], gercek: list[dict] | None = None) -> np.ndarray:
    out = img.copy()
    kal = ransac_homografi(noktalar)
    if kal is not None:
        out = cizgileri_goruntuye_ciz(out, kal.H_ters, renk=(0, 255, 255), kalinlik=2)
    for g in gercek or []:   # elle etiket: yeşil artı
        p = tuple(np.round(g["px"]).astype(int))
        cv2.drawMarker(out, p, (0, 255, 0), cv2.MARKER_CROSS, 22, 3)
    for n in noktalar:       # tahmin: kırmızı nokta + ID
        p = tuple(np.round(n["px"]).astype(int))
        cv2.circle(out, p, 7, (0, 0, 255), -1)
        cv2.putText(out, str(n["id"]), (p[0] + 8, p[1] - 6), 0, 0.9, (0, 0, 0), 4)
        cv2.putText(out, str(n["id"]), (p[0] + 8, p[1] - 6), 0, 0.9, (255, 255, 255), 2)
    yazi = f"{len(noktalar)} nokta | " + (f"homografi OK" if kal else "homografi YOK")
    cv2.putText(out, yazi, (15, 45), 0, 1.2, (0, 0, 0), 5)
    cv2.putText(out, yazi, (15, 45), 0, 1.2, (255, 255, 255), 2)
    return out


def gorsel_pano(model: SahaModeli, kareler: list[Path], cikti: Path):
    kucuk = []
    for k in kareler:
        img = oku(k)
        kucuk.append(cv2.resize(ciz(img, model.tahmin(img)), (480, 270)))
    while len(kucuk) % 3:
        kucuk.append(np.zeros_like(kucuk[0]))
    pano = np.vstack([np.hstack(kucuk[i:i + 3]) for i in range(0, len(kucuk), 3)])
    cikti.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(cikti), pano)
    print("Pano:", cikti)


def onerileri_yaz(model: SahaModeli, kareler: list[Path]):
    n = 0
    for k in kareler:
        img = oku(k)
        yol = ONERI_KLASOR / k.parent.name / f"{k.stem}.json"
        yol.parent.mkdir(parents=True, exist_ok=True)
        yol.write_text(json.dumps({"goruntu": k.name, "noktalar": model.tahmin(img)}, ensure_ascii=False),
                       encoding="utf-8")
        n += 1
    print(f"{n} kare için öneri yazıldı → {ONERI_KLASOR}")


def degerlendir(model: SahaModeli, ciftler: list[tuple[Path, Path]], esikler=(5, 10, 20)):
    """ciftler: (görüntü, etiket json). Sadece elle işaretlenmiş noktalar değerlendirilir."""
    hatalar, bulunan, toplam = [], 0, 0
    # Uçtan uca ölçüm: modelin noktalarından (RANSAC) kurulan homografi, ELLE etiketlenen
    # noktaları sahada doğru yere taşıyor mu? (metre cinsinden — projenin asıl amacı bu)
    h_kare, h_basari, metre_hatalari = 0, 0, []
    for g, e in ciftler:
        gercek = json.loads(e.read_text(encoding="utf-8"))
        if gercek.get("durum") == "atlandi" or not gercek["noktalar"]:
            continue
        img = oku(g)
        olcek = 1280 / img.shape[1]   # hatayı 1280 genişliğe göre ölç (karşılaştırılabilir olsun)
        tahminler = model.tahmin(img)
        tahmin = {n["id"]: n["px"] for n in tahminler}
        for n in gercek["noktalar"]:
            toplam += 1
            if n["id"] in tahmin:
                bulunan += 1
                hatalar.append(np.hypot(*(np.subtract(tahmin[n["id"]], n["px"]))) * olcek)
        # Etiket kendi başına homografi kurabiliyorsa (≥4, eşdoğrusal değil) bu kare sayılır
        ids = [n["id"] for n in gercek["noktalar"]]
        if len(ids) >= 4 and dejenere_mi(pitch.NOKTA_KOORD[ids]) is None:
            h_kare += 1
            kal = ransac_homografi(tahminler)
            if kal is not None:
                m = noktalari_donustur(kal.H, np.array([n["px"] for n in gercek["noktalar"]]))
                hata = np.linalg.norm(m - pitch.NOKTA_KOORD[ids], axis=1)
                if np.isfinite(hata).all():
                    h_basari += 1
                    metre_hatalari.append(float(np.mean(hata)))
    if not toplam:
        print("Değerlendirilecek etiket yok.")
        return None
    h = np.array(hatalar)
    sonuc = {"kare": len(ciftler), "etiketli_nokta": toplam, "bulunan_oran": bulunan / toplam,
             "medyan_hata_px": float(np.median(h)) if len(h) else None}
    for e_ in esikler:
        sonuc[f"PCK@{e_}px"] = float((h <= e_).sum() / toplam)
    if h_kare:
        mh = np.array(metre_hatalari)
        sonuc.update({
            "homografi_kare": h_kare,
            "homografi_kurulan_oran": h_basari / h_kare,
            "metre_hata_medyan": float(np.median(mh)) if len(mh) else None,
            # Başarı: homografi kuruldu VE ortalama hata < 1 m (oyuncu konumu için kullanılabilir)
            "basari_1m_oran": float((mh < 1.0).sum() / h_kare),
        })
    print(json.dumps(sonuc, indent=2, ensure_ascii=False))
    return sonuc


def main():
    ap = argparse.ArgumentParser(description="Saha keypoint modeliyle tahmin")
    ap.add_argument("--model", required=True)
    ap.add_argument("--esik", type=float, default=0.5, help="nokta güven eşiği")
    ap.add_argument("--imgsz", type=int, default=None, help="boş = modelin eğitim çözünürlüğü")
    ap.add_argument("--oneri", action="store_true")
    ap.add_argument("--gorsel", type=int, default=0)
    ap.add_argument("--degerlendir", action="store_true")
    ap.add_argument("--kume", default="hepsi", choices=["hepsi", "train", "val", "test"],
                    help="değerlendirmeyi hangi kümede yap (bolme.json)")
    ap.add_argument("--cikti", default=str(PROJE / "results" / "keypoint_tahmin.jpg"))
    args = ap.parse_args()

    model = model_yukle(args.model, args.imgsz, args.esik)
    kareler = sorted(KARE_KLASOR.glob("*/*.jpg"))
    if args.gorsel:
        rng = np.random.default_rng(0)
        secim = [kareler[i] for i in sorted(rng.choice(len(kareler), min(args.gorsel, len(kareler)), replace=False))]
        gorsel_pano(model, secim, Path(args.cikti))
    if args.oneri:
        onerileri_yaz(model, kareler)
    if args.degerlendir:
        bolme = json.loads((PROJE / "data" / "bolme.json").read_text(encoding="utf-8"))
        ciftler = [(KARE_KLASOR / e.parent.name / f"{e.stem}.jpg", e) for e in sorted(ETIKET_KLASOR.glob("*/*.json"))
                   if args.kume == "hepsi" or bolme.get(e.parent.name) == args.kume]
        if args.kume == "hepsi":
            ciftler += [(PROJE / "data" / "ornekler" / "ornek2.webp", PROJE / "data" / "kalibrasyon" / "ornek2.json")]
        print(f"Küme: {args.kume}")
        degerlendir(model, [c for c in ciftler if c[0].exists()])


if __name__ == "__main__":
    main()
