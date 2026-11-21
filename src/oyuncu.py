"""
oyuncu.py — Oyuncu tespiti (YOLO) ve forma rengine göre takım ayırma (K-means)
==============================================================================

AYAK NOKTASI
  YOLO her kişi için bir kutu verir: (x1, y1) sol üst, (x2, y2) sağ alt köşe.
  Homografi sadece ZEMİNDEKİ noktalar için doğrudur, bu yüzden kutunun ortasını
  değil alt kenarının ortasını, yani ayakları kullanıyoruz:
        ayak = ((x1 + x2) / 2,  y2)

TAKIM AYIRMA (şimdilik basit yöntem — ileride CNN ile karşılaştırılacak)
  1) Gövdeyi kırp: kutunun üst-orta bölgesi ≈ forma
  2) Yeşil pikselleri at (kutuya çim de girer)
  3) Kalanların ortalama rengini al (Lab renk uzayında)
  4) K-means ile tüm oyuncuları 2 gruba ayır  → takim_A / takim_B
  K-means DENETİMSİZ öğrenmedir: kimse "bu mavi takım" demez, benzerlikten bulur.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

PROJE = Path(__file__).resolve().parent.parent
COCO_INSAN, COCO_TOP = 0, 32   # hazır YOLO modelinde sınıf numaraları


@dataclass
class Tespit:
    kutu: tuple[float, float, float, float]   # x1, y1, x2, y2 (piksel)
    guven: float                              # modelin eminlik skoru (0-1)
    sinif: str                                # "oyuncu" / "top" (sonra takim_A / takim_B)
    renk_bgr: tuple[int, int, int] | None = None   # ölçülen forma rengi
    saha: tuple[float, float] | None = None        # sahadaki konum (metre), sonradan dolar
    ek: dict = field(default_factory=dict)

    @property
    def ayak(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.kutu
        return ((x1 + x2) / 2, y2)


_model_onbellek = {}


def oyunculari_bul(goruntu: np.ndarray, model_adi: str = "yolov8n.pt",
                   guven_esigi: float = 0.25) -> list[Tespit]:
    """
    Hazır YOLO modeliyle insanları ve topu bulur.
    imgsz=1280: Görüntüyü büyük işler → uzaktaki küçük oyuncular daha iyi bulunur
    (varsayılan 640'tır; ekran kartımız buna rahat yeter).
    """
    from ultralytics import YOLO
    if model_adi not in _model_onbellek:      # modeli bir kez yükle, tekrar kullan
        _model_onbellek[model_adi] = YOLO(str(PROJE / "models" / model_adi))
    model = _model_onbellek[model_adi]
    sonuc = model.predict(goruntu, classes=[COCO_INSAN, COCO_TOP], conf=guven_esigi,
                          imgsz=1280, verbose=False)[0]
    tespitler = []
    for kutu, guven, sinif in zip(sonuc.boxes.xyxy.cpu().numpy(),
                                  sonuc.boxes.conf.cpu().numpy(),
                                  sonuc.boxes.cls.cpu().numpy()):
        tespitler.append(Tespit(tuple(map(float, kutu)), float(guven),
                                "top" if int(sinif) == COCO_TOP else "oyuncu"))
    return tespitler


def oyunculari_takip_et(goruntu: np.ndarray, model_adi: str = "yolov8n.pt", sifirla: bool = False,
                        guven_esigi: float = 0.1) -> list[Tespit]:
    """
    Videoda tespit + TAKİP (ByteTrack): her tespite kareler boyunca sabit bir kimlik verilir
    (t.ek['id']). ByteTrack, kutuları önceki karelerin tahmini konumlarıyla (Kalman filtresi)
    eşleştirir; düşük güvenli tespitleri de ikinci turda eşleştirerek kısa kapanmalarda kimliği korur.
    sifirla=True → izleyici sıfırlanır (sahne kesmesinden sonra).
    Ayrıca her oyuncunun forma rengi (t.renk_bgr) ölçülür.
    """
    from ultralytics import YOLO
    anahtar = ("takip", model_adi)
    if anahtar not in _model_onbellek:      # takip modeli ayrı tutulur (izleyici durumu modele bağlı)
        _model_onbellek[anahtar] = YOLO(str(PROJE / "models" / model_adi))
    model = _model_onbellek[anahtar]
    sonuc = model.track(goruntu, classes=[COCO_INSAN, COCO_TOP], conf=guven_esigi, imgsz=1280,
                        persist=not sifirla, tracker="bytetrack.yaml", verbose=False)[0]
    b = sonuc.boxes
    kimlik = b.id.int().cpu().numpy() if b.id is not None else [None] * len(b)
    tespitler = []
    for kutu, guven, sinif, tid in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy(), kimlik):
        t = Tespit(tuple(map(float, kutu)), float(guven), "top" if int(sinif) == COCO_TOP else "oyuncu")
        t.ek["id"] = None if tid is None else int(tid)
        if t.sinif == "oyuncu":
            r = forma_rengi(goruntu, t.kutu)
            if r is not None:
                t.renk_bgr = tuple(int(c) for c in r)
        tespitler.append(t)
    return tespitler


def forma_rengi(goruntu: np.ndarray, kutu) -> np.ndarray | None:
    """Kutunun gövde bölgesindeki yeşil olmayan piksellerin ortalama rengi (BGR)."""
    x1, y1, x2, y2 = [int(round(v)) for v in kutu]
    w, h = x2 - x1, y2 - y1
    # Gövde: yükseklikte %15-%50 arası (kafa ve bacaklar hariç), genişlikte orta %60
    gx1, gx2 = x1 + int(0.2 * w), x2 - int(0.2 * w)
    gy1, gy2 = y1 + int(0.15 * h), y1 + int(0.5 * h)
    govde = goruntu[max(gy1, 0):max(gy2, 0), max(gx1, 0):max(gx2, 0)]
    if govde.size == 0:
        return None
    hsv = cv2.cvtColor(govde, cv2.COLOR_BGR2HSV)
    # OpenCV'de renk tonu (H) 0-180 arasıdır; yeşil ≈ 35-85. Doygunluk (S) düşükse
    # renk grimsi/beyazdır — beyaz formaları kaybetmemek için onları silmiyoruz.
    yesil = (hsv[..., 0] > 35) & (hsv[..., 0] < 85) & (hsv[..., 1] > 50)
    pikseller = govde[~yesil]
    if len(pikseller) < 10:
        return None
    return np.median(pikseller, axis=0)   # ortalama yerine medyan: aykırı piksellere dayanıklı


def takimlara_ayir(goruntu: np.ndarray, tespitler: list[Tespit]) -> None:
    """Oyuncuları forma rengine göre K-means ile 2 takıma ayırır (listeyi yerinde günceller)."""
    oyuncular = [t for t in tespitler if t.sinif == "oyuncu"]
    renkler, gecerli = [], []
    for t in oyuncular:
        r = forma_rengi(goruntu, t.kutu)
        if r is not None:
            t.renk_bgr = tuple(int(c) for c in r)
            renkler.append(r)
            gecerli.append(t)
    if len(gecerli) < 2:
        for t in oyuncular:
            t.sinif = "takim_A"
        return
    # BGR → Lab: Lab'da iki renk arasındaki sayısal fark, gözün gördüğü farka daha yakındır
    lab = cv2.cvtColor(np.uint8(renkler).reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    # OpenCV k-means: sklearn KMeans ile aynı algoritma ama ~20 nokta için çok daha hızlı
    # (sklearn'ün her çağrıdaki sabit maliyeti videoda kare başına ~60 ms ediyordu)
    cv2.setRNGSeed(0)
    kriter = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    _, etiket, _ = cv2.kmeans(lab.astype(np.float32), 2, None, kriter, 10, cv2.KMEANS_PP_CENTERS)
    etiket = etiket.ravel()
    for t, e in zip(gecerli, etiket):
        t.sinif = "takim_A" if e == 0 else "takim_B"
    for t in oyuncular:                 # rengi ölçülemeyenler
        if t.sinif == "oyuncu":
            t.sinif = "takim_A"
    # Mini haritada her takımı kendi gerçek forma rengiyle göstermek için grup ortalaması
    for takim in ("takim_A", "takim_B"):
        grup = [t.renk_bgr for t in gecerli if t.sinif == takim]
        if grup:
            ort = tuple(int(c) for c in np.median(np.array(grup), axis=0))
            for t in oyuncular:
                if t.sinif == takim:
                    t.ek["takim_renk"] = ort
