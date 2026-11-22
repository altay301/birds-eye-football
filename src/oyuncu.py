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
        return ((x1 + x2) / 2, self.ek.get("ayak_y", y2))   # kenarda kesikse tahmin edilen ayak (bkz. aşağı)


def kesik_ayaklari_tahmin_et(tespitler: list[Tespit], goruntu_yuksekligi: int, pay: int = 2) -> int:
    """
    Görüntünün ALT KENARINA değen kutuda ayaklar görünmez: alt kenar ayak değil kesim çizgisidir ve oyuncu
    kameraya olduğundan yakın görünür. Perspektif: kameradan aynı uzaklıktaki oyuncular görüntüde aynı boydadır
    → üst kenarı en yakın 3 TAM görünen kişinin medyan boyu, kesik kişinin boyu kabul edilir.
    Benzetimle ölçüldü (tam kutular yapay kesilip): kesim çizgisi 1.9–3.6 m, bu yöntem 0.5–0.6 m medyan hata
    (src/ayak_kenar_analiz.py). Döner: düzeltilen kişi sayısı.
    """
    kisiler = [t for t in tespitler if t.sinif != "top"]
    kesik = [t for t in kisiler if t.kutu[3] >= goruntu_yuksekligi - pay]
    tam = [t for t in kisiler if t.kutu[3] < goruntu_yuksekligi - pay]
    if not kesik or len(tam) < 3:
        return 0
    ust = np.array([t.kutu[1] for t in tam])
    boy = np.array([t.kutu[3] - t.kutu[1] for t in tam])
    for t in kesik:
        yakin = np.argsort(np.abs(ust - t.kutu[1]))[:3]
        t.ek["ayak_y"] = t.kutu[1] + max(float(np.median(boy[yakin])), t.kutu[3] - t.kutu[1])
    return len(kesik)


_model_onbellek = {}


def _siniflar(model) -> tuple[list[int] | None, dict[int, str]]:
    """
    İki tür model desteklenir:
      * hazır COCO modeli  → sadece 'person' ve 'sports ball' sınıfları; herkes "oyuncu" sayılır
      * kendi modelimiz    → sınıflar doğrudan rol: oyuncu / kaleci / hakem / top
    Döner: (YOLO'ya verilecek sınıf filtresi, sınıf no → bizim sınıf adımız)
    """
    adlar = model.names
    if "person" in adlar.values():
        return [COCO_INSAN, COCO_TOP], {COCO_INSAN: "oyuncu", COCO_TOP: "top"}
    return None, {i: a for i, a in adlar.items()}


def _yukle(anahtar, model_adi):
    from ultralytics import YOLO
    if anahtar not in _model_onbellek:      # modeli bir kez yükle, tekrar kullan
        _model_onbellek[anahtar] = YOLO(str(PROJE / "models" / model_adi))
    return _model_onbellek[anahtar]


def _imgsz(model) -> int:
    """Model eğitildiği çözünürlükte çalıştırılır (en az 1280: küçük oyuncular ve top için)."""
    return max(1280, (getattr(model, "ckpt", None) or {}).get("train_args", {}).get("imgsz", 1280))


def oyunculari_bul(goruntu: np.ndarray, model_adi: str = "yolov8n.pt",
                   guven_esigi: float = 0.25) -> list[Tespit]:
    """
    YOLO ile kişileri (rolleriyle, kendi modelimizde) ve topu bulur.
    imgsz=1280: Görüntüyü büyük işler → uzaktaki küçük oyuncular ve top daha iyi bulunur.
    """
    model = _yukle(model_adi, model_adi)
    filtre, ad = _siniflar(model)
    sonuc = model.predict(goruntu, classes=filtre, conf=guven_esigi, imgsz=_imgsz(model), verbose=False)[0]
    return [Tespit(tuple(map(float, kutu)), float(guven), ad[int(sinif)])
            for kutu, guven, sinif in zip(sonuc.boxes.xyxy.cpu().numpy(), sonuc.boxes.conf.cpu().numpy(),
                                          sonuc.boxes.cls.cpu().numpy())]


def izleyici_ayari(k: float = 1.0) -> str:
    """
    ByteTrack ayar dosyası. Kayıp bir iz 'track_buffer' KARE tutulur (varsayılan 30 = 12 fps'de 2.5 sn).
    k = işlenen fps / 12 → aynı SÜRE için track_buffer = 30·k. k = 1 → hazır dosya.
    """
    if abs(k - 1.0) < 1e-6:
        return "bytetrack.yaml"
    import tempfile
    from pathlib import Path
    import ultralytics
    kaynak = Path(ultralytics.__file__).parent / "cfg" / "trackers" / "bytetrack.yaml"
    metin = kaynak.read_text(encoding="utf-8")
    metin = "\n".join((f"track_buffer: {round(30 * k)}" if s.startswith("track_buffer:") else s)
                      for s in metin.splitlines())
    yol = Path(tempfile.gettempdir()) / f"bytetrack_k{k:.3f}.yaml"
    yol.write_text(metin, encoding="utf-8")
    return str(yol)


def oyunculari_takip_et(goruntu: np.ndarray, model_adi: str = "yolov8n.pt", sifirla: bool = False,
                        guven_esigi: float = 0.1, izleyici: str = "bytetrack.yaml") -> list[Tespit]:
    """
    Videoda tespit + TAKİP (ByteTrack): her tespite kareler boyunca sabit bir kimlik verilir
    (t.ek['id']). ByteTrack, kutuları önceki karelerin tahmini konumlarıyla (Kalman filtresi)
    eşleştirir; düşük güvenli tespitleri de ikinci turda eşleştirerek kısa kapanmalarda kimliği korur.
    sifirla=True → izleyici sıfırlanır (sahne kesmesinden sonra).
    Ayrıca her oyuncunun forma rengi (t.renk_bgr) ölçülür.
    """
    model = _yukle(("takip", model_adi), model_adi)   # takip modeli ayrı tutulur (izleyici durumu modele bağlı)
    filtre, ad = _siniflar(model)
    sonuc = model.track(goruntu, classes=filtre, conf=guven_esigi, imgsz=_imgsz(model),
                        persist=not sifirla, tracker=izleyici, verbose=False)[0]
    b = sonuc.boxes
    kimlik = b.id.int().cpu().numpy() if b.id is not None else [None] * len(b)
    tespitler = []
    for kutu, guven, sinif, tid in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy(), kimlik):
        t = Tespit(tuple(map(float, kutu)), float(guven), ad[int(sinif)])
        t.ek["id"] = None if tid is None else int(tid)
        if t.sinif != "top":       # hakem ve kalecinin de rengi ölçülür (hakem–kaleci ayrımı için)
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
    # ÇİMİ YEREL ÖĞREN: sabit bir "yeşil" aralığı (ör. ton 35–85) sarımsı, solgun ya da gölgeli
    # çimde başarısız olur (bir maçta çim tonu ≈ 32 → çim pikselleri forma rengine karışıyordu).
    # Bunun yerine oyuncunun hemen çevresindeki zeminin medyan rengi çim kabul edilir ve gövdedeki
    # bu renge (Lab uzayında) yakın pikseller atılır.
    H, W = goruntu.shape[:2]
    cx1, cx2 = max(0, x1 - w), min(W, x2 + w)
    cy1, cy2 = max(0, y1 + h // 2), min(H, y2 + h // 3)
    cevre = goruntu[cy1:cy2, cx1:cx2]
    maske = np.ones(cevre.shape[:2], bool)
    maske[max(0, y1 - cy1):max(0, y2 - cy1), max(0, x1 - cx1):max(0, x2 - cx1)] = False   # kutunun içi hariç
    lab = cv2.cvtColor(govde, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    if maske.sum() >= 20:
        cim = np.median(cv2.cvtColor(cevre, cv2.COLOR_BGR2LAB)[maske].astype(np.float32), axis=0)
        forma = np.linalg.norm(lab - cim, axis=1) > 18
    else:                                   # çevre yoksa (kenar) eski sabit yeşil maskesine dön
        hsv = cv2.cvtColor(govde, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        forma = ~((hsv[:, 0] > 25) & (hsv[:, 0] < 85) & (hsv[:, 1] > 50))
    pikseller = govde.reshape(-1, 3)[forma]
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
