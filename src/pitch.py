"""
pitch.py — Futbol sahası şablonu ve kuş bakışı mini harita
==========================================================

Bu dosya projenin "referans dünyası"dır:
  * Sahanın gerçek ölçüleri (metre)
  * Etiketleme ve keypoint modelinde kullanılacak 35 anahtar nokta
  * Kuş bakışı mini haritayı çizen fonksiyonlar

KOORDİNAT SİSTEMİ (metre)
--------------------------
    (0,0) ─────────────── x (uzunluk, 0..105) ──────────────► (105,0)
      │   ┌──────┐                                 ┌──────┐   │
      │   │ sol  │               ( o )             │ sağ  │   │
      y   │ kale │                                 │ kale │   │
      │   └──────┘                                 └──────┘   │
    (0,68) ──────────────────────────────────────────────── (105,68)

  * x: sol kale çizgisinden sağa doğru (0 → 105)
  * y: yayın kamerasının KARŞI tarafındaki taç çizgisinden (uzak kenar, y=0)
       kameraya yakın taç çizgisine doğru (yakın kenar, y=68)
  Yani yayın görüntüsünde ekranın üstü ≈ y=0, altı ≈ y=68 olur.

Çalıştırma (demo görseli üretir):
    .venv\\Scripts\\python.exe src\\pitch.py
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 1) SAHA ÖLÇÜLERİ (FIFA / UEFA standardı, metre)
# ---------------------------------------------------------------------------
SAHA_UZUNLUK = 105.0          # x ekseni
SAHA_GENISLIK = 68.0          # y ekseni
CEZA_DERINLIK = 16.5          # ceza sahası kale çizgisinden içeri
CEZA_GENISLIK = 40.32
ALTILIK_DERINLIK = 5.5        # kale alanı (6'lık)
ALTILIK_GENISLIK = 18.32
PENALTI_MESAFE = 11.0         # penaltı noktasının kale çizgisine uzaklığı
ORTA_YUVARLAK_R = 9.15        # orta yuvarlak ve ceza yayı yarıçapı
KALE_GENISLIK = 7.32

_ORTA_X = SAHA_UZUNLUK / 2    # 52.5
_ORTA_Y = SAHA_GENISLIK / 2   # 34.0
_CEZA_Y1 = _ORTA_Y - CEZA_GENISLIK / 2        # 13.84
_CEZA_Y2 = _ORTA_Y + CEZA_GENISLIK / 2        # 54.16
_ALTI_Y1 = _ORTA_Y - ALTILIK_GENISLIK / 2     # 24.84
_ALTI_Y2 = _ORTA_Y + ALTILIK_GENISLIK / 2     # 43.16
# Ceza yayının ceza sahası çizgisini kestiği noktalar (Pisagor):
#   yay merkezi penaltı noktası (11, 34), yarıçap 9.15, çizgi x = 16.5
_YAY_DY = math.sqrt(ORTA_YUVARLAK_R**2 - (CEZA_DERINLIK - PENALTI_MESAFE) ** 2)  # ≈ 7.31


# ---------------------------------------------------------------------------
# 2) ANAHTAR NOKTALAR
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AnahtarNokta:
    id: int          # 0'dan başlar (modelin çıkış kanalı = id)
    ad: str          # etiketleme aracında görünecek isim
    x: float         # metre
    y: float         # metre


def _sol_yari() -> list[tuple[str, float, float]]:
    """Sol yarı sahadaki 14 nokta. Sağ yarı bunun aynası olarak üretilir."""
    return [
        ("kose_ust",               0.0,            0.0),
        ("ceza_ust_kalecizgi",     0.0,            _CEZA_Y1),
        ("altilik_ust_kalecizgi",  0.0,            _ALTI_Y1),
        ("altilik_alt_kalecizgi",  0.0,            _ALTI_Y2),
        ("ceza_alt_kalecizgi",     0.0,            _CEZA_Y2),
        ("kose_alt",               0.0,            SAHA_GENISLIK),
        ("altilik_ust_ic",         ALTILIK_DERINLIK, _ALTI_Y1),
        ("altilik_alt_ic",         ALTILIK_DERINLIK, _ALTI_Y2),
        ("ceza_ust_ic",            CEZA_DERINLIK,  _CEZA_Y1),
        ("ceza_alt_ic",            CEZA_DERINLIK,  _CEZA_Y2),
        ("yay_ust",                CEZA_DERINLIK,  _ORTA_Y - _YAY_DY),
        ("yay_alt",                CEZA_DERINLIK,  _ORTA_Y + _YAY_DY),
        ("penalti_noktasi",        PENALTI_MESAFE, _ORTA_Y),
        # Ceza yayının en uç noktası. Yakın çekimlerde 8-10-11 noktaları aynı
        # doğru (x=16.5) üzerinde kalır; bu nokta o doğrunun DIŞINDA olduğu için
        # homografiyi mümkün kılar.
        ("yay_tepe",               PENALTI_MESAFE + ORTA_YUVARLAK_R, _ORTA_Y),
    ]


def _anahtar_noktalari_olustur() -> list[AnahtarNokta]:
    liste: list[tuple[str, float, float]] = []
    # Sol yarı (id 0-13)
    liste += [(f"sol_{ad}", x, y) for ad, x, y in _sol_yari()]
    # Orta çizgi + orta yuvarlak (id 14-20)
    # NOT: orta çizgi üzerindeki noktalar aynı doğru üzerindedir (homografi için
    # tek başına yetmez). Yuvarlağın sol/sağ uçları bu yüzden önemlidir.
    liste += [
        ("orta_cizgi_ust",     _ORTA_X, 0.0),
        ("orta_yuvarlak_ust",  _ORTA_X, _ORTA_Y - ORTA_YUVARLAK_R),
        ("orta_nokta",         _ORTA_X, _ORTA_Y),
        ("orta_yuvarlak_alt",  _ORTA_X, _ORTA_Y + ORTA_YUVARLAK_R),
        ("orta_cizgi_alt",     _ORTA_X, SAHA_GENISLIK),
        ("orta_yuvarlak_sol",  _ORTA_X - ORTA_YUVARLAK_R, _ORTA_Y),
        ("orta_yuvarlak_sag",  _ORTA_X + ORTA_YUVARLAK_R, _ORTA_Y),
    ]
    # Sağ yarı (id 21-34): sol yarının x'e göre aynası
    liste += [(f"sag_{ad}", SAHA_UZUNLUK - x, y) for ad, x, y in _sol_yari()]
    return [AnahtarNokta(i, ad, round(x, 2), round(y, 2)) for i, (ad, x, y) in enumerate(liste)]


ANAHTAR_NOKTALAR: list[AnahtarNokta] = _anahtar_noktalari_olustur()
NOKTA_SAYISI = len(ANAHTAR_NOKTALAR)  # 35

# (35, 2) numpy dizisi: homografi hesabında doğrudan kullanılır
NOKTA_KOORD = np.array([[k.x, k.y] for k in ANAHTAR_NOKTALAR], dtype=np.float32)


def _yatay_cevirme_eslesmesi() -> list[int]:
    """
    Görüntü yatay çevrildiğinde (data augmentation) sol kale sağ kale olur.
    Bu yüzden keypoint ID'leri de yer değiştirmelidir:  x  →  105 - x
    Dönen liste: FLIP_IDX[i] = çevrildikten sonra i. noktanın yeni ID'si.
    (Ultralytics YOLO-pose'un 'flip_idx' parametresi tam olarak budur.)
    """
    esles = []
    for k in ANAHTAR_NOKTALAR:
        hedef = (round(SAHA_UZUNLUK - k.x, 2), k.y)
        j = next(m.id for m in ANAHTAR_NOKTALAR if (m.x, m.y) == hedef)
        esles.append(j)
    return esles


FLIP_IDX: list[int] = _yatay_cevirme_eslesmesi()


# ---------------------------------------------------------------------------
# 3) MİNİ HARİTA ÇİZİMİ
# ---------------------------------------------------------------------------
@dataclass
class HaritaAyar:
    olcek: int = 8               # 1 metre = kaç piksel (8 → 840x544 saha)
    kenar: int = 30              # saha etrafındaki boşluk (piksel)
    cim: tuple = (60, 140, 60)   # BGR renkler (OpenCV BGR kullanır)
    cim_koyu: tuple = (52, 125, 52)
    cizgi: tuple = (255, 255, 255)
    kalinlik: int = 2


def metre_to_piksel(x: float, y: float, ayar: HaritaAyar = HaritaAyar()) -> tuple[int, int]:
    """Saha koordinatını (metre) mini harita pikseline çevirir."""
    return (int(round(ayar.kenar + x * ayar.olcek)),
            int(round(ayar.kenar + y * ayar.olcek)))


def saha_ciz(ayar: HaritaAyar = HaritaAyar()) -> np.ndarray:
    """Boş kuş bakışı saha görüntüsü döndürür (BGR, numpy dizisi)."""
    s, k = ayar.olcek, ayar.kenar
    W = int(SAHA_UZUNLUK * s + 2 * k)
    H = int(SAHA_GENISLIK * s + 2 * k)
    img = np.full((H, W, 3), ayar.cim, dtype=np.uint8)

    # Biçilmiş çim şeritleri (sadece görsel)
    serit = SAHA_UZUNLUK / 14
    for i in range(0, 14, 2):
        p1 = metre_to_piksel(i * serit, 0, ayar)
        p2 = metre_to_piksel((i + 1) * serit, SAHA_GENISLIK, ayar)
        cv2.rectangle(img, p1, p2, ayar.cim_koyu, -1)

    P = lambda x, y: metre_to_piksel(x, y, ayar)
    c, t = ayar.cizgi, ayar.kalinlik
    aa = cv2.LINE_AA

    # Dış çizgiler ve orta çizgi
    cv2.rectangle(img, P(0, 0), P(SAHA_UZUNLUK, SAHA_GENISLIK), c, t, aa)
    cv2.line(img, P(_ORTA_X, 0), P(_ORTA_X, SAHA_GENISLIK), c, t, aa)
    # Orta yuvarlak + orta nokta
    cv2.circle(img, P(_ORTA_X, _ORTA_Y), int(ORTA_YUVARLAK_R * s), c, t, aa)
    cv2.circle(img, P(_ORTA_X, _ORTA_Y), 3, c, -1, aa)

    for taraf in (0, 1):  # 0 = sol, 1 = sağ
        ayna = (lambda x: x) if taraf == 0 else (lambda x: SAHA_UZUNLUK - x)
        # Ceza sahası ve 6'lık
        cv2.rectangle(img, P(ayna(0), _CEZA_Y1), P(ayna(CEZA_DERINLIK), _CEZA_Y2), c, t, aa)
        cv2.rectangle(img, P(ayna(0), _ALTI_Y1), P(ayna(ALTILIK_DERINLIK), _ALTI_Y2), c, t, aa)
        # Penaltı noktası
        cv2.circle(img, P(ayna(PENALTI_MESAFE), _ORTA_Y), 3, c, -1, aa)
        # Ceza yayı: penaltı noktası merkezli, ceza sahası dışında kalan kısım
        aci = math.degrees(math.acos((CEZA_DERINLIK - PENALTI_MESAFE) / ORTA_YUVARLAK_R))  # ≈ 53°
        bas, bit = (-aci, aci) if taraf == 0 else (180 - aci, 180 + aci)
        cv2.ellipse(img, P(ayna(PENALTI_MESAFE), _ORTA_Y),
                    (int(ORTA_YUVARLAK_R * s),) * 2, 0, bas, bit, c, t, aa)
        # Kale (çizginin dışına küçük dikdörtgen)
        derin = -2.0 if taraf == 0 else 2.0
        cv2.rectangle(img, P(ayna(0), _ORTA_Y - KALE_GENISLIK / 2),
                      P(ayna(0) + derin, _ORTA_Y + KALE_GENISLIK / 2), c, t, aa)
    return img


def saha_cizgileri(adim: float = 0.5) -> list[np.ndarray]:
    """
    Sahadaki tüm çizgileri METRE cinsinden nokta dizileri (polyline) olarak verir.
    Kalibrasyonu kontrol etmek için bu çizgiler homografi ile maç görüntüsüne
    izdüşürülür: görüntüdeki gerçek çizgilerle örtüşüyorsa kalibrasyon doğrudur.

    Düz çizgileri de 'adim' metre aralıklarla örnekliyoruz. Neden? Perspektifte
    düz çizgi düz kalır ama çizginin bir ucu kameranın arkasına düşerse tek
    parça çizmek bozuk sonuç verir; küçük parçalara bölmek bunu önler.
    """
    def dogru(x1, y1, x2, y2):
        n = max(2, int(math.hypot(x2 - x1, y2 - y1) / adim) + 1)
        return np.stack([np.linspace(x1, x2, n), np.linspace(y1, y2, n)], axis=1)

    def yay(cx, cy, r, bas_derece, bit_derece):
        n = max(8, int(math.radians(abs(bit_derece - bas_derece)) * r / adim))
        a = np.radians(np.linspace(bas_derece, bit_derece, n))
        return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)

    L, W = SAHA_UZUNLUK, SAHA_GENISLIK
    cizgiler = [
        dogru(0, 0, L, 0), dogru(L, 0, L, W), dogru(L, W, 0, W), dogru(0, W, 0, 0),
        dogru(_ORTA_X, 0, _ORTA_X, W),
        yay(_ORTA_X, _ORTA_Y, ORTA_YUVARLAK_R, 0, 360),
    ]
    aci = math.degrees(math.acos((CEZA_DERINLIK - PENALTI_MESAFE) / ORTA_YUVARLAK_R))
    for ayna, isaret in ((lambda x: x, 1), (lambda x: L - x, -1)):
        for d, y1, y2 in ((CEZA_DERINLIK, _CEZA_Y1, _CEZA_Y2),
                          (ALTILIK_DERINLIK, _ALTI_Y1, _ALTI_Y2)):
            cizgiler += [dogru(ayna(0), y1, ayna(d), y1),
                         dogru(ayna(d), y1, ayna(d), y2),
                         dogru(ayna(d), y2, ayna(0), y2)]
        merkez = 0 if isaret == 1 else 180
        cizgiler.append(yay(ayna(PENALTI_MESAFE), _ORTA_Y, ORTA_YUVARLAK_R,
                            merkez - aci, merkez + aci))
    return [c.astype(np.float32) for c in cizgiler]


# Takım renkleri (BGR)
TAKIM_RENK = {
    "takim_A": (200, 80, 20),    # mavi
    "takim_B": (240, 240, 240),  # beyaz
    "kaleci":  (0, 200, 255),    # sarı/turuncu
    "hakem":   (20, 20, 20),     # siyah
    "top":     (0, 0, 255),      # kırmızı
}


# Kuş bakışı haritada GÖSTERİLMEYEN sınıflar: hakem ve yan hakemler oyunun parçası değil,
# haritada sadece kalabalık yapar. Görüntü tarafında kutuları yine çizilir (tespit kontrolü için).
HARITADA_GIZLI = {"hakem"}


def oyunculari_ciz(img: np.ndarray, oyuncular: list[dict],
                   ayar: HaritaAyar = HaritaAyar()) -> np.ndarray:
    """
    Mini haritaya oyuncuları çizer.
    oyuncular: [{"x": 30.2, "y": 40.1, "sinif": "takim_A", "no": 7}, ...]
               x, y metre cinsindendir. 'no' isteğe bağlıdır.
               'renk' (BGR) verilirse sınıf rengi yerine o kullanılır
               (ör. görüntüden ölçülen gerçek forma rengi).
    """
    out = img.copy()
    for o in oyuncular:
        if o.get("sinif") in HARITADA_GIZLI:
            continue
        p = metre_to_piksel(o["x"], o["y"], ayar)
        renk = o.get("renk") or TAKIM_RENK.get(o.get("sinif", "takim_A"), (128, 128, 128))
        renk = tuple(int(c) for c in renk)
        r = 5 if o.get("sinif") == "top" else 9
        cv2.circle(out, p, r, renk, -1, cv2.LINE_AA)
        cv2.circle(out, p, r, (0, 0, 0), 1, cv2.LINE_AA)
        if "no" in o:
            cv2.putText(out, str(o["no"]), (p[0] - 6, p[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def anahtar_noktalari_ciz(img: np.ndarray, ayar: HaritaAyar = HaritaAyar(),
                          etiket: bool = True) -> np.ndarray:
    """Şablondaki 35 noktayı ID'leriyle çizer (etiketleme kılavuzu için)."""
    out = img.copy()
    for k in ANAHTAR_NOKTALAR:
        p = metre_to_piksel(k.x, k.y, ayar)
        cv2.circle(out, p, 6, (0, 0, 255), -1, cv2.LINE_AA)
        if etiket:
            # Sağ kale çizgisindeki noktaların yazısı saha dışındaki boşluğa yazılır
            # (içeri yazılırsa 6'lık noktalarının yazısıyla çakışıyor)
            dx = 8
            dy = -8 if k.y < 3 else (18 if k.y > SAHA_GENISLIK - 3 else -6)
            cv2.putText(out, str(k.id), (p[0] + dx, p[1] + dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, str(k.id), (p[0] + dx, p[1] + dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# 4) ŞABLONU DIŞA AKTARMA (etiketleme aracı ve eğitim için)
# ---------------------------------------------------------------------------
def sablonu_kaydet(yol: Path) -> None:
    """Anahtar nokta şablonunu JSON olarak kaydeder."""
    veri = {
        "aciklama": "Futbol sahasi anahtar noktalari. Koordinatlar metre; "
                    "x: sol kale cizgisinden saga (0-105), y: uzak tac cizgisinden "
                    "kameraya yakin tac cizgisine (0-68).",
        "saha": {"uzunluk": SAHA_UZUNLUK, "genislik": SAHA_GENISLIK},
        "nokta_sayisi": NOKTA_SAYISI,
        "flip_idx": FLIP_IDX,
        "noktalar": [{"id": k.id, "ad": k.ad, "x": k.x, "y": k.y} for k in ANAHTAR_NOKTALAR],
    }
    yol.parent.mkdir(parents=True, exist_ok=True)
    yol.write_text(json.dumps(veri, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    PROJE = Path(__file__).resolve().parent.parent
    cikti = PROJE / "results" / "pitch_demo"
    cikti.mkdir(parents=True, exist_ok=True)

    # a) Boş saha
    bos = saha_ciz()
    cv2.imwrite(str(cikti / "1_bos_saha.png"), bos)

    # b) Anahtar noktalar + ID'ler (etiketleme kılavuzunda kullanılacak)
    cv2.imwrite(str(cikti / "2_anahtar_noktalar.png"), anahtar_noktalari_ciz(bos))

    # c) Örnek kadro: 4-3-3'e karşı 4-4-2 (sadece çizimi göstermek için uydurma konumlar)
    A = [(4, 34), (18, 12), (16, 27), (16, 41), (18, 56), (32, 22), (30, 34), (32, 46),
         (46, 14), (50, 34), (46, 54)]
    B = [(101, 34), (86, 14), (88, 28), (88, 40), (86, 54), (72, 12), (70, 28), (70, 40),
         (72, 56), (58, 30), (57, 40)]
    oyuncular = ([{"x": x, "y": y, "sinif": "takim_A", "no": i + 1} for i, (x, y) in enumerate(A)]
                 + [{"x": x, "y": y, "sinif": "takim_B", "no": i + 1} for i, (x, y) in enumerate(B)])
    oyuncular[0]["sinif"] = "kaleci"
    oyuncular[11]["sinif"] = "kaleci"
    oyuncular += [{"x": 55, "y": 24, "sinif": "hakem"}, {"x": 51.5, "y": 34.5, "sinif": "top"}]
    cv2.imwrite(str(cikti / "3_ornek_kadro.png"), oyunculari_ciz(bos, oyuncular))

    # d) JSON şablonu
    sablonu_kaydet(PROJE / "data" / "saha_anahtar_noktalari.json")

    # Konsola tablo
    print(f"{NOKTA_SAYISI} anahtar nokta:")
    for k in ANAHTAR_NOKTALAR:
        print(f"  {k.id:2d}  {k.ad:28s} ({k.x:6.2f}, {k.y:5.2f})  çevirince → {FLIP_IDX[k.id]}")
    print("Görseller:", cikti)
