"""
manuel_kalibrasyon.py — Elle kalibrasyon aracı (uçtan uca ilk prototip)
=======================================================================

Ne yapar?
  1) Maç görüntüsünü açar, yanında numaralı saha şablonunu gösterir.
  2) Sen görünen saha noktalarına tıklayıp numarasını girersin.
  3) 4+ nokta olunca homografi hesaplanır, saha çizgileri canlı olarak çizilir.
  4) Boşluk tuşu: YOLO oyuncuları bulur, takımlara ayırır, ayak noktalarını
     sahaya taşır ve kuş bakışı mini haritayı gösterir.

Kullanım (proje klasöründe):
    .venv\\Scripts\\python.exe src\\manuel_kalibrasyon.py data\\ornekler\\ornek2.webp

    --otomatik : Pencere açmadan, daha önce kaydedilmiş noktalarla sonucu üretir.

Tuşlar (pencere seçiliyken):
    Fare tekerleği   imlecin olduğu yere yakınlaştır / uzaklaştır (16 kata kadar)
    Sağ tuş sürükle  yakınlaştırılmış görüntüde gezin       R   görünümü sıfırla
    Sol tık          nokta işaretle      0-9 + Enter   numarasını gir
    Noktayı sürükle  yerini düzelt       D             imlece en yakın noktayı sil
    A                model önerilerini (turuncu halkalar) kabul et
    Backspace        rakam sil           Esc           bekleyen tıklamayı iptal
    Z                son noktayı sil     Boşluk        oyuncuları haritaya yerleştir
    S                kaydet              Q             çık (otomatik kaydeder)

Kaydedilenler:
    data\\kalibrasyon\\<görüntü_adı>.json      tıklanan noktalar (ileride etiket formatı)
    results\\kalibrasyon\\<görüntü_adı>.png    yan yana sonuç görseli
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

import pitch
from homografi import cizgileri_goruntuye_ciz, homografi_hesapla, isareti_duzelt, noktalari_donustur
from oyuncu import oyunculari_bul, takimlara_ayir

PROJE = Path(__file__).resolve().parent.parent
NOKTA_KLASOR = PROJE / "data" / "kalibrasyon"
SONUC_KLASOR = PROJE / "results" / "kalibrasyon"
HATALI_NOKTA_ESIGI_M = 1.5   # bir noktanın hatası bundan büyükse muhtemelen yanlış ID


# ---------------------------------------------------------------------------
# Dosya işlemleri
# ---------------------------------------------------------------------------
def goruntu_oku(yol: Path) -> np.ndarray:
    # cv2.imread Türkçe karakterli yollarda sorun çıkarabildiği için bu yöntem daha güvenli
    veri = np.fromfile(str(yol), dtype=np.uint8)
    img = cv2.imdecode(veri, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"Görüntü okunamadı: {yol}")
    return img


def noktalari_yukle(yol: Path) -> list[dict]:
    if yol.exists():
        return json.loads(yol.read_text(encoding="utf-8"))["noktalar"]
    return []


def durum_yukle(yol: Path) -> str | None:
    """Etiket dosyasındaki durum: 'etiketli', 'atlandi' ya da None (hiç açılmamış)."""
    if yol.exists():
        return json.loads(yol.read_text(encoding="utf-8")).get("durum")
    return None


def noktalari_kaydet(yol: Path, goruntu_yolu: Path, noktalar: list[dict],
                     durum: str | None = None, ek: dict | None = None) -> None:
    yol.parent.mkdir(parents=True, exist_ok=True)
    veri = {"goruntu": goruntu_yolu.name, "noktalar": noktalar}
    if durum:
        veri["durum"] = durum
    veri.update(ek or {})   # ör. {"yontem": "onerisiz"} — raporda etiket kaynağını ayırmak için
    yol.write_text(json.dumps(veri, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Hesap: noktalar → homografi → oyuncular → mini harita
# ---------------------------------------------------------------------------
def kalibre_et(noktalar: list[dict]):
    """Noktalardan homografi hesaplar. Yetersizse (None, hata_mesajı) döner."""
    if len(noktalar) < 4:
        return None, f"{len(noktalar)} nokta var, en az 4 gerekli"
    piksel = np.array([n["px"] for n in noktalar], dtype=np.float64)
    metre = pitch.NOKTA_KOORD[[n["id"] for n in noktalar]]
    try:
        return homografi_hesapla(piksel, metre), None
    except ValueError as e:
        return None, str(e)


def sonuc_uret(goruntu: np.ndarray, noktalar: list[dict], kal=None,
               yolo: str = "yolov8n.pt") -> tuple[np.ndarray, list, object]:
    """
    Uçtan uca: homografi + YOLO + takım + izdüşüm → yan yana sonuç görseli.
    kal verilirse (ör. model tahmini + RANSAC ile hesaplanmış) noktalardan yeniden hesaplanmaz.
    """
    if kal is None:
        kal, hata = kalibre_et(noktalar)
        if kal is None:
            raise ValueError(hata)

    tespitler = oyunculari_bul(goruntu, yolo)
    takimlara_ayir(goruntu, tespitler)

    # Ayak noktalarını sahaya taşı; saha dışına düşenleri (tribün, yedek kulübesi) at
    ayaklar = np.array([t.ayak for t in tespitler]).reshape(-1, 2)
    saha_konum = noktalari_donustur(kal.H, ayaklar) if len(tespitler) else np.empty((0, 2))
    sahadakiler = []
    for t, (x, y) in zip(tespitler, saha_konum):
        if np.isnan(x) or not (-2 <= x <= pitch.SAHA_UZUNLUK + 2 and -2 <= y <= pitch.SAHA_GENISLIK + 2):
            continue
        t.saha = (float(x), float(y))
        # Kaleci kendi ceza sahası çevresinde durur: kaleye 25 m'den uzakta "kaleci" denen kişi
        # büyük olasılıkla hakemdir → hakem (haritada gizli, pitch.HARITADA_GIZLI)
        if t.sinif == "kaleci" and 25 < x < pitch.SAHA_UZUNLUK - 25:
            t.sinif = "hakem"
        sahadakiler.append(t)

    # --- Sol: maç görüntüsü + saha çizgileri + kutular ---
    sol = cizgileri_goruntuye_ciz(goruntu, kal.H_ters)
    for t in sahadakiler:
        x1, y1, x2, y2 = map(int, t.kutu)
        renk = t.ek.get("takim_renk", pitch.TAKIM_RENK.get(t.sinif, (0, 0, 255)))
        cv2.rectangle(sol, (x1, y1), (x2, y2), renk, 2)
        cv2.rectangle(sol, (x1, y1), (x2, y2), (0, 0, 0), 1)
        cv2.circle(sol, tuple(map(int, t.ayak)), 4, (0, 0, 255), -1)   # ayak noktası

    # --- Sağ: kuş bakışı mini harita ---
    harita = pitch.saha_ciz()
    harita = pitch.oyunculari_ciz(harita, [
        {"x": t.saha[0], "y": t.saha[1], "sinif": t.sinif, "renk": t.ek.get("takim_renk")}
        for t in sahadakiler])
    # Kameranın gördüğü alanı haritada göster (görüntünün 4 köşesi → saha)
    h, w = goruntu.shape[:2]
    gorus = noktalari_donustur(kal.H, [[0, h], [w, h], [w, 0], [0, 0]])
    if not np.isnan(gorus).any():
        poligon = np.array([pitch.metre_to_piksel(x, y) for x, y in
                            np.clip(gorus, [-50, -50], [155, 118])], np.int32)
        katman = harita.copy()
        cv2.fillPoly(katman, [poligon], (255, 255, 255))
        harita = cv2.addWeighted(katman, 0.15, harita, 0.85, 0)

    # Yan yana birleştir (yükseklikleri eşitle)
    olcek = sol.shape[0] / harita.shape[0]
    harita = cv2.resize(harita, (int(harita.shape[1] * olcek), sol.shape[0]))
    birlesik = np.hstack([sol, np.full((sol.shape[0], 10, 3), 255, np.uint8), harita])
    return birlesik, sahadakiler, kal


# ---------------------------------------------------------------------------
# Etkileşimli pencere
# ---------------------------------------------------------------------------
class Arayuz:
    PENCERE = "Mac goruntusu"
    SABLON = "Saha sablonu (nokta numaralari)"
    SONUC = "Sonuc"
    SERIT = 56   # görüntünün üstündeki bilgi şeridinin yüksekliği (piksel)

    def __init__(self, goruntu_yolu: Path, json_yol: Path | None = None,
                 oneri: list[dict] | None = None):
        self.yol = goruntu_yolu
        self.goruntu = goruntu_oku(goruntu_yolu)
        self.json_yol = json_yol or NOKTA_KLASOR / f"{goruntu_yolu.stem}.json"
        self.noktalar = noktalari_yukle(self.json_yol)
        # Öneriler: halka olarak gösterilir, 'A' ile kabul edilir.
        #   "model"     → keypoint modelinin tahmini (turuncu)
        #   "homografi" → SENİN noktalarından hesaplanan homografinin tahmini (camgöbeği, H tuşu)
        self.oneri = oneri or []
        self.oneri_turu = "model"
        self.tasinan = None         # sol tuşla sürüklenen noktanın sırası
        self.baslik_ek = ""         # bilgi şeridinin başına eklenecek yazı (ör. "Kare 12/437")
        self.tus_yardim = "[Teker zoom | Sag tus kaydir | R sifirla | Z geri | D sil | Bosluk harita | S kaydet | Q cik]"
        self.bekleyen = None        # tıklanmış ama numarası girilmemiş nokta (orijinal piksel)
        self.yazilan = ""           # klavyeden girilen rakamlar
        self.mesaj = "Bir saha noktasina tikla, numarasini yaz, Enter."
        # --- GÖRÜNÜM DÖNÜŞÜMÜ (yakınlaştırma / kaydırma) ---
        # Ekrandaki bir piksel (ex, ey) ile orijinal görüntü pikseli (ox, oy) arasında:
        #     ox = gx + ex / s        ey = (oy - gy) * s
        # gx, gy : ekranın sol üstünde görünen orijinal piksel (görünüm başlangıcı)
        # s      : toplam ölçek = ekrana sığdırma ölçeği × yakınlaştırma
        h, w = self.goruntu.shape[:2]
        self.olcek = min(1.0, 1280 / w, 800 / h)   # ekrana sığdırma ölçeği (sabit)
        self.ekran_w, self.ekran_h = int(w * self.olcek), int(h * self.olcek)
        self.zoom = 1.0                              # 1 = tüm görüntü, 8 = 8 kat büyük
        self.gx, self.gy = 0.0, 0.0
        self.fare_ekran = None                       # artı imleci için farenin yeri
        self.surukle = None                          # sağ tuşla kaydırma başlangıcı

    # ---- Görünüm dönüşümü yardımcıları ----
    @property
    def s(self) -> float:
        return self.olcek * self.zoom

    def ekran_to_orijinal(self, ex: float, ey: float) -> tuple[float, float]:
        return self.gx + ex / self.s, self.gy + ey / self.s

    def orijinal_to_ekran(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        return (p - [self.gx, self.gy]) * self.s

    def _gorunumu_sinirla(self):
        """Görünüm görüntünün dışına taşmasın."""
        h, w = self.goruntu.shape[:2]
        self.gx = float(np.clip(self.gx, 0, w - self.ekran_w / self.s))
        self.gy = float(np.clip(self.gy, 0, h - self.ekran_h / self.s))

    def fare(self, olay, x, y, bayraklar, *_):
        # Pencerede görüntünün üstünde bilgi şeridi var: fare y'si şeridi de içerir.
        y -= self.SERIT
        self.fare_ekran = (x, y) if y >= 0 else None

        if olay == cv2.EVENT_MOUSEWHEEL and y >= 0:
            # İmlecin altındaki orijinal nokta, yakınlaştırmadan sonra da imlecin altında kalsın
            ox, oy = self.ekran_to_orijinal(x, y)
            # Tekerlek yönü 'bayraklar'ın üst 16 bitinde işaretli sayı olarak gelir:
            # yukarı → pozitif, aşağı → negatif (işaret biti 1 olur)
            yukari = bayraklar > 0
            self.zoom = float(np.clip(self.zoom * (1.25 if yukari else 0.8), 1.0, 16.0))
            self.gx, self.gy = ox - x / self.s, oy - y / self.s
            self._gorunumu_sinirla()
        elif olay == cv2.EVENT_RBUTTONDOWN:
            self.surukle = (x, y, self.gx, self.gy)
        elif olay == cv2.EVENT_MOUSEMOVE and self.surukle and (bayraklar & cv2.EVENT_FLAG_RBUTTON):
            x0, y0, gx0, gy0 = self.surukle
            self.gx, self.gy = gx0 - (x - x0) / self.s, gy0 - (y - y0) / self.s
            self._gorunumu_sinirla()
        elif olay == cv2.EVENT_RBUTTONUP:
            self.surukle = None
        elif olay == cv2.EVENT_LBUTTONDOWN and y >= 0:
            # Var olan bir noktanın üstüne tıklandıysa: o noktayı sürükle
            yakin = self.en_yakin_nokta(x, y, esik_ekran=10)
            if yakin is not None:
                self.tasinan = yakin
                self.mesaj = f"Tasiniyor: {self.noktalar[yakin]['id']}"
            else:
                self.bekleyen = self.ekran_to_orijinal(x, y)
                self.yazilan = ""
                self.mesaj = "Numara yaz + Enter (iptal: Esc)"
        elif olay == cv2.EVENT_MOUSEMOVE and self.tasinan is not None and (bayraklar & cv2.EVENT_FLAG_LBUTTON):
            ox, oy = self.ekran_to_orijinal(x, y)
            self.noktalar[self.tasinan]["px"] = [round(ox, 1), round(oy, 1)]
        elif olay == cv2.EVENT_LBUTTONUP and self.tasinan is not None:
            self.mesaj = f"Tasindi: {self.noktalar[self.tasinan]['id']}"
            self.tasinan = None

    def en_yakin_nokta(self, ex: float, ey: float, esik_ekran: float) -> int | None:
        """Ekran koordinatına en yakın noktanın sırası (eşikten uzaksa None)."""
        if not self.noktalar:
            return None
        p = self.orijinal_to_ekran([n["px"] for n in self.noktalar]).reshape(-1, 2)
        d = np.hypot(p[:, 0] - ex, p[:, 1] - ey)
        i = int(np.argmin(d))
        return i if d[i] <= esik_ekran else None

    def onerileri_kabul_et(self):
        """Model önerilerinden, henüz elle girilmemiş ID'leri noktalara ekler."""
        var = {n["id"] for n in self.noktalar}
        eklenen = [dict(o) for o in self.oneri if o["id"] not in var]
        self.noktalar += [{"id": o["id"], "ad": pitch.ANAHTAR_NOKTALAR[o["id"]].ad,
                           "px": [round(o["px"][0], 1), round(o["px"][1], 1)],
                           "kaynak": self.oneri_turu} for o in eklenen]
        self.mesaj = f"{len(eklenen)} oneri kabul edildi. Yanlis olanlari surukle (D: sil)."
        if self.oneri_turu == "homografi":
            self.oneri = []

    def homografi_onerisi(self):
        """
        H tuşu: işaretlediğin noktalardan homografi kur, şablondaki DİĞER noktaların
        görüntüde nereye düşmesi gerektiğini öneri olarak göster.
        Hiçbir modelden gelmez → test etiketlerini bir modele doğru KAYDIRMAZ.
        Ama homografi kapanmayı bilmez: oyuncunun/tabelanın arkasındaki noktaları da
        önerir → onları kabul ettikten sonra D ile sil. Uzak noktalarda birkaç piksel
        kayabilir (lens bükülmesi) → yakınlaştırıp sürükleyerek düzelt.
        """
        kal, hata = kalibre_et(self.noktalar)
        if kal is None:
            self.mesaj = f"H onerisi icin once 4+ net nokta isaretle ({hata})"
            return
        h, w = self.goruntu.shape[:2]
        var = {n["id"] for n in self.noktalar}
        p = noktalari_donustur(kal.H_ters, pitch.NOKTA_KOORD)
        icerde = (~np.isnan(p).any(axis=1) & (p[:, 0] >= 0) & (p[:, 0] < w)
                  & (p[:, 1] >= 0) & (p[:, 1] < h))
        # BELİRSİZLİK: tıklamalarındaki ~1.5 px'lik titreme, işaretlediğin noktalardan UZAK
        # yerlerde büyür (dışdeğerleme / extrapolation: 5 noktadan 50-100 px kayma görüldü).
        # Tıklamaları 40 kez rastgele titretip her noktanın nereye düştüğüne bakıyoruz;
        # yeri 3 px'ten fazla oynayan noktaları ÖNERMİYORUZ (yanlış öneri, hiç öneriden kötü).
        rng = np.random.default_rng(0)
        g = np.array([n["px"] for n in self.noktalar], dtype=np.float64)
        s_m = pitch.NOKTA_KOORD[[n["id"] for n in self.noktalar]]
        ornekler = []
        for _ in range(40):
            gt = g + rng.normal(0, 1.5, g.shape)
            Hs, _ = cv2.findHomography(gt, s_m, 0)            # görüntü → saha (titretilmiş)
            if Hs is not None:
                Hs = isareti_duzelt(Hs, gt)                       # w > 0 olsun (bkz. homografi.py)
                ornekler.append(noktalari_donustur(np.linalg.inv(Hs), pitch.NOKTA_KOORD))
        with warnings.catch_warnings():   # kameranın arkasındaki noktalar hep NaN → uyarı verir, önemsiz
            warnings.simplefilter("ignore", RuntimeWarning)
            oynama = np.nanstd(np.array(ornekler), axis=0).max(axis=1) if ornekler else np.full(len(p), np.inf)
        oynama = np.nan_to_num(oynama, nan=np.inf)
        guvenli = icerde & (oynama < 3.0)
        self.oneri = [{"id": i, "px": [float(p[i, 0]), float(p[i, 1])]}
                      for i in range(pitch.NOKTA_SAYISI) if guvenli[i] and i not in var]
        belirsiz = int((icerde & ~guvenli).sum() - sum(1 for i in var if icerde[i] and not guvenli[i]))
        self.oneri_turu = "homografi"
        self.mesaj = (f"{len(self.oneri)} geometrik oneri" + (f" ({belirsiz} uzak nokta belirsiz: elle ekle ya da "
                      "daha fazla nokta koyup tekrar H)" if belirsiz > 0 else "") +
                      ". A: kabul, kapalilari D ile sil, kaymislari surukle.")

    def ciz(self) -> np.ndarray:
        # 1) Görünen bölgeyi kırp + büyüt (tek adımda, alt-piksel hassasiyetle)
        M = np.float32([[self.s, 0, -self.gx * self.s], [0, self.s, -self.gy * self.s]])
        img = cv2.warpAffine(self.goruntu, M, (self.ekran_w, self.ekran_h),
                             flags=cv2.INTER_LINEAR)

        # 2) Çizimleri EKRAN koordinatında yap: yakınlaştırınca çizgiler kalınlaşıp
        #    altındaki görüntüyü kapatmasın diye
        kal, hata = kalibre_et(self.noktalar)
        if kal is not None:
            for cizgi in pitch.saha_cizgileri():
                p = self.orijinal_to_ekran(noktalari_donustur(kal.H_ters, cizgi))
                for a, b in zip(p[:-1], p[1:]):
                    if np.isnan(a).any() or np.isnan(b).any():
                        continue
                    if (np.abs(a) > 20000).any() or (np.abs(b) > 20000).any():
                        continue
                    cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)),
                             (0, 255, 255), 1, cv2.LINE_AA)
        # Model önerileri (henüz kabul edilmemiş olanlar): turuncu halka
        var = {n["id"] for n in self.noktalar}
        oneri_renk = (255, 255, 0) if self.oneri_turu == "homografi" else (0, 140, 255)
        for o in self.oneri:
            if o["id"] in var:
                continue
            p = tuple(np.round(self.orijinal_to_ekran(o["px"])).astype(int))
            cv2.circle(img, p, 7, oneri_renk, 2, cv2.LINE_AA)
            cv2.putText(img, str(o["id"]), (p[0] + 9, p[1] + 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, oneri_renk, 1, cv2.LINE_AA)
        for i, n in enumerate(self.noktalar):
            p = tuple(np.round(self.orijinal_to_ekran(n["px"])).astype(int))
            kotu = kal is not None and kal.nokta_hatalari[i] > HATALI_NOKTA_ESIGI_M
            renk = (0, 0, 255) if kotu else (0, 200, 0)
            # Yakınlaştırınca nokta küçük çizilir ki tam merkezini görebilesin
            r = 6 if self.zoom < 2 else 3
            cv2.circle(img, p, r, renk, -1, cv2.LINE_AA)
            cv2.circle(img, p, r, (0, 0, 0), 1, cv2.LINE_AA)
            yazi = str(n["id"]) + (" ?" if kotu else "")
            cv2.putText(img, yazi, (p[0] + 9, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(img, yazi, (p[0] + 9, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if self.bekleyen:
            p = tuple(np.round(self.orijinal_to_ekran(self.bekleyen)).astype(int))
            cv2.drawMarker(img, p, (0, 255, 255), cv2.MARKER_CROSS, 24, 1)
            cv2.putText(img, f"#{self.yazilan}_", (p[0] + 12, p[1] + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        # İnce artı imleci: farenin tam olarak nereyi gösterdiği
        if self.fare_ekran:
            fx, fy = self.fare_ekran
            cv2.line(img, (fx - 15, fy), (fx - 4, fy), (255, 0, 255), 1)
            cv2.line(img, (fx + 4, fy), (fx + 15, fy), (255, 0, 255), 1)
            cv2.line(img, (fx, fy - 15), (fx, fy - 4), (255, 0, 255), 1)
            cv2.line(img, (fx, fy + 4), (fx, fy + 15), (255, 0, 255), 1)

        # 3) Üst bilgi şeridi
        durum = (self.baslik_ek + f"Nokta: {len(self.noktalar)} | " +
                 (f"Hata: {kal.hata_metre:.2f} m / {kal.hata_piksel:.1f} px" if kal else hata) +
                 f" | Zoom: {self.zoom:.1f}x")
        if self.fare_ekran:
            ox, oy = self.ekran_to_orijinal(*self.fare_ekran)
            durum += f" | Piksel: ({ox:.1f}, {oy:.1f})"
        serit = np.zeros((self.SERIT, img.shape[1], 3), np.uint8)
        cv2.putText(serit, durum, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(serit, self.mesaj + "  " + self.tus_yardim,
                    (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        return np.vstack([serit, img])

    def tus(self, k: int) -> bool:
        """Tuşa göre işlem yapar. Çıkılacaksa False döner."""
        if self.bekleyen is not None:
            if ord("0") <= k <= ord("9"):
                self.yazilan += chr(k)
                return True
            if k == 8:                       # Backspace
                self.yazilan = self.yazilan[:-1]
                return True
            if k == 27:                      # Esc
                self.bekleyen, self.mesaj = None, "Iptal edildi."
                return True
            if k == 13:                      # Enter
                if self.yazilan.isdigit() and int(self.yazilan) < pitch.NOKTA_SAYISI:
                    nid = int(self.yazilan)
                    # Aynı numara daha önce girildiyse eskisini değiştir
                    self.noktalar = [n for n in self.noktalar if n["id"] != nid]
                    self.noktalar.append({"id": nid, "ad": pitch.ANAHTAR_NOKTALAR[nid].ad,
                                          "px": [round(self.bekleyen[0], 1), round(self.bekleyen[1], 1)]})
                    self.mesaj = f"Eklendi: {nid} ({pitch.ANAHTAR_NOKTALAR[nid].ad})"
                    self.bekleyen = None
                else:
                    self.mesaj = f"Gecersiz numara (0-{pitch.NOKTA_SAYISI - 1} olmali)"
                return True
            return True                      # numara yazarken diğer tuşlar (Z, D...) iş yapmasın
        if k in (ord("h"), ord("H")):        # homografi önerisi
            self.homografi_onerisi()
        elif k in (ord("z"), ord("Z")) and self.noktalar:
            silinen = self.noktalar.pop()
            self.mesaj = f"Silindi: {silinen['id']}"
        elif k in (ord("d"), ord("D")) and self.fare_ekran:   # imlece en yakın noktayı sil
            i = self.en_yakin_nokta(*self.fare_ekran, esik_ekran=25)
            if i is not None:
                self.mesaj = f"Silindi: {self.noktalar.pop(i)['id']}"
        elif k in (ord("a"), ord("A")) and self.oneri:        # model önerilerini kabul et
            self.onerileri_kabul_et()
        elif k in (ord("r"), ord("R")):      # görünümü sıfırla
            self.zoom, self.gx, self.gy = 1.0, 0.0, 0.0
            self.mesaj = "Gorunum sifirlandi."
        elif k == 32:                        # Boşluk
            self.sonucu_goster()
        elif k in (ord("s"), ord("S")):
            self.kaydet(sonuc_da=True)
        elif k in (ord("q"), ord("Q")):
            return False
        return True

    def sonucu_goster(self):
        try:
            self.mesaj = "Oyuncular araniyor..."
            self.son_sonuc, sahadakiler, _ = sonuc_uret(self.goruntu, self.noktalar)
        except ValueError as e:
            self.mesaj = f"Olmadi: {e}"
            return
        self.mesaj = f"Haritada {len(sahadakiler)} kisi. S ile kaydet."
        g = self.son_sonuc
        o = min(1.0, 1600 / g.shape[1])
        cv2.imshow(self.SONUC, cv2.resize(g, None, fx=o, fy=o))

    def kaydet(self, sonuc_da: bool = False):
        noktalari_kaydet(self.json_yol, self.yol, self.noktalar)
        self.mesaj = f"Noktalar kaydedildi: data/kalibrasyon/{self.json_yol.name}"
        if sonuc_da:
            try:
                birlesik, _, _ = sonuc_uret(self.goruntu, self.noktalar)
                SONUC_KLASOR.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(SONUC_KLASOR / f"{self.yol.stem}.png"), birlesik)
                self.mesaj += " + sonuc gorseli"
            except ValueError as e:
                self.mesaj += f" (sonuc yok: {e})"

    def calistir(self):
        cv2.namedWindow(self.PENCERE, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.PENCERE, self.fare)
        # Numara bakmak için şablonu ayrı pencerede göster
        sablon = pitch.anahtar_noktalari_ciz(pitch.saha_ciz(pitch.HaritaAyar(olcek=7)),
                                             pitch.HaritaAyar(olcek=7))
        cv2.imshow(self.SABLON, sablon)
        while True:
            cv2.imshow(self.PENCERE, self.ciz())
            k = cv2.waitKey(30) & 0xFF
            if k != 255 and not self.tus(k):
                break
            # Pencere X ile kapatıldıysa çık
            if cv2.getWindowProperty(self.PENCERE, cv2.WND_PROP_VISIBLE) < 1:
                break
        self.kaydet()
        cv2.destroyAllWindows()
        print(f"Noktalar kaydedildi: {self.json_yol}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Elle saha kalibrasyonu")
    ap.add_argument("goruntu", type=Path, help="maç görüntüsünün yolu")
    ap.add_argument("--otomatik", action="store_true",
                    help="pencere açmadan kayıtlı noktalarla sonucu üret")
    args = ap.parse_args()

    if args.otomatik:
        img = goruntu_oku(args.goruntu)
        noktalar = noktalari_yukle(NOKTA_KLASOR / f"{args.goruntu.stem}.json")
        try:
            birlesik, sahadakiler, kal = sonuc_uret(img, noktalar)
        except ValueError as e:
            sys.exit(f"Kalibrasyon yapılamadı: {e}")
        SONUC_KLASOR.mkdir(parents=True, exist_ok=True)
        cikti = SONUC_KLASOR / f"{args.goruntu.stem}.png"
        cv2.imwrite(str(cikti), birlesik)
        print(f"Kalibrasyon hatası: {kal.hata_metre:.2f} m | {kal.hata_piksel:.1f} piksel")
        for n, e in zip(noktalar, kal.nokta_hatalari):
            uyari = "   <-- yanlış numara olabilir" if e > HATALI_NOKTA_ESIGI_M else ""
            print(f"  nokta {n['id']:2d} {n['ad']:28s} hata {e:5.2f} m{uyari}")
        print(f"Sahadaki kişi sayısı: {len(sahadakiler)}")
        for t in sorted(sahadakiler, key=lambda t: t.saha[0]):
            print(f"  {t.sinif:8s} x={t.saha[0]:6.1f} m  y={t.saha[1]:5.1f} m  (güven {t.guven:.2f})")
        print("Sonuç görseli:", cikti)
    else:
        Arayuz(args.goruntu).calistir()


if __name__ == "__main__":
    main()
