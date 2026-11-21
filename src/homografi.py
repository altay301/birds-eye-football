"""
homografi.py — Görüntü (piksel) ile saha (metre) arasındaki çeviri kuralı
=========================================================================

HOMOGRAFİ NEDİR?
  İki düz yüzey (düzlem) arasındaki perspektif dönüşümüdür. 3x3'lük bir matris (H)
  ile ifade edilir. Bizim iki düzlemimiz:
      * Kameranın görüntüsü  (piksel)
      * Futbol sahası zemini (metre)

  H matrisi kameranın açısını, eğimini, yüksekliğini ve zoom'unu tek seferde
  "kodlar". 9 sayısı var ama ölçek serbest olduğu için 8 bilinmeyen vardır.
  Her nokta eşleşmesi 2 denklem verir (x ve y)  →  en az 4 nokta gerekir.

HOMOJEN KOORDİNATLAR (perspektifin sırrı)
  Bir noktayı dönüştürmek için:
      1) (x, y)  →  (x, y, 1)               sonuna 1 ekle
      2) [X, Y, w] = H @ [x, y, 1]          matrisle çarp
      3) (X / w, Y / w)                     w'ye böl
  3. adımdaki bölme, uzaktaki şeylerin küçük görünmesini (perspektifi) sağlar.
  w <= 0 çıkarsa nokta kameranın ARKASINDA demektir; görüntüye çizilemez.

Bu dosyada:
  homografi_hesapla()       → eşleşmelerden H'yi bulur (+ hata ölçer)
  noktalari_donustur()      → noktaları H ile taşır
  dejenere_mi()             → noktalar homografi için yeterli mi?
  cizgileri_goruntuye_ciz() → saha çizgilerini maç görüntüsüne çizer (kontrol)
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np

from pitch import saha_cizgileri


@dataclass
class HomografiSonuc:
    H: np.ndarray            # görüntü (piksel)  →  saha (metre)
    H_ters: np.ndarray       # saha (metre)      →  görüntü (piksel)
    hata_metre: float        # tıklanan noktalar sahaya taşınınca ortalama sapma
    hata_piksel: float       # şablon noktaları görüntüye taşınınca ortalama sapma
    nokta_hatalari: np.ndarray  # her noktanın ayrı hatası (metre) — yanlış ID'yi bulmak için
    icerdeki: np.ndarray | None = None   # RANSAC'ın kabul ettiği noktalar (True/False)


def noktalari_donustur(H: np.ndarray, noktalar: np.ndarray) -> np.ndarray:
    """
    Noktaları (N, 2) H ile dönüştürür. Yukarıdaki 3 adımı açıkça yapıyoruz
    (cv2.perspectiveTransform da aynısını yapar ama burada görmek öğretici).
    Kameranın arkasına düşen noktalar NaN (sayı değil) olarak döner.
    """
    noktalar = np.asarray(noktalar, dtype=np.float64).reshape(-1, 2)
    homojen = np.hstack([noktalar, np.ones((len(noktalar), 1))])   # (N, 3): sonuna 1 ekle
    sonuc = homojen @ H.T                                             # (N, 3): matrisle çarp
    w = sonuc[:, 2:3]
    out = sonuc[:, :2] / w                                            # w'ye böl
    out[w[:, 0] <= 1e-9] = np.nan                                     # kameranın arkası
    return out


def isareti_duzelt(H: np.ndarray, goruntu_noktalari: np.ndarray) -> np.ndarray:
    """
    H ile -H AYNI dönüşümdür (ölçek serbest, eksi de bir ölçek). Ama w'nin İŞARETİNE
    bakarak "kameranın arkası" kararı verdiğimiz için işaret önemli: sahadaki
    noktalar için w > 0 olmalı.
    cv2.findHomography H[2,2] = 1 yapar → görüntünün (0,0) köşesinde w = 1 > 0 olur.
    Ufuk görüntüdeyse (0,0) tribündedir, yani "ufkun öbür tarafı" → sahadaki
    bütün noktalar w < 0 çıkar ve hepsi NaN olur (hata: 'nan m', çizgi yok).
    Çözüm: bilinen saha noktalarında w çoğunlukla negatifse H'yi -1 ile çarp.
    """
    g = np.asarray(goruntu_noktalari, dtype=np.float64).reshape(-1, 2)
    w = g @ H[2, :2] + H[2, 2]
    return -H if np.median(w) < 0 else H


def dejenere_mi(saha_noktalari: np.ndarray, tolerans_m: float = 0.5) -> str | None:
    """
    Homografi için noktalar yeterli mi? Sorun varsa açıklama, yoksa None döner.

    Kural: Noktaların hepsi (ya da biri hariç hepsi) aynı doğru üzerindeyse
    düzlem belirlenemez. Örnek: sadece orta çizgi üzerindeki 14-15-16-17 noktaları.
    """
    p = np.asarray(saha_noktalari, dtype=np.float64)
    n = len(p)
    if n < 4:
        return f"En az 4 nokta gerekli, şu an {n} tane var."
    # Her nokta çifti bir doğru tanımlar; o doğruya yakın kaç nokta var sayalım.
    for i, j in combinations(range(n), 2):
        d = p[j] - p[i]
        uzunluk = np.hypot(*d)
        if uzunluk < 1e-6:
            return "Aynı saha noktası iki kez işaretlenmiş."
        # Her noktanın bu doğruya dik uzaklığı (2D vektörel çarpım formülü)
        uzaklik = np.abs(d[0] * (p[:, 1] - p[i, 1]) - d[1] * (p[:, 0] - p[i, 0])) / uzunluk
        dogru_uzerinde = int((uzaklik < tolerans_m).sum())
        if dogru_uzerinde >= n - 1:
            return (f"{n} noktanın {dogru_uzerinde} tanesi aynı doğru üzerinde. "
                    "Bu doğrunun DIŞINDA en az 2 nokta daha işaretle.")
    return None


def homografi_hesapla(goruntu_noktalari: np.ndarray, saha_noktalari: np.ndarray,
                      ransac: bool = False) -> HomografiSonuc:
    """
    goruntu_noktalari: (N, 2) piksel   — tıkladığın yerler
    saha_noktalari:    (N, 2) metre    — o noktaların şablondaki gerçek yerleri

    ransac=False: Tüm noktaları kullanan en küçük kareler çözümü. Elle dikkatli
                  tıklanan az sayıda nokta için en iyisi.
    ransac=True : Bazı noktalar YANLIŞSA (ör. modelin hatalı tahmini) onları
                  otomatik ayıklar. İleride keypoint modeliyle kullanacağız.
    """
    g = np.asarray(goruntu_noktalari, dtype=np.float64)
    s = np.asarray(saha_noktalari, dtype=np.float64)

    sorun = dejenere_mi(s)
    if sorun:
        raise ValueError(sorun)

    yontem = cv2.RANSAC if ransac else 0
    H, maske = cv2.findHomography(g, s, yontem, 1.0)   # 1.0 = RANSAC eşiği (metre)
    if H is None:
        raise ValueError("Homografi hesaplanamadı (noktalar tutarsız olabilir).")
    icerdeki = maske.ravel().astype(bool) if maske is not None else np.ones(len(g), bool)
    H = isareti_duzelt(H, g[icerdeki])          # ufuk görüntüdeyse işaret ters gelir
    H_ters = np.linalg.inv(H)
    if ransac:
        # RANSAC yanlış noktaları attıktan sonra kalanlar YİNE kontrol edilmeli:
        # (1) kalanlar eşdoğrusal olabilir (ornek1: 8-10-11 aynı çizgide + 13 → saçma H)
        # (2) tam 4 nokta her zaman "sıfır hatayla" uyar → doğrulama imkânı yok; en az 5 iste
        if icerdeki.sum() < 5:
            raise ValueError(f"RANSAC'tan sonra {int(icerdeki.sum())} tutarlı nokta kaldı (en az 5 gerekli).")
        sorun = dejenere_mi(s[icerdeki])
        if sorun:
            raise ValueError("RANSAC'ın kabul ettiği noktalar: " + sorun)

    # HATA ÖLÇÜMÜ — kalibrasyon ne kadar iyi?
    # 1) Tıklanan pikselleri sahaya taşı, şablondaki gerçek yerle karşılaştır (metre)
    nokta_hatalari = np.linalg.norm(noktalari_donustur(H, g) - s, axis=1)
    # 2) Şablon noktalarını görüntüye taşı, tıklanan yerle karşılaştır (piksel)
    piksel_hatalari = np.linalg.norm(noktalari_donustur(H_ters, s) - g, axis=1)
    # Ortalama hata SADECE RANSAC'ın kabul ettiği noktalardan: atılan (yanlış) noktalar
    # zaten modele katılmadı; onları ortalamaya katmak hatayı yapay olarak şişirir
    # (ilk denemede bu yüzden doğru bir kalibrasyon "9.7 m hata" diye raporlanmıştı).
    return HomografiSonuc(H, H_ters, float(nokta_hatalari[icerdeki].mean()),
                          float(piksel_hatalari[icerdeki].mean()), nokta_hatalari, icerdeki)


def cizgileri_goruntuye_ciz(goruntu: np.ndarray, H_ters: np.ndarray,
                            renk=(0, 255, 255), kalinlik: int = 2) -> np.ndarray:
    """
    Saha çizgilerini (metre) H_ters ile görüntüye izdüşürüp çizer.
    Çizilen sarı çizgiler gerçek beyaz çizgilerin üstüne oturuyorsa
    kalibrasyon doğrudur — en hızlı görsel kontrol budur.
    """
    out = goruntu.copy()
    h, w = out.shape[:2]
    for cizgi in saha_cizgileri():
        p = noktalari_donustur(H_ters, cizgi)
        # Ardışık iki nokta da geçerli ve makul alandaysa aralarını çiz
        for a, b in zip(p[:-1], p[1:]):
            if np.isnan(a).any() or np.isnan(b).any():
                continue
            if (np.abs(a) > 4 * max(w, h)).any() or (np.abs(b) > 4 * max(w, h)).any():
                continue
            cv2.line(out, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)),
                     renk, kalinlik, cv2.LINE_AA)
    return out
