"""
takip.py — Videoda zamansal tutarlılık: homografi filtresi, oyuncu takibi, kalıcı takım ataması
==============================================================================================

Tek kare yöntemi her kareyi bağımsız işler → videoda üç tür titreme oluşur:
  1) HOMOGRAFİ SIÇRAMASI: bir karede noktalar kötü bulunursa H birden kayar, bütün harita zıplar.
  2) OYUNCU TİTREMESİ   : kutular kareden kareye birkaç piksel oynar; uzakta 1 px = birkaç metre.
  3) TAKIM RENGİ        : K-means her karede yeniden çalışır; renk ve hakem ataması kareden kareye değişir.

Çözümler:
  1) HomografiFiltresi
     * H'ler matris olarak ortalanmaz (ölçek/işaret belirsiz, geometrik anlamı yok). Bunun yerine
       görüntüdeki sabit bir NOKTA IZGARASININ sahadaki karşılıkları ortalanır ve H bu ortalanmış
       eşleşmelerden yeniden kurulur → geometrik olarak anlamlı yumuşatma.
     * SIÇRAMA REDDİ: yeni H ızgarayı öncekine göre medyan 'sicrama_m' metreden fazla kaydırıyorsa
       kullanılmaz (önceki H ile devam). Aynı yönde 'sabir' kare üst üste ısrar ederse bu gerçek bir
       kamera hareketidir → kabul edilir.
  2) OyuncuTakipci (ByteTrack kimlikleri üzerine)
     * Her kimliğin saha konumu üstel ortalama ile yumuşatılır.
     * Tek karede 'kapi_m' metreden büyük sıçrama → ölçüm yok sayılır (kimlik karışması / kötü kutu).
     * Birkaç kare görünmeyen oyuncu son konumunda tutulur (kısa kapanmalar).
  3) TakimAtayici
     * Sahnenin ilk karesinde K-means ile 2 takım merkezi (Lab) bulunur; sonraki karelerde her oyuncu
       en yakın merkeze atanır, merkezler yavaşça güncellenir → renkler kareden kareye sabit.
     * Her iki merkeze de uzak renk → "diger" (hakem vb.). Kimlik başına çoğunluk oyu.
"""
from __future__ import annotations

from collections import Counter, deque

import cv2
import numpy as np

import pitch
from homografi import HomografiSonuc, noktalari_donustur


def _izgara(h: int, w: int) -> np.ndarray:
    """Görüntünün alt %70'ine yayılmış 6×4 nokta (saha genelde burada; ufuk üstte olabilir)."""
    xs = np.linspace(0.08 * w, 0.92 * w, 6)
    ys = np.linspace(0.35 * h, 0.95 * h, 4)
    return np.array([(x, y) for y in ys for x in xs], dtype=np.float64)


def _gecerli(p: np.ndarray) -> np.ndarray:
    return np.isfinite(p).all(1) & (np.abs(p[:, 0] - 52.5) < 150) & (np.abs(p[:, 1] - 34) < 120)


class HomografiFiltresi:
    def __init__(self, alfa: float = 0.5, sicrama_m: float = 3.0, sabir: int = 3, tolerans: int = 6):
        self.alfa, self.sicrama_m, self.sabir, self.tolerans = alfa, sicrama_m, sabir, tolerans
        self.reddedilen = 0      # toplam reddedilen sıçrama (sahne kesmesinde sıfırlanmaz)
        self.sifirla()

    def sifirla(self):
        self.kal = None          # şu an kullanılan (yumuşatılmış) homografi
        self.saha = None         # ızgaranın sahadaki yeri (yumuşatılmış)
        self.aday = []           # art arda reddedilen ama birbiriyle tutarlı yeni H'ler
        self.kayip = 0

    def _kur(self, saha: np.ndarray, kaynak: HomografiSonuc) -> HomografiSonuc | None:
        ok = _gecerli(saha)
        if ok.sum() < 4:
            return None
        H, _ = cv2.findHomography(self.izgara[ok], saha[ok], 0)
        if H is None:
            return None
        return HomografiSonuc(H, np.linalg.inv(H), kaynak.hata_metre, kaynak.hata_piksel,
                              kaynak.nokta_hatalari, kaynak.icerdeki)

    def guncelle(self, kal: HomografiSonuc | None, shape) -> HomografiSonuc | None:
        self.izgara = _izgara(*shape[:2])
        if kal is None:                                   # bu karede H yok → kısa süre öncekiyle devam
            self.kayip += 1
            if self.kayip > self.tolerans:
                self.sifirla()
            return self.kal
        self.kayip = 0
        yeni = noktalari_donustur(kal.H, self.izgara)
        if self.kal is None:
            if _gecerli(yeni).sum() >= 4:
                self.saha, self.kal = yeni, self._kur(yeni, kal)
            return self.kal
        ok = _gecerli(yeni) & _gecerli(self.saha)
        if ok.sum() < 4:
            return self.kal
        kayma = float(np.median(np.linalg.norm(yeni[ok] - self.saha[ok], axis=1)))
        if kayma > self.sicrama_m:
            # Sıçrama: tek kare hatası mı, gerçek kamera hareketi mi? Adaylar birbirini tutuyorsa gerçek.
            if self.aday and np.median(np.linalg.norm(yeni[ok] - self.aday[-1][ok], axis=1)) > self.sicrama_m:
                self.aday = []
            self.aday.append(yeni)
            if len(self.aday) >= self.sabir:
                self.saha, self.kal, self.aday = yeni, self._kur(yeni, kal), []
            else:
                self.reddedilen += 1
            return self.kal
        self.aday = []
        self.saha = np.where(ok[:, None], self.alfa * yeni + (1 - self.alfa) * self.saha, yeni)
        self.kal = self._kur(self.saha, kal) or self.kal
        return self.kal


class TakimAtayici:
    def __init__(self, guncelleme: float = 0.05, diger_carpan: float = 2.5, oy: int = 15):
        self.guncelleme, self.diger_carpan, self.oy = guncelleme, diger_carpan, oy
        self.sifirla()

    def sifirla(self):
        self.merkez = None       # (2, 3) Lab
        self.yayilim = None      # takım içi tipik renk uzaklığı
        self.oylar: dict[int, deque] = {}

    @staticmethod
    def _lab(bgr) -> np.ndarray:
        return cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB).reshape(3).astype(np.float32)

    def ata(self, tespitler) -> None:
        """Tespit.renk_bgr dolu oyuncular için t.sinif = takim_A / takim_B / diger, t.ek['takim_renk']."""
        oyuncular = [t for t in tespitler if t.sinif != "top" and t.renk_bgr is not None]
        if not oyuncular:
            return
        lab = np.array([self._lab(t.renk_bgr) for t in oyuncular])
        if self.merkez is None:
            if len(oyuncular) < 6:
                return
            cv2.setRNGSeed(0)
            kriter = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
            _, e, m = cv2.kmeans(lab, 2, None, kriter, 10, cv2.KMEANS_PP_CENTERS)
            self.merkez = m
            d = np.linalg.norm(lab - m[e.ravel()], axis=1)
            self.yayilim = max(8.0, float(np.median(d)))
        d = np.linalg.norm(lab[:, None, :] - self.merkez[None], axis=2)      # (n, 2)
        en_yakin = d.argmin(1)
        diger = d.min(1) > self.diger_carpan * self.yayilim
        for k in (0, 1):                                                    # merkezleri yavaşça güncelle
            sec = (en_yakin == k) & ~diger
            if sec.any():
                self.merkez[k] = (1 - self.guncelleme) * self.merkez[k] + self.guncelleme * lab[sec].mean(0)
        for t, k, dg in zip(oyuncular, en_yakin, diger):
            etiket = "diger" if dg else ("takim_A" if k == 0 else "takim_B")
            tid = t.ek.get("id")
            if tid is not None:                                             # kimlik başına çoğunluk oyu
                self.oylar.setdefault(tid, deque(maxlen=self.oy)).append(etiket)
                etiket = Counter(self.oylar[tid]).most_common(1)[0][0]
            t.sinif = etiket
        renk = {}
        for k, ad in ((0, "takim_A"), (1, "takim_B")):
            lab_m = np.clip(self.merkez[k], 0, 255).astype(np.uint8).reshape(1, 1, 3)
            renk[ad] = tuple(int(c) for c in cv2.cvtColor(lab_m, cv2.COLOR_LAB2BGR).reshape(3))
        renk["diger"] = (40, 40, 40)
        for t in oyuncular:
            t.ek["takim_renk"] = renk[t.sinif]


class OyuncuTakipci:
    def __init__(self, beta: float = 0.4, kapi_m: float = 4.0, bekle: int = 6, iz: int = 18):
        self.beta, self.kapi_m, self.bekle, self.iz_uzunluk = beta, kapi_m, bekle, iz
        self.sifirla()

    def sifirla(self):
        self.izler: dict[int, dict] = {}
        self.kare = 0

    def guncelle(self, tespitler) -> list[dict]:
        """
        tespitler: saha konumu (t.saha) ve kimliği (t.ek['id']) dolu Tespit'ler.
        Döner: haritada çizilecek [{'x','y','sinif','renk','id','iz'}] (yumuşatılmış).
        """
        self.kare += 1
        for t in tespitler:
            tid = t.ek.get("id")
            if tid is None or t.saha is None:
                continue
            p = np.array(t.saha)
            iz = self.izler.get(tid)
            if iz is None:
                self.izler[tid] = {"p": p, "son": self.kare, "atlanan": 0, "t": t,
                                   "gecmis": deque([p.copy()], maxlen=self.iz_uzunluk)}
                continue
            if np.linalg.norm(p - iz["p"]) > self.kapi_m and iz["atlanan"] < 2:
                iz["atlanan"] += 1                         # tek karelik saçma sıçrama → yok say
                iz["son"] = self.kare
                continue
            a = 1.0 if t.sinif == "top" else self.beta    # top hızlı: yumuşatma yok
            iz["p"] = a * p + (1 - a) * iz["p"]
            iz["son"], iz["atlanan"], iz["t"] = self.kare, 0, t
            iz["gecmis"].append(iz["p"].copy())
        for tid in [k for k, v in self.izler.items() if self.kare - v["son"] > self.bekle]:
            del self.izler[tid]
        return [{"x": float(v["p"][0]), "y": float(v["p"][1]), "sinif": v["t"].sinif, "id": tid,
                 "renk": v["t"].ek.get("takim_renk"), "iz": list(v["gecmis"])}
                for tid, v in self.izler.items()]
