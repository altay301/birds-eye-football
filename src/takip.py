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
from homografi import HomografiSonuc, isareti_duzelt, noktalari_donustur


def _izgara(h: int, w: int) -> np.ndarray:
    """Görüntünün alt %70'ine yayılmış 6×4 nokta (saha genelde burada; ufuk üstte olabilir)."""
    xs = np.linspace(0.08 * w, 0.92 * w, 6)
    ys = np.linspace(0.35 * h, 0.95 * h, 4)
    return np.array([(x, y) for y in ys for x in xs], dtype=np.float64)


def _gecerli(p: np.ndarray) -> np.ndarray:
    return np.isfinite(p).all(1) & (np.abs(p[:, 0] - 52.5) < 150) & (np.abs(p[:, 1] - 34) < 120)


class HomografiFiltresi:
    def __init__(self, alfa: float = 0.5, sicrama_m: float = 3.0, sabir: int = 3, tolerans: int = 6,
                 sabir_ilk: int = 2):
        self.alfa, self.sicrama_m, self.sabir, self.tolerans = alfa, sicrama_m, sabir, tolerans
        self.sabir_ilk = sabir_ilk
        self.yeniden_kuruldu = False
        self.makulsuz = 0        # makullük kontrolüyle reddedilen homografi sayısı (sıfırlanmaz)
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
        # findHomography ölçeği H[2,2] = 1 yapar ve bu işareti ters çevirebilir (H ile −H aynı dönüşüm,
        # ama noktalari_donustur işarete bakarak "kameranın arkası"nı eler → tüm noktalar NaN olur,
        # harita boş kalır). Ufuk görüntüdeki karelerde yaşadığımız sorunun aynısı → işareti düzelt.
        H = isareti_duzelt(H, self.izgara[ok])
        return HomografiSonuc(H, np.linalg.inv(H), kaynak.hata_metre, kaynak.hata_piksel,
                              kaynak.nokta_hatalari, kaynak.icerdeki)

    def guncelle(self, kal: HomografiSonuc | None, shape) -> HomografiSonuc | None:
        """
        Döner: bu karede kullanılacak homografi (None → harita çizme).
        self.yeniden_kuruldu: bu karede H sıfırdan kuruldu ya da büyük bir sıçrama kabul edildi →
        oyuncu konumlarının zamansal yumuşatması da sıfırlanmalı (eski konumlar artık geçersiz).
        """
        self.yeniden_kuruldu = False
        self.izgara = _izgara(*shape[:2])
        if kal is not None:
            # MAKULLÜK: yayın kamerası sahaya bakar → görüntünün alt kısmındaki ızgaranın ortası sahanın
            # içine ya da hemen yanına düşmeli. Saha dışına > 10 m düşüren H (ör. ayna simetrisi, yanlış
            # nokta numaraları) geçersiz sayılır.
            p = noktalari_donustur(kal.H, self.izgara)
            p = p[_gecerli(p)]
            if len(p) < 0.75 * len(self.izgara):          # ızgaranın dörtte birinden fazlası anlamsız yerde
                kal = None
                self.makulsuz += 1
            else:
                mx, my = np.median(p, axis=0)
                if not (-10 < mx < pitch.SAHA_UZUNLUK + 10 and -10 < my < pitch.SAHA_GENISLIK + 10):
                    kal = None
                    self.makulsuz += 1
        if kal is None:                                   # bu karede H yok → kısa süre öncekiyle devam
            self.kayip += 1
            if self.kayip > self.tolerans:
                self.sifirla()
            return self.kal
        self.kayip = 0
        yeni = noktalari_donustur(kal.H, self.izgara)
        if self.kal is None:
            # BAŞLANGIÇTA DA ISRAR: sahne kesmesinden sonraki ilk homografi sık sık yanlıştır (ölçüldü:
            # bir çekimin ilk H'si ızgarayı 46 m yanlış yere koydu ve filtre ona kilitlendi). Filtre ancak
            # 'sabir_ilk' ardışık homografi birbirini tutarsa başlar; o zamana kadar harita çizilmez.
            if _gecerli(yeni).sum() < 4:
                return None
            if self.aday:
                ok = _gecerli(yeni) & _gecerli(self.aday[-1])
                if ok.sum() < 4 or np.median(np.linalg.norm(yeni[ok] - self.aday[-1][ok], axis=1)) > self.sicrama_m:
                    self.aday = []
            self.aday.append(yeni)
            if len(self.aday) >= self.sabir_ilk:
                self.saha, self.kal, self.aday = yeni, self._kur(yeni, kal), []
                self.yeniden_kuruldu = True
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
                self.yeniden_kuruldu = True             # gerçek kamera hareketi kabul edildi
            else:
                self.reddedilen += 1
            return self.kal
        self.aday = []
        self.saha = np.where(ok[:, None], self.alfa * yeni + (1 - self.alfa) * self.saha, yeni)
        self.kal = self._kur(self.saha, kal) or self.kal
        return self.kal


def _kapali(kutular: np.ndarray, oran: float = 0.2) -> np.ndarray:
    """Her kutu için: alanının 'oran'dan fazlası başka bir kutuyla örtüşüyor mu? (renk karışır)"""
    n = len(kutular)
    if n < 2:
        return np.zeros(n, bool)
    x1 = np.maximum(kutular[:, None, 0], kutular[None, :, 0]); y1 = np.maximum(kutular[:, None, 1], kutular[None, :, 1])
    x2 = np.minimum(kutular[:, None, 2], kutular[None, :, 2]); y2 = np.minimum(kutular[:, None, 3], kutular[None, :, 3])
    kes = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    np.fill_diagonal(kes, 0)
    alan = (kutular[:, 2] - kutular[:, 0]) * (kutular[:, 3] - kutular[:, 1])
    return kes.max(1) > oran * np.maximum(alan, 1e-9)


class TakimAtayici:
    """
    kararli=True (varsayılan) — kararları KARE değil İZ (takip kimliği) düzeyinde verir:
      1) ROL KARARI: tespit modeli rolü (oyuncu/kaleci/hakem) her karede yeniden tahmin eder ve
         arada yanılır. Rol, izin son ~5 saniyesindeki tahminlerden asimetrik bir eşikle belirlenir
         (bkz. _rol): hakem birkaç karede "oyuncu" dense bile hakem kalır (haritada gizli).
      2) KAPANMA KONTROLÜ: kutusu başka kutuyla belirgin örtüşen oyuncunun o karedeki forma rengi
         (iki formanın karışımı) takım oylamasına KATILMAZ.
      3) Uzun takım oylaması (son 'oy' kare) + HİSTEREZİS: etiket ancak yeni aday mevcut etiketi
         'fark' oy geçerse değişir.
    kararli=False — önceki davranış (karşılaştırma için).
    """

    def __init__(self, guncelleme: float = 0.05, diger_carpan: float = 3.5, oy: int = 30,
                 kararli: bool = True, fark: int = 4, rol_esik: float = 0.25, kaleci_m: float = 25.0,
                 hakem_renk_m: float = 15.0, rol_pencere: int = 60):
        self.guncelleme, self.diger_carpan, self.kararli, self.fark = guncelleme, diger_carpan, kararli, fark
        self.rol_esik, self.kaleci_m, self.hakem_renk_m = rol_esik, kaleci_m, hakem_renk_m
        # Rol, izin son 'rol_pencere' karesinden (12 fps'de ~5 sn) belirlenir: tüm geçmiş kullanılırsa
        # takipçinin hakem↔oyuncu kimlik karışmasından sonra oyuncu izin sonuna kadar "hakem" kalırdı.
        self.rol_pencere = rol_pencere
        self.kaleci_reddi = Counter()   # hangi kural kaç iz-karede "kaleci"yi düzeltti (sıfırlanmaz)
        self.hakem_rengi = None         # hakem forması rengi (Lab); aynı maçta sahne kesmesinde de geçerli
        self.oy = oy if kararli else 15
        self.sifirla()

    def sifirla(self):
        # Takım renkleri her çekimde yeniden öğrenilir: ışık ve kamera değişince aynı forma farklı görünür
        # (kalıcı tutmak denendi: ikinci çekimde beyaz formalar "diğer"e düştü). Kötü başlangıç
        # (iki merkezin aynı takıma oturması) ayrıklık kontrolüyle engellenir (bkz. ata).
        self.merkez = None       # (2, 3) Lab
        self.yayilim = None      # takım içi tipik renk uzaklığı
        self.oylar: dict[int, deque] = {}
        self.rol_gecmis: dict[int, deque] = {}
        self.renk_ort: dict[int, np.ndarray] = {}   # iz → ortalama forma rengi (Lab, kapanmasız kareler)
        self.secili: dict[tuple, str] = {}       # (tür, iz) → şu an gösterilen etiket (histerezis için)

    def _rol(self, say: Counter) -> str:
        """
        ASİMETRİK ROL KURALI. Val karelerinde (v3 modeli): gerçek oyuncuya %6 "hakem", %7 "kaleci"
        deniyor; gerçek hakeme %74 "hakem", kaleciye %57 "kaleci". Model rolü KAÇIRMAYA yatkın ama
        nadiren UYDURUYOR → çoğunluk oyu hakemleri oyuncu sayar. Kural: izin son penceresinde en az
        'rol_esik' oranında (ve en az 3 kez) hakem denmişse hakem; kaleci için aynı. Oyuncunun yanlış
        tahmin oranı (%6–7) bu eşiğin çok altında → 30 karelik bir oyuncu izinin eşiği tesadüfen
        aşma olasılığı çok küçük (binom).
        """
        n = sum(say.values())
        for rol in ("hakem", "kaleci"):
            if say[rol] >= max(3, self.rol_esik * n):
                return rol
        return "oyuncu"

    def _karar(self, tur: str, tid: int, oylar: deque) -> str:
        """
        HİSTEREZİS: etiket ancak yeni aday, mevcut etiketi oy sayısında 'fark' kadar geçerse değişir.
        (Oy sayısı azken başa baş çoğunluğun kareden kareye dönüp durmasını engeller.)
        """
        say = Counter(oylar)
        aday, n = say.most_common(1)[0]
        mevcut = self.secili.get((tur, tid))
        if mevcut is None or not self.kararli or n >= say.get(mevcut, 0) + self.fark:
            self.secili[(tur, tid)] = mevcut = aday
        return mevcut

    @staticmethod
    def _lab(bgr) -> np.ndarray:
        return cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB).reshape(3).astype(np.float32)

    def _hakem_kaleci_ayir(self, kisiler) -> None:
        """
        Hakem ve kaleci ikisi de takımlardan farklı renk giyer; model ikisini karıştırabilir.
        Üç futbol bilgisi kullanılır (sırayla):
          a) KONUM: kaleci kendi ceza sahası çevresindedir. Kaleye 'kaleci_m' metreden uzakta
             "kaleci" denen kişi → hakem (gizli).
          b) HAKEM RENGİ: güvenle hakem denen izlerin forma renginden bir "hakem rengi" öğrenilir
             (sahne boyunca). Rengi buna 'hakem_renk_m' (Lab) mesafeden yakın "kaleci" → hakem.
          c) BENZERSİZLİK: her yarı sahada en fazla bir kaleci olur. Birden fazla aday varsa
             kale çizgisine EN YAKIN olan kaleci kalır, diğerleri → diger (soluk gösterilir).
        """
        hakem_renk = [self.renk_ort[t.ek["id"]] for t in kisiler
                      if t.sinif == "hakem" and t.ek.get("id") in self.renk_ort]
        if hakem_renk:                                                      # (b) referans rengi güncelle
            m = np.mean(hakem_renk, axis=0)
            self.hakem_rengi = m if self.hakem_rengi is None else 0.9 * self.hakem_rengi + 0.1 * m
        for t in kisiler:
            if t.sinif != "kaleci" or t.saha is None:
                continue
            if self.kaleci_m < t.saha[0] < pitch.SAHA_UZUNLUK - self.kaleci_m:               # (a)
                t.sinif = "hakem"           # orta sahadaki "kaleci" neredeyse her zaman hakemdir → gizli
                self.kaleci_reddi["konum"] += 1
                continue
            renk = self.renk_ort.get(t.ek.get("id"))
            if self.hakem_rengi is not None and renk is not None and \
                    np.linalg.norm(renk - self.hakem_rengi) < self.hakem_renk_m:              # (b)
                t.sinif = "hakem"
                self.kaleci_reddi["renk"] += 1
        for sol in (True, False):                                           # (c) yarı saha başına bir kaleci
            adaylar = [t for t in kisiler if t.sinif == "kaleci" and t.saha is not None
                       and (t.saha[0] < pitch.SAHA_UZUNLUK / 2) == sol]
            if len(adaylar) > 1:
                cizgiye = (lambda t: t.saha[0]) if sol else (lambda t: pitch.SAHA_UZUNLUK - t.saha[0])
                for t in sorted(adaylar, key=cizgiye)[1:]:
                    t.sinif = "diger"
                    self.kaleci_reddi["benzersizlik"] += 1

    def ata(self, tespitler) -> None:
        """
        Kişiler için t.sinif = takim_A / takim_B / diger / kaleci / hakem, oyunculara t.ek['takim_renk'].
        """
        kisiler = [t for t in tespitler if t.sinif != "top"]
        if self.kararli:
            kap = _kapali(np.array([t.kutu for t in kisiler], float).reshape(-1, 4))
            for t, kp in zip(kisiler, kap):                                 # 1) rol kararı (izin tüm geçmişi)
                tid = t.ek.get("id")
                if tid is not None:
                    gecmis = self.rol_gecmis.setdefault(tid, deque(maxlen=self.rol_pencere))
                    gecmis.append(t.sinif)
                    t.sinif = self._rol(Counter(gecmis))
                    if t.renk_bgr is not None and not kp:                   # izin ortalama forma rengi
                        lab = self._lab(t.renk_bgr)
                        m = self.renk_ort.get(tid)
                        self.renk_ort[tid] = lab if m is None else 0.9 * m + 0.1 * lab
            self._hakem_kaleci_ayir(kisiler)
        oyuncular = [t for t in kisiler if t.sinif == "oyuncu" and t.renk_bgr is not None]
        for t in kisiler:                       # kaleci/hakem/diger takım rengi almaz (sınıf rengiyle çizilir)
            if t.sinif not in ("oyuncu",) and not t.sinif.startswith("takim"):
                t.ek.pop("takim_renk", None)
        if not oyuncular:
            return
        kapali = (_kapali(np.array([t.kutu for t in oyuncular], float)) if self.kararli
                  else np.zeros(len(oyuncular), bool))                      # 2) kapanma
        lab = np.array([self._lab(t.renk_bgr) for t in oyuncular])
        if self.merkez is None:
            temiz = lab[~kapali] if (~kapali).sum() >= 6 else lab
            if len(temiz) < 6:
                return
            cv2.setRNGSeed(0)
            kriter = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
            _, e, m = cv2.kmeans(temiz, 2, None, kriter, 10, cv2.KMEANS_PP_CENTERS)
            d = np.linalg.norm(temiz - m[e.ravel()], axis=1)
            yayilim = max(8.0, float(np.median(d)))
            # Merkezleri ancak iki küme gerçekten ayrıksa kur: aradaki mesafe takım içi yayılımın en az
            # 2 katı ve her kümede en az 3 oyuncu. Değilse (ör. karede tek takım baskın) sonraki kareyi bekle.
            if self.kararli and (np.linalg.norm(m[0] - m[1]) < 2 * yayilim or np.bincount(e.ravel(), minlength=2).min() < 3):
                return
            self.merkez, self.yayilim = m, yayilim
        d = np.linalg.norm(lab[:, None, :] - self.merkez[None], axis=2)      # (n, 2)
        en_yakin = d.argmin(1)
        diger = d.min(1) > self.diger_carpan * self.yayilim
        for k in (0, 1):                                                    # merkezleri yavaşça güncelle
            sec = (en_yakin == k) & ~diger & ~kapali
            if sec.any():
                self.merkez[k] = (1 - self.guncelleme) * self.merkez[k] + self.guncelleme * lab[sec].mean(0)
        for t, k, dg, kp in zip(oyuncular, en_yakin, diger, kapali):
            etiket = "diger" if dg else ("takim_A" if k == 0 else "takim_B")
            tid = t.ek.get("id")
            if tid is not None:                                             # 3) iz başına çoğunluk oyu
                oylar = self.oylar.setdefault(tid, deque(maxlen=self.oy))
                if not kp or not oylar:                                     # kapalı karenin rengi oy değil
                    oylar.append(etiket)
                etiket = self._karar("takim", tid, oylar)
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
            sicradi = np.linalg.norm(p - iz["p"]) > self.kapi_m
            if sicradi and iz["atlanan"] < 2:
                iz["atlanan"] += 1                         # tek karelik saçma sıçrama → yok say
                iz["son"] = self.kare
                continue
            # Sıçrama ısrar ettiyse gerçektir (ör. kimlik düzeldi): ortalamayla yavaşça kaydırma, doğrudan atla.
            a = 1.0 if (t.sinif == "top" or sicradi) else self.beta   # top hızlı: yumuşatma yok
            if sicradi:
                iz["gecmis"].clear()
            iz["p"] = a * p + (1 - a) * iz["p"]
            iz["son"], iz["atlanan"], iz["t"] = self.kare, 0, t
            iz["gecmis"].append(iz["p"].copy())
        for tid in [k for k, v in self.izler.items() if self.kare - v["son"] > self.bekle]:
            del self.izler[tid]
        return [{"x": float(v["p"][0]), "y": float(v["p"][1]), "sinif": v["t"].sinif, "id": tid,
                 "renk": v["t"].ek.get("takim_renk"), "iz": list(v["gecmis"])}
                for tid, v in self.izler.items()]
