"""
kutu_etiketle.py — Oyuncu / kaleci / hakem / top kutularını etiketleme aracı
============================================================================

YOLO'yu futbola ince ayar (fine-tune) yapmak için kutu etiketleri gerekir.
Sıfırdan çizmek yerine: hazır YOLO'nun bulduğu kutular ÖNERİ olarak gelir
(kutu_oneri.py üretir), sen sadece DÜZELTİRSİN.

SINIFLAR (rol tabanlı — "takım A/B" DEĞİL; çünkü bir maçtaki "A" diğerinde
anlamsızdır. Takım, her karede forma renginden K-means ile ayrılır):
    1 oyuncu   2 kaleci   3 hakem (yan hakemler dahil)   4 top

Kullanım:
    .venv\\Scripts\\python.exe src\\kutu_etiketle.py
    .venv\\Scripts\\python.exe src\\kutu_etiketle.py --video 07_h68azhf5Ik0
    .venv\\Scripts\\python.exe src\\kutu_etiketle.py --her-mactan 5     # her maçtan eşit yayılmış 5 kare

Tuşlar / fare:
    Sol tık KUTUNUN İÇİNE     sınıfını değiştir (oyuncu → kaleci → hakem → top → oyuncu)
    Sol tık + sürükle (boşta) yeni kutu çiz (seçili sınıfla)
    1 / 2 / 3 / 4             yeni kutular için sınıf seç
    D                         imlecin altındaki kutuyu sil
    Tekerlek / sağ tuş        yakınlaştır / gez        R  görünümü sıfırla
    N / P                     kaydet → sonraki / önceki      X  kareyi atla      Q  çık

Kaydedilen:  data\\labels\\kutular\\<video>\\<kare>.json
    {"goruntu", "durum", "kutular": [{"sinif": "oyuncu", "xyxy": [x1, y1, x2, y2]}, ...]}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from manuel_kalibrasyon import Arayuz

PROJE = Path(__file__).resolve().parent.parent
KARE = PROJE / "data" / "frames"
ETIKET = PROJE / "data" / "labels" / "kutular"
ONERI = PROJE / "data" / "labels" / "kutular_oneri"
SINIFLAR = ["oyuncu", "kaleci", "hakem", "top"]
RENK = {"oyuncu": (255, 200, 0), "kaleci": (0, 200, 255), "hakem": (40, 40, 40), "top": (0, 0, 255)}


def etiket_yolu(kare: Path) -> Path:
    return ETIKET / kare.parent.name / f"{kare.stem}.json"


def yukle(kare: Path) -> tuple[list[dict], str | None, bool]:
    """Döner: (kutular, durum, öneri_mi)."""
    e = etiket_yolu(kare)
    if e.exists():
        v = json.loads(e.read_text(encoding="utf-8"))
        return v["kutular"], v.get("durum"), False
    o = ONERI / kare.parent.name / f"{kare.stem}.json"
    if o.exists():
        return json.loads(o.read_text(encoding="utf-8"))["kutular"], None, True
    return [], None, False


class KutuArayuz(Arayuz):
    """Arayuz'un yakınlaştırma/kaydırma altyapısını kullanır; nokta yerine kutu çizer."""

    def __init__(self, kare: Path):
        super().__init__(kare, json_yol=PROJE / "data" / "labels" / "_yok.json")
        self.kutular, self.durum, oneri = yukle(kare)
        self.secili_sinif = "oyuncu"
        self.ciziliyor = None      # sürüklenerek çizilen kutunun başlangıcı (orijinal piksel)
        self.mesaj = "Oneriler yuklendi: duzelt, eksikleri ciz." if oneri else "Kutulari ciz."
        self.tus_yardim = "[Kutuya tik: sinif | Surukle: yeni | 1-4 sinif | D sil | N/P | X atla | Q]"

    def kutu_altinda(self, ox, oy) -> int | None:
        """İmlecin altındaki EN KÜÇÜK kutu (iç içe kutularda küçük olanı seç)."""
        adaylar = [(i, (k["xyxy"][2] - k["xyxy"][0]) * (k["xyxy"][3] - k["xyxy"][1]))
                   for i, k in enumerate(self.kutular)
                   if k["xyxy"][0] <= ox <= k["xyxy"][2] and k["xyxy"][1] <= oy <= k["xyxy"][3]]
        return min(adaylar, key=lambda a: a[1])[0] if adaylar else None

    def fare(self, olay, x, y, bayraklar, *_):
        ey = y - self.SERIT
        if olay == cv2.EVENT_LBUTTONDOWN and ey >= 0:
            self.ciziliyor = self.ekran_to_orijinal(x, ey)
            self.fare_ekran = (x, ey)
            return
        if olay == cv2.EVENT_LBUTTONUP and self.ciziliyor is not None and ey >= 0:
            x0, y0 = self.ciziliyor
            x1, y1 = self.ekran_to_orijinal(x, ey)
            self.ciziliyor = None
            if abs(x1 - x0) * self.s < 4 and abs(y1 - y0) * self.s < 4:   # sürüklemedi = tık
                i = self.kutu_altinda(x1, y1)
                if i is not None:
                    k = self.kutular[i]
                    k["sinif"] = SINIFLAR[(SINIFLAR.index(k["sinif"]) + 1) % len(SINIFLAR)]
                    self.mesaj = f"Sinif: {k['sinif']}"
                return
            self.kutular.append({"sinif": self.secili_sinif,
                                 "xyxy": [round(min(x0, x1), 1), round(min(y0, y1), 1),
                                          round(max(x0, x1), 1), round(max(y0, y1), 1)]})
            self.mesaj = f"Eklendi: {self.secili_sinif}"
            return
        if olay in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP):
            return
        super().fare(olay, x, y, bayraklar)   # tekerlek, sağ tuş, imleç

    def ciz(self) -> np.ndarray:
        M = np.float32([[self.s, 0, -self.gx * self.s], [0, self.s, -self.gy * self.s]])
        img = cv2.warpAffine(self.goruntu, M, (self.ekran_w, self.ekran_h), flags=cv2.INTER_LINEAR)
        for k in self.kutular:
            a = tuple(np.round(self.orijinal_to_ekran(k["xyxy"][:2])).astype(int))
            b = tuple(np.round(self.orijinal_to_ekran(k["xyxy"][2:])).astype(int))
            cv2.rectangle(img, a, b, RENK[k["sinif"]], 2)
            if k["sinif"] != "oyuncu":
                cv2.putText(img, k["sinif"][:3].upper(), (a[0], a[1] - 4), 0, 0.5, RENK[k["sinif"]], 2)
        if self.ciziliyor is not None and self.fare_ekran:
            a = tuple(np.round(self.orijinal_to_ekran(self.ciziliyor)).astype(int))
            cv2.rectangle(img, a, tuple(self.fare_ekran), RENK[self.secili_sinif], 1)
        if self.fare_ekran:
            fx, fy = self.fare_ekran
            cv2.line(img, (fx - 12, fy), (fx + 12, fy), (255, 0, 255), 1)
            cv2.line(img, (fx, fy - 12), (fx, fy + 12), (255, 0, 255), 1)
        say = {s: sum(k["sinif"] == s for k in self.kutular) for s in SINIFLAR}
        durum = self.baslik_ek + " | ".join(f"{s}: {n}" for s, n in say.items()) + \
            f" | yeni kutu: {self.secili_sinif.upper()} | Zoom {self.zoom:.1f}x"
        serit = np.zeros((self.SERIT, img.shape[1], 3), np.uint8)
        cv2.putText(serit, durum, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(serit, self.mesaj + "  " + self.tus_yardim, (10, 46), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 255, 255), 1)
        return np.vstack([serit, img])

    def tus(self, k: int) -> bool:
        if ord("1") <= k <= ord("4"):
            self.secili_sinif = SINIFLAR[k - ord("1")]
            self.mesaj = f"Yeni kutular: {self.secili_sinif}"
        elif k in (ord("d"), ord("D")) and self.fare_ekran:
            i = self.kutu_altinda(*self.ekran_to_orijinal(*self.fare_ekran))
            if i is not None:
                self.mesaj = f"Silindi: {self.kutular.pop(i)['sinif']}"
        elif k in (ord("r"), ord("R")):
            self.zoom, self.gx, self.gy = 1.0, 0.0, 0.0
        return True

    def kaydet(self, atla: bool = False) -> str:
        durum = "atlandi" if atla else "etiketli"
        yol = etiket_yolu(self.yol)
        yol.parent.mkdir(parents=True, exist_ok=True)
        yol.write_text(json.dumps({"goruntu": self.yol.name, "durum": durum, "kutular": self.kutular},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
        return durum


def main():
    ap = argparse.ArgumentParser(description="Oyuncu/top kutusu etiketleme")
    ap.add_argument("--video")
    ap.add_argument("--bastan", action="store_true")
    ap.add_argument("--her-mactan", type=int, default=0,
                    help="her maçtan videoya eşit yayılmış en fazla N kare (ör. 5)")
    args = ap.parse_args()
    kareler = sorted(KARE.glob(f"{args.video}/*.jpg" if args.video else "*/*.jpg"))
    if args.her_mactan:
        # Sadece önerisi olan kareler (homografi kurulabilen saha görüntüleri), maç başına eşit yayılmış N tane
        kareler = [k for k in kareler if (ONERI / k.parent.name / f"{k.stem}.json").exists()]
        secilen = []
        for v in sorted({k.parent.name for k in kareler}):
            vk = [k for k in kareler if k.parent.name == v]
            idx = sorted(set(np.linspace(0, len(vk) - 1, min(args.her_mactan, len(vk))).round().astype(int)))
            secilen += [vk[i] for i in idx]
        kareler = secilen
    if not kareler:
        print("Kare yok.")
        return
    i = 0
    if not args.bastan:
        i = next((j for j, k in enumerate(kareler) if not etiket_yolu(k).exists()), 0)
    aktif = {"a": None}

    def ac(j):
        a = KutuArayuz(kareler[j])
        a.baslik_ek = f"Kare {j + 1}/{len(kareler)} | "
        aktif["a"] = a

    cv2.namedWindow(Arayuz.PENCERE, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(Arayuz.PENCERE, lambda *x: aktif["a"].fare(*x))
    ac(i)
    while True:
        cv2.imshow(Arayuz.PENCERE, aktif["a"].ciz())
        k = cv2.waitKey(30) & 0xFF
        if cv2.getWindowProperty(Arayuz.PENCERE, cv2.WND_PROP_VISIBLE) < 1:
            aktif["a"].kaydet()
            break
        if k == 255:
            continue
        if k in (ord("n"), ord("N"), ord("x"), ord("X")):
            aktif["a"].kaydet(atla=k in (ord("x"), ord("X")))
            if i + 1 < len(kareler):
                i += 1
                ac(i)
        elif k in (ord("p"), ord("P")):
            aktif["a"].kaydet()
            if i > 0:
                i -= 1
                ac(i)
        elif k in (ord("q"), ord("Q")):
            aktif["a"].kaydet()
            break
        else:
            aktif["a"].tus(k)
    cv2.destroyAllWindows()
    n = sum(etiket_yolu(k).exists() for k in kareler)
    print(f"Kutu etiketli kare: {n}/{len(kareler)}")


if __name__ == "__main__":
    main()
