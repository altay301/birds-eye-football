"""
etiketle.py — Saha anahtar noktası etiketleme aracı (klasör modu)
=================================================================

data\\frames içindeki tüm kareleri sırayla açar; sen her karede GÖRÜNEN TÜM saha
noktalarını işaretlersin. Pencere ve tuşlar manuel_kalibrasyon.py ile aynıdır,
üstüne kareler arası gezinme eklenmiştir.

Kullanım (proje klasöründe):
    .venv\\Scripts\\python.exe src\\etiketle.py                      # ilk etiketlenmemiş kareden başla
    .venv\\Scripts\\python.exe src\\etiketle.py --video 07_h68azhf5Ik0  # sadece bir video
    .venv\\Scripts\\python.exe src\\etiketle.py --bastan             # ilk kareden başla
    .venv\\Scripts\\python.exe src\\etiketle.py --ozet               # pencere açmadan ilerlemeyi yazdır
    .venv\\Scripts\\python.exe src\\etiketle.py --her-mactan 8       # her maçtan eşit yayılmış 8 kare
    .venv\\Scripts\\python.exe src\\etiketle.py --kontrol            # Claude'un ön-etiketlerini sırayla kontrol et
    .venv\\Scripts\\python.exe src\\etiketle.py --kareler 01:t00186 10:t00258   # sadece belirli kareler

Ek tuşlar:
    N   kaydet → sonraki kare          P   kaydet → önceki kare
    X   bu kareyi ATLA (kullanılamaz: tören, yakın plan, grafik...) → sonraki
    A   önerileri kabul et (turuncu = model, camgöbeği = homografi)
    H   geometrik öneri: işaretlediğin 4+ noktadan homografi kur, kalan noktaların
        yerini öner (hiçbir modelden gelmez)
    Q   kaydet ve çık

TEST MAÇLARI (bolme.json'da "test"): model önerisi otomatik GİZLENİR. Öneriyi yapan
model test etiketine "bulaşırsa" ölçüm o modeli kayırır (04'te etiketlerin %66'sı
önerinin birebir aynısıydı). Etiket dosyasına "yontem": "onerisiz" yazılır.

KURALLAR (ayrıntı: report\\etiketleme_kilavuzu.md):
    * Görünen TÜM noktaları işaretle. İşaretlenmeyen görünür nokta, modele
      "orada nokta yok" diye YANLIŞ bilgi verir.
    * 1-2 nokta görünüyorsa bile işaretle — keypoint modeli için değerli.
      (Homografi için 4 nokta gerekir ama eğitim için gerekmez.)
    * Noktayı çizgi kalınlığının ORTASINA koy; zoom kullan.

Kaydedilenler:
    data\\labels\\noktalar\\<video>\\<kare>.json   {"goruntu", "noktalar", "durum"}
    durum: "etiketli" (≥1 nokta) | "atlandi" | "bos" (açıldı ama nokta konmadı)
Öneriler (varsa) buradan okunur:
    data\\labels\\oneri_noktalar\\<video>\\<kare>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import pitch
from manuel_kalibrasyon import Arayuz, durum_yukle, noktalari_kaydet

PROJE = Path(__file__).resolve().parent.parent
KARE_KLASOR = PROJE / "data" / "frames"
ETIKET_KLASOR = PROJE / "data" / "labels" / "noktalar"
ONERI_KLASOR = PROJE / "data" / "labels" / "oneri_noktalar"


def kareleri_listele(video: str | None = None) -> list[Path]:
    desen = f"{video}/*.jpg" if video else "*/*.jpg"
    return sorted(KARE_KLASOR.glob(desen))


def etiket_yolu(kare: Path) -> Path:
    return ETIKET_KLASOR / kare.parent.name / f"{kare.stem}.json"


def kume_haritasi() -> dict:
    yol = PROJE / "data" / "bolme.json"
    return json.loads(yol.read_text(encoding="utf-8")) if yol.exists() else {}


def oneri_yukle(kare: Path) -> list[dict]:
    yol = ONERI_KLASOR / kare.parent.name / f"{kare.stem}.json"
    if yol.exists():
        return json.loads(yol.read_text(encoding="utf-8"))["noktalar"]
    return []


def ilerleme(kareler: list[Path]) -> dict:
    say = {"etiketli": 0, "atlandi": 0, "bos": 0, "acilmadi": 0}
    for k in kareler:
        d = durum_yukle(etiket_yolu(k))
        say[d if d in say else "acilmadi"] += 1
    return say


class Etiketleyici:
    def __init__(self, kareler: list[Path], baslangic: int, test_oneri: bool = False):
        self.kareler = kareler
        self.i = baslangic
        self.aktif: Arayuz | None = None
        self.mesaj_devir = ""
        self.kumeler = kume_haritasi()
        self.test_oneri = test_oneri    # True: test maçlarında da model önerisi göster (önerilmez)

    def onerisiz_mi(self, kare: Path) -> bool:
        """TEST maçlarında model önerisi gizlenir: etiket modele doğru kaymasın (onay yanlılığı)."""
        return self.kumeler.get(kare.parent.name) == "test" and not self.test_oneri

    def kareyi_ac(self):
        kare = self.kareler[self.i]
        onerisiz = self.onerisiz_mi(kare)
        a = Arayuz(kare, json_yol=etiket_yolu(kare), oneri=[] if onerisiz else oneri_yukle(kare))
        a.tus_yardim = "[N sonraki | P onceki | X atla | H geometrik oneri | A oneri kabul | D sil | Z geri | Q cik]"
        d = durum_yukle(a.json_yol)
        a.baslik_ek = (f"Kare {self.i + 1}/{len(self.kareler)} {kare.parent.name[:2]}"
                       f"{' TEST-onerisiz' if onerisiz else ''}"
                       f"{' [ATLANDI]' if d == 'atlandi' else ''} | ")
        if onerisiz:
            ilk = "TEST: model onerisi yok. 4-5 net noktayi kendin isaretle, sonra H."
        elif a.oneri:
            ilk = "Oneriler var: A ile kabul et"
        else:
            ilk = "Gorunen TUM noktalari isaretle."
        a.mesaj = self.mesaj_devir or ilk
        self.mesaj_devir = ""
        self.aktif = a

    def kaydet(self, atla: bool = False):
        a = self.aktif
        eski_veri = json.loads(a.json_yol.read_text(encoding="utf-8")) if a.json_yol.exists() else {}
        durum = "atlandi" if atla else ("etiketli" if a.noktalar else "bos")
        # Atlanmış kareye bakıp nokta koymadan N'ye basmak onu "boş"a ÇEVİRMESİN
        # ("boş" = "görünen nokta yok" demek, eğitimde kullanılır; "atlandı" hiç kullanılmaz)
        if durum == "bos" and eski_veri.get("durum") == "atlandi":
            durum = "atlandi"
        # Etiketin nasıl yapıldığını kaydet → raporda test sonuçlarını ayrı verebilmek için
        # (Dosyada zaten yazıyorsa koru: 04 öneriyle yapıldı, P ile dönüp kaydetmek bunu değiştirmesin)
        yontem = eski_veri.get("yontem") or ("onerisiz" if self.onerisiz_mi(a.yol) else "model_onerili")
        if yontem == "claude_oneri_kontrol_bekliyor":
            yontem = "claude_oneri_kontrollu"     # Claude'un ön-etiketi: sen açıp kaydettin = kontrol edildi
        noktalari_kaydet(a.json_yol, a.yol, a.noktalar, durum=durum, ek={"yontem": yontem})
        return durum

    def git(self, adim: int):
        yeni = self.i + adim
        if 0 <= yeni < len(self.kareler):
            self.i = yeni
            self.kareyi_ac()
        else:
            self.aktif.mesaj = "Listenin sonu." if adim > 0 else "Listenin basi."

    def calistir(self):
        cv2.namedWindow(Arayuz.PENCERE, cv2.WINDOW_AUTOSIZE)
        # Fare olaylarını o an açık olan kareye yönlendir
        cv2.setMouseCallback(Arayuz.PENCERE, lambda *a: self.aktif.fare(*a))
        sablon = pitch.anahtar_noktalari_ciz(pitch.saha_ciz(pitch.HaritaAyar(olcek=7)),
                                             pitch.HaritaAyar(olcek=7))
        cv2.imshow(Arayuz.SABLON, sablon)
        self.kareyi_ac()
        while True:
            cv2.imshow(Arayuz.PENCERE, self.aktif.ciz())
            k = cv2.waitKey(30) & 0xFF
            if cv2.getWindowProperty(Arayuz.PENCERE, cv2.WND_PROP_VISIBLE) < 1:
                self.kaydet()
                break
            if k == 255:
                continue
            yaziyor = self.aktif.bekleyen is not None   # numara girerken harf tuşlarını yutma
            if not yaziyor and k in (ord("n"), ord("N")):
                d = self.kaydet()
                self.mesaj_devir = f"Onceki kare kaydedildi ({d})."
                self.git(+1)
            elif not yaziyor and k in (ord("p"), ord("P")):
                self.kaydet()
                self.git(-1)
            elif not yaziyor and k in (ord("x"), ord("X")):
                self.kaydet(atla=True)
                self.mesaj_devir = "Onceki kare ATLANDI olarak isaretlendi."
                self.git(+1)
            elif not yaziyor and k in (ord("q"), ord("Q")):
                self.kaydet()
                break
            elif not yaziyor and k in (ord("s"), ord("S")):
                d = self.kaydet()
                self.aktif.mesaj = f"Kaydedildi ({d})."
            else:
                self.aktif.tus(k)
        cv2.destroyAllWindows()
        s = ilerleme(self.kareler)
        print(f"İlerleme: {s['etiketli']} etiketli, {s['atlandi']} atlandı, "
              f"{s['bos']} boş, {s['acilmadi']} açılmadı (toplam {len(self.kareler)})")


def main():
    ap = argparse.ArgumentParser(description="Saha noktası etiketleme (klasör modu)")
    ap.add_argument("--video", help="sadece bu videonun kareleri (klasör adı)")
    ap.add_argument("--bastan", action="store_true", help="ilk kareden başla")
    ap.add_argument("--ozet", action="store_true", help="sadece ilerlemeyi yazdır")
    ap.add_argument("--her-mactan", type=int, default=0,
                    help="her maçtan eşit yayılmış en fazla N kare (train maçları için, ör. 8)")
    ap.add_argument("--kontrol", action="store_true",
                    help="sadece Claude'un ön-etiketlediği kareler (sırayla kontrol için)")
    ap.add_argument("--test-oneri", action="store_true",
                    help="test maçlarında da model önerisini göster (dürüst ölçümü bozar, önerilmez)")
    ap.add_argument("--kareler", nargs="+",
                    help="sadece bu kareler, 'MAÇ:KARE' biçiminde (ör. 01:t00186 10:t00258)")
    args = ap.parse_args()

    kareler = kareleri_listele(args.video)
    if args.kareler:
        # 'MAÇ:KARE' → maç klasörü adı MAÇ ile başlayan, dosya adında '_KARE' geçen kare
        istenen = [s.split(":") for s in args.kareler]
        kareler = [k for m, t in istenen for k in kareler
                   if k.parent.name.startswith(m) and f"_{t}" in k.stem]
        args.bastan = True
    if args.her_mactan:
        # Her maçtan videoya EŞİT YAYILMIŞ N kare (ilk N kare değil: maçın başı sonundan farklı olabilir)
        secilen = []
        for v in sorted({k.parent.name for k in kareler}):
            vk = [k for k in kareler if k.parent.name == v]
            idx = sorted(set(np.linspace(0, len(vk) - 1, min(args.her_mactan, len(vk))).round().astype(int)))
            secilen += [vk[i] for i in idx]
        kareler = secilen
    if args.kontrol:
        def yontem(k):
            y = etiket_yolu(k)
            return json.loads(y.read_text(encoding="utf-8")).get("yontem", "") if y.exists() else ""
        kareler = [k for k in kareler if yontem(k).startswith("claude_oneri")]
        bekleyen = [i for i, k in enumerate(kareler) if yontem(k) == "claude_oneri_kontrol_bekliyor"]
        print(f"Kontrol modu: {len(kareler)} kare, {len(bekleyen)} tanesi henüz kontrol edilmedi")
        if not bekleyen and not args.ozet:
            print("Kontrol edilecek kare kalmadı.")
            return
    if not kareler:
        print(f"Kare bulunamadı: {KARE_KLASOR}")
        return
    s = ilerleme(kareler)
    print(f"{len(kareler)} kare | {s['etiketli']} etiketli, {s['atlandi']} atlandı, "
          f"{s['bos']} boş, {s['acilmadi']} açılmadı")
    if args.ozet:
        return
    baslangic = 0
    if args.kontrol and not args.bastan:
        baslangic = bekleyen[0]           # ilk kontrol edilmemiş kare
    elif not args.bastan:
        for i, k in enumerate(kareler):
            if durum_yukle(etiket_yolu(k)) is None:
                baslangic = i
                break
    Etiketleyici(kareler, baslangic, test_oneri=args.test_oneri).calistir()


if __name__ == "__main__":
    main()
