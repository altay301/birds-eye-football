"""
video_titreme.py — Videoda zamansal yumuşatmanın etkisini ölçer
==============================================================

Videoda gerçek konum etiketi yok; ama GERÇEK hareket (koşu, kamera dönüşü) akıcıdır, titreme ise
kareden kareye zikzaktır. Akıcılık ölçüsü olarak İKİNCİ FARK kullanılır:
        a_t = ‖ p(t+1) − 2·p(t) + p(t−1) ‖      (metre; sabit hızlı harekette 0)

Üç yöntem AYNI karelerde, AYNI saha noktası tahminleri ve AYNI tespit/kimliklerle karşılaştırılır:
    ham  : her kare bağımsız (RANSAC homografisi, ham ayak konumu)
    eski : önceki sürüm — H matrisinin üstel ortalaması (α = 0.3), ham ayak konumu
    yeni : HomografiFiltresi (sıçrama reddi + nokta uzayında yumuşatma) + OyuncuTakipci

Ölçütler (sadece aynı çekim içindeki ardışık kare üçlüleri):
    H titremesi       : görüntüdeki sabit ızgaranın sahadaki konumunun ikinci farkı (medyan / %90)
    oyuncu titremesi  : aynı kimlikli oyuncunun saha konumunun ikinci farkı (medyan / %90)
    kapsama           : homografisi olan kare oranı
    ham konuma uzaklık: yumuşatılmış konum ile o karenin ham konumu arasındaki fark
                        (aşırı yumuşatma / gecikme göstergesi: oyuncuyu "dondurmak" titremeyi
                        sıfırlar ama bu fark büyür)

Parametreler önceden seçildi ve bu kliplerde AYARLANMADI. Klipler test maçlarından.

Kullanım:  .venv\\Scripts\\python.exe -X utf8 src\\video_titreme.py
Çıktı   :  results\\video_titreme.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from homografi import HomografiSonuc, isareti_duzelt, noktalari_donustur
from keypoint_tahmin import model_yukle, ransac_homografi
from oyuncu import oyunculari_takip_et
from sistem import video_kareleri
from takip import HomografiFiltresi, OyuncuTakipci, _gecerli, _izgara

PROJE = Path(__file__).resolve().parent.parent
VIDEO = PROJE / "data" / "raw_videos"
KLIPLER = [("12_gxsTs2qtV8A", 60, 60), ("11_gsDyRUL4KX0", 90, 60), ("04_UDsByIrG9GE", 120, 60)]   # test maçları
KLIPLER_VAL = [("10_8_WZSknXW98", 60, 45), ("02_nagW6n39jCs", 120, 45)]                                # ayar için
YONTEMLER = ("ham", "eski", "yeni")


def eski_ema(kal, onceki, noktalar, alfa=0.3):
    """Önceki sürümün yöntemi: H matrislerinin (H[2,2] = 1 normalize) üstel ortalaması."""
    if kal is None or onceki is None:
        return kal
    H = alfa * kal.H / kal.H[2, 2] + (1 - alfa) * onceki.H / onceki.H[2, 2]
    H = isareti_duzelt(H, np.array([n["px"] for n in noktalar]))
    return HomografiSonuc(H, np.linalg.inv(H), kal.hata_metre, kal.hata_piksel, kal.nokta_hatalari)


def ikinci_fark(seriler: dict) -> list[float]:
    """seriler: anahtar → [(kare, çekim, (k, 2) dizi)]. Ardışık ve aynı çekimdeki üçlülerin ikinci farkı."""
    out = []
    for s in seriler.values():
        for (k0, c0, p0), (k1, c1, p1), (k2, c2, p2) in zip(s, s[1:], s[2:]):
            if k1 == k0 + 1 and k2 == k1 + 1 and c0 == c1 == c2:
                a = np.linalg.norm(p2 - 2 * p1 + p0, axis=-1)
                out += list(np.ravel(a[np.isfinite(a)]))
    return out


def klip_olc(model, ad: str, bas: float, sure: float, alfa: float = 0.5, beta: float = 0.4) -> dict:
    filtre, takipci = HomografiFiltresi(alfa=alfa), OyuncuTakipci(beta=beta)
    onceki_eski, kayip_eski = None, 0
    h_seri = {y: defaultdict(list) for y in YONTEMLER}
    o_seri = {y: defaultdict(list) for y in YONTEMLER}
    kapsama = {y: 0 for y in YONTEMLER}
    uzaklik = {"eski": [], "yeni": []}
    n, cekim = 0, 0
    for k, (img, kesme, _) in enumerate(video_kareleri(VIDEO / f"{ad}.mp4", bas, sure)):
        n += 1
        if kesme:
            cekim += 1
            filtre.sifirla(), takipci.sifirla()
            onceki_eski, kayip_eski = None, 0
        noktalar = model.tahmin(img)
        ham = ransac_homografi(noktalar)
        eski = eski_ema(ham, onceki_eski, noktalar)
        if eski is None and onceki_eski is not None and kayip_eski < 6:   # eski tolerans davranışı
            eski, kayip_eski = onceki_eski, kayip_eski + 1
        elif eski is not None:
            kayip_eski = 0
        onceki_eski = eski
        yeni = filtre.guncelle(ham, img.shape)
        tespitler = oyunculari_takip_et(img, "yolov8n.pt", sifirla=(k == 0 or kesme))
        izgara = _izgara(*img.shape[:2])
        ayak = np.array([t.ayak for t in tespitler]).reshape(-1, 2)
        kimlik = [t.ek.get("id") for t in tespitler]
        ham_konum = {}
        for y, kal in (("ham", ham), ("eski", eski), ("yeni", yeni)):
            if kal is None:
                continue
            kapsama[y] += 1
            p = noktalari_donustur(kal.H, izgara)
            p[~_gecerli(p)] = np.nan
            h_seri[y]["izgara"].append((k, cekim, p))
            if not len(ayak):
                continue
            s = noktalari_donustur(kal.H, ayak)
            if y == "ham":
                ham_konum = {i: s[j] for j, i in enumerate(kimlik) if i is not None}
            if y == "yeni":
                for t, q in zip(tespitler, s):
                    t.saha = tuple(q) if np.isfinite(q).all() else None
                for o in takipci.guncelle([t for t in tespitler if t.saha is not None]):
                    p_ = np.array([o["x"], o["y"]])
                    o_seri["yeni"][o["id"]].append((k, cekim, p_))
                    if o["id"] in ham_konum and np.isfinite(ham_konum[o["id"]]).all():
                        uzaklik["yeni"].append(float(np.linalg.norm(p_ - ham_konum[o["id"]])))
                continue
            for j, i in enumerate(kimlik):
                if i is not None and np.isfinite(s[j]).all():
                    o_seri[y][i].append((k, cekim, s[j]))
                    if y == "eski" and i in ham_konum and np.isfinite(ham_konum[i]).all():
                        uzaklik["eski"].append(float(np.linalg.norm(s[j] - ham_konum[i])))
    sonuc = {"kare": n, "cekim": cekim + 1, "reddedilen_sicrama": filtre.reddedilen}
    for y in YONTEMLER:
        hj, oj = ikinci_fark(h_seri[y]), ikinci_fark(o_seri[y])
        sonuc[y] = {"kapsama": kapsama[y] / n,
                    "H_titreme_medyan": float(np.median(hj)), "H_titreme_p90": float(np.percentile(hj, 90)),
                    "oyuncu_titreme_medyan": float(np.median(oj)), "oyuncu_titreme_p90": float(np.percentile(oj, 90)),
                    "oyuncu_uclu": len(oj)}
        if y in uzaklik and uzaklik[y]:
            sonuc[y]["ham_konuma_uzaklik_medyan"] = float(np.median(uzaklik[y]))
    return sonuc


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", action="store_true", help="parametre seçimi için val maçlarının klipleri")
    # α val kliplerinde seçildi: 0.3 ile 0.5 oyuncu titremesinde eşit, 0.5 ham konuma daha yakın (daha az gecikme)
    ap.add_argument("--alfa", type=float, default=0.5, help="homografi filtresinin yumuşatma katsayısı")
    ap.add_argument("--beta", type=float, default=0.4, help="oyuncu konumu yumuşatma katsayısı")
    args = ap.parse_args()
    model = model_yukle(str(PROJE / "models" / "unet_kp_gercek_v1.pt"))
    sonuclar = {"parametre": {"alfa": args.alfa, "beta": args.beta}}
    for ad, bas, sure in (KLIPLER_VAL if args.val else KLIPLER):
        sonuclar[f"{ad}@{bas}s"] = s = klip_olc(model, ad, bas, sure, args.alfa, args.beta)
        print(f"\n{ad} ({bas}–{bas + sure} sn): {s['kare']} kare, {s['cekim']} çekim, "
              f"{s['reddedilen_sicrama']} sıçrama reddedildi")
        print(f"  {'yöntem':6s} {'kapsama':>8s} {'H tit. med/p90 (m)':>20s} {'oyuncu tit. med/p90 (m)':>25s} {'ham konuma uzaklık':>19s}")
        for y in YONTEMLER:
            r = s[y]
            u = r.get("ham_konuma_uzaklik_medyan")
            print(f"  {y:6s} {r['kapsama']:8.0%} {r['H_titreme_medyan']:9.3f} / {r['H_titreme_p90']:6.3f}   "
                  f"{r['oyuncu_titreme_medyan']:11.3f} / {r['oyuncu_titreme_p90']:6.3f}   "
                  f"{(f'{u:.2f} m' if u is not None else '-'):>17s}")
    if args.val:
        return
    (PROJE / "results" / "video_titreme.json").write_text(json.dumps(sonuclar, indent=2, ensure_ascii=False),
                                                          encoding="utf-8")


if __name__ == "__main__":
    main()
