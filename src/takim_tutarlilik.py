"""
takim_tutarlilik.py — Videoda rol / takım etiketinin kareden kareye kararlılığını ölçer
=====================================================================================

Gözlem (kullanıcı): aynı oyuncu ardışık karelerde farklı takım renginde görünüyor; kaleci ve hakem
bazen sarı, bazen siyah ("diğer"); hakem haritaya giriyor.

Ölçütler (aynı kareler, aynı tespitler ve aynı takip kimlikleri; sadece TakimAtayici farklı):
    değişim / 100 iz-karesi : bir izin haritada gösterilen etiketi (takim_A, takim_B, kaleci, hakem, diger)
                              ardışık iki görünüşü arasında kaç kez değişiyor
    haritada takım dışı     : haritada GÖRÜNEN noktaların yüzde kaçı takım dışı (kaleci hariç):
                              "diger" (siyah) — önceki sürümde gizli değildi
    gizlenen iz-kare oranı  : hakem + diger olarak haritadan gizlenen iz-karelerinin oranı

Kullanım:  .venv\\Scripts\\python.exe -X utf8 src\\takim_tutarlilik.py
Çıktı   :  results\\takim_tutarlilik.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from keypoint_tahmin import model_yukle
from sistem import VideoHatti, video_kareleri

PROJE = Path(__file__).resolve().parent.parent
VIDEO = PROJE / "data" / "raw_videos"
KLIPLER = [("12_gxsTs2qtV8A", 60, 40), ("11_gsDyRUL4KX0", 90, 40), ("04_UDsByIrG9GE", 120, 40)]   # test maçları
YOLO = "tespit_futbol_v3_1536.pt"


def olc(model, ad, bas, sure, kararli: bool) -> dict:
    hat = VideoHatti(model, YOLO, yumusat=True, kararli_takim=kararli)
    son = {}                                     # iz → son gösterilen etiket
    degisim = goruntu = gizli = diger_gorunen = 0
    for img, kesme, _ in video_kareleri(VIDEO / f"{ad}.mp4", bas, sure):
        if kesme:
            son.clear()
        kal, _, oyuncular = hat.isle(img, kesme)
        if kal is None:
            continue
        for o in oyuncular:
            if o["sinif"] == "top" or o.get("id") is None:
                continue
            goruntu += 1
            etiket = o["sinif"]
            if o["id"] in son and son[o["id"]] != etiket:
                degisim += 1
            son[o["id"]] = etiket
            if etiket in ("hakem", "diger") and kararli:
                gizli += 1
            elif etiket in ("hakem",):
                gizli += 1
            elif etiket == "diger":
                diger_gorunen += 1               # önceki sürümde "diger" haritada siyah görünüyordu
    return {"iz_kare": goruntu, "degisim_100": 100 * degisim / max(goruntu, 1),
            "haritada_diger_orani": diger_gorunen / max(goruntu, 1), "gizlenen_orani": gizli / max(goruntu, 1)}


def main():
    model = model_yukle(str(PROJE / "models" / "unet_kp_gercek_v1.pt"))
    sonuc = {}
    print(f"{'klip':22s} {'yöntem':8s} {'iz-kare':>8s} {'değişim/100':>12s} {'haritada siyah':>15s} {'gizlenen':>9s}")
    for ad, bas, sure in KLIPLER:
        for kararli, isim in ((False, "önceki"), (True, "yeni")):
            s = olc(model, ad, bas, sure, kararli)
            sonuc[f"{ad}@{bas}s/{isim}"] = s
            print(f"{ad:22s} {isim:8s} {s['iz_kare']:8d} {s['degisim_100']:12.2f} "
                  f"{s['haritada_diger_orani']:14.1%} {s['gizlenen_orani']:9.1%}")
    (PROJE / "results" / "takim_tutarlilik.json").write_text(json.dumps(sonuc, indent=2, ensure_ascii=False),
                                                             encoding="utf-8")


if __name__ == "__main__":
    main()
