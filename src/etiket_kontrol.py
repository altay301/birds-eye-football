"""
etiket_kontrol.py — Etiketlerin otomatik kalite kontrolü
========================================================

FİKİR: Bir karede 4+ nokta varsa homografi hesaplanabilir. Doğru etiketlenmiş
noktalar aynı düzlem dönüşümüne UYAR. Yanlış ID verilmiş ya da kaymış bir nokta
uymaz → o noktanın "kalibrasyon hatası" (metre) diğerlerinden çok büyük çıkar.
Yani etiketlerin kendisi, birbirini kontrol eder.

Çıktı:
  * Video ve küme bazında ilerleme tablosu
  * Şüpheli kareler listesi (bir noktası > 1.5 m sapıyor)  → etiketle.py ile düzelt
  * results\\etiket_kontrol.csv

Kullanım:
    .venv\\Scripts\\python.exe src\\etiket_kontrol.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import pitch
from homografi import homografi_hesapla

PROJE = Path(__file__).resolve().parent.parent
ETIKET = PROJE / "data" / "labels" / "noktalar"
KARE = PROJE / "data" / "frames"
ESIK_M = 1.5


def main():
    bolme = json.loads((PROJE / "data" / "bolme.json").read_text(encoding="utf-8"))
    ozet = defaultdict(lambda: {"kare": 0, "etiketli": 0, "atlandi": 0, "nokta": 0, "kalibre": 0})
    supheli = []
    satirlar = []
    for video in sorted(bolme):
        ozet[video]["kare"] = len(list((KARE / video).glob("*.jpg")))
        for d in sorted((ETIKET / video).glob("*.json")):
            e = json.loads(d.read_text(encoding="utf-8"))
            durum = e.get("durum")
            if durum == "atlandi":
                ozet[video]["atlandi"] += 1
                continue
            if durum != "etiketli" or not e["noktalar"]:
                continue
            o = ozet[video]
            o["etiketli"] += 1
            o["nokta"] += len(e["noktalar"])
            ids = [n["id"] for n in e["noktalar"]]
            if len(ids) != len(set(ids)):
                supheli.append((video, d.stem, "aynı ID iki kez"))
            hata_m, en_kotu = "", ""
            if len(e["noktalar"]) >= 4:
                try:
                    k = homografi_hesapla(np.array([n["px"] for n in e["noktalar"]]), pitch.NOKTA_KOORD[ids])
                    o["kalibre"] += 1
                    hata_m = f"{k.hata_metre:.2f}"
                    i = int(np.argmax(k.nokta_hatalari))
                    en_kotu = f"{ids[i]} ({k.nokta_hatalari[i]:.2f} m)"
                    if k.nokta_hatalari[i] > ESIK_M:
                        supheli.append((video, d.stem, f"nokta {en_kotu} uyumsuz"))
                except ValueError:
                    pass
            satirlar.append([video, bolme[video], d.stem, len(ids), hata_m, en_kotu])

    print(f"{'video':18s} {'küme':5s} {'kare':>4s} {'etiketli':>8s} {'atlandı':>7s} {'nokta':>6s} {'kalibre':>7s}")
    toplam = defaultdict(int)
    for v in sorted(bolme):
        o = ozet[v]
        print(f"{v:18s} {bolme[v]:5s} {o['kare']:4d} {o['etiketli']:8d} {o['atlandi']:7d} {o['nokta']:6d} {o['kalibre']:7d}")
        for a in o:
            toplam[(bolme[v], a)] += o[a]
    for k in ("train", "val", "test"):
        print(f"  {k:5s}: {toplam[(k, 'etiketli')]} etiketli / {toplam[(k, 'kare')]} kare")
    if supheli:
        print(f"\nŞÜPHELİ KARELER ({len(supheli)}) — etiketle.py --video <video> ile aç ve düzelt:")
        for v, s, neden in supheli:
            print(f"  {v}/{s}: {neden}")
    else:
        print("\nŞüpheli kare yok.")
    (PROJE / "results").mkdir(exist_ok=True)
    with open(PROJE / "results" / "etiket_kontrol.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["video", "kume", "kare", "nokta_sayisi", "ort_hata_m", "en_kotu_nokta"])
        w.writerows(satirlar)


if __name__ == "__main__":
    main()
