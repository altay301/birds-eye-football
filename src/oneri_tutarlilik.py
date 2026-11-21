"""
oneri_tutarlilik.py — Model önerilerinin kalitesini ETİKETSİZ tahmin eder
=========================================================================

Fikir: Doğru noktalar aynı homografiye uyar. Her karede RANSAC'ın önerilerin ne
kadarını "tutarlı" (inlier) bulduğuna bakarak gerçek etiket olmadan kabaca kalite
ölçülür. Yüksek tutarlılık doğruluğu GARANTİ etmez (hepsi aynı yönde yanlış olabilir)
ama düşük tutarlılık kesinlikle sorun demektir.

Çıktı: video ve küme bazında "homografi kurulabilen kare oranı" ve "ortalama inlier oranı".
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import pitch
from homografi import homografi_hesapla

PROJE = Path(__file__).resolve().parent.parent


def main():
    bolme = json.loads((PROJE / "data" / "bolme.json").read_text(encoding="utf-8"))
    sonuc = defaultdict(lambda: {"kare": 0, "kurulan": 0, "inlier": []})
    for d in sorted((PROJE / "data" / "labels" / "oneri_noktalar").glob("*/*.json")):
        v = d.parent.name
        n = json.loads(d.read_text(encoding="utf-8"))["noktalar"]
        s = sonuc[v]
        s["kare"] += 1
        if len(n) < 5:
            continue
        try:
            k = homografi_hesapla(np.array([p["px"] for p in n]), pitch.NOKTA_KOORD[[p["id"] for p in n]],
                                  ransac=True)
            s["kurulan"] += 1
            s["inlier"].append(k.icerdeki.mean())
        except ValueError:
            pass
    print(f"{'video':18s} {'küme':5s} {'kare':>4s} {'homografi':>9s} {'inlier ort.':>11s}")
    kume = defaultdict(lambda: [0, 0, []])
    for v in sorted(sonuc):
        s = sonuc[v]
        ort = np.mean(s["inlier"]) if s["inlier"] else 0
        print(f"{v:18s} {bolme.get(v, '?'):5s} {s['kare']:4d} {s['kurulan'] / s['kare']:8.0%} {ort:10.0%}")
        k = kume[bolme.get(v, "?")]
        k[0] += s["kare"]
        k[1] += s["kurulan"]
        k[2] += s["inlier"]
    for k, (n, kur, inl) in kume.items():
        print(f"  {k:5s}: homografi {kur}/{n} kare ({kur / n:.0%}), ortalama inlier {np.mean(inl):.0%}")


if __name__ == "__main__":
    main()
