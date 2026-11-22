"""
tohum_ozet.py — Eğitim rastgeleliği: her öğrenme eğrisi koşusunun 3 tohumdaki sonucu
====================================================================================

Bootstrap güven aralıkları (guven_araligi.py) TEST kümesinden gelen belirsizliği ölçer. Eğitimin kendi
rastgeleliği (başlangıç ağırlıkları, veri sırası, veri çoğaltma) ayrı bir kaynaktır: aynı tarif farklı
tohumla biraz farklı bir model üretir. Bu script, her koşunun orijinal + 2 tekrarını (tohum 1, 2) okuyup
önerisiz test (11 + 12) PCK@10'unun ortalamasını, standart sapmasını ve aralığını verir; öğrenme eğrisi
bulgularının (A düzleşiyor, A > B, sentetik ≈ 143 gerçek kare) tohumdan bağımsız olup olmadığına bakar.

Kullanım:  .venv\\Scripts\\python.exe -X utf8 src\\tohum_ozet.py
Çıktı   :  results\\tohum_ozet.json
"""
import json
from pathlib import Path

import numpy as np

PROJE = Path(__file__).resolve().parent.parent
GRUP, MET = "test_onerisiz", "PCK@10px"
KOSULAR = [("A", 39, "egri_A3"), ("A", 65, "egri_A5"), ("A", 89, "egri_A8"), ("A", 143, "v2"),
           ("B", 39, "egri_B3"), ("B", 65, "egri_B5"), ("B", 89, "egri_B8"), ("B", 143, "egri_Bhep")]


def main():
    d = json.loads((PROJE / "results" / "deneyler.json").read_text(encoding="utf-8"))
    sonuc = {}
    print(f"{'koşu':8s} {'kare':>4s} {'tohumlar (PCK@10 %)':>28s} {'ort.':>6s} {'std':>5s}")
    for egri, n, ad in KOSULAR:
        adlar = [f"unet_kp_{ad}", f"unet_kp_{ad}_t1", f"unet_kp_{ad}_t2"]
        v = [d[a][GRUP][MET] * 100 for a in adlar if a in d]
        sonuc[f"{egri}{n}"] = {"degerler": v, "ort": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else None}
        print(f"{egri}{n:<7d} {n:4d} {', '.join(f'{x:.1f}' for x in v):>28s} {np.mean(v):6.1f} "
              f"{(np.std(v, ddof=1) if len(v) > 1 else float('nan')):5.2f}")
    a39, a143 = sonuc["A39"], sonuc["A143"]
    b39, b143 = sonuc["B39"], sonuc["B143"]
    print(f"\nA: 143 − 39 kare (ortalama) {a143['ort'] - a39['ort']:+.1f} puan; "
          f"B: {b143['ort'] - b39['ort']:+.1f} puan; A − B 39 karede {a39['ort'] - b39['ort']:+.1f}, "
          f"143 karede {a143['ort'] - b143['ort']:+.1f}")
    (PROJE / "results" / "tohum_ozet.json").write_text(json.dumps(sonuc, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
