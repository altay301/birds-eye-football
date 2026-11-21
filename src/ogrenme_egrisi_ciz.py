"""
ogrenme_egrisi_ciz.py — results\\deneyler.json'dan öğrenme eğrisi grafiği
    x: eğitimde kullanılan GERÇEK etiketli kare sayısı
    y: önerisiz test (11+12) PCK@10
    A = sentetik ön-eğitim + sentetik karışık, B = sentetik yok
Çıktı: results\\ogrenme_egrisi.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJE = Path(__file__).resolve().parent.parent
d = json.loads((PROJE / "results" / "deneyler.json").read_text(encoding="utf-8"))
GRUP, MET = "test_onerisiz", "PCK@10px"

X = [39, 65, 89, 143]
A = ["unet_kp_egri_A3", "unet_kp_egri_A5", "unet_kp_egri_A8", "unet_kp_v2"]
B = ["unet_kp_egri_B3", "unet_kp_egri_B5", "unet_kp_egri_B8", "unet_kp_egri_Bhep"]
ya = [d[m][GRUP][MET] * 100 for m in A]
yb = [d[m][GRUP][MET] * 100 for m in B]
y0 = d["unet_kp_sentetik_v1"][GRUP][MET] * 100

MAVI, TURUNCU, METIN, IKINCIL, ZEMIN = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"
fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150, facecolor=ZEMIN)
ax.set_facecolor(ZEMIN)
ax.plot([0] + X, [y0] + ya, color=MAVI, lw=2, marker="o", ms=7, mec=ZEMIN, mew=2,
        label="A: sentetik ön-eğitim + gerçek")
ax.plot(X, yb, color=TURUNCU, lw=2, marker="o", ms=7, mec=ZEMIN, mew=2, label="B: sadece gerçek (sentetik yok)")
ax.annotate("sadece sentetik\n(0 gerçek kare)", (0, y0), xytext=(8, -28), textcoords="offset points",
            fontsize=8, color=IKINCIL)
ax.text(X[-1] + 3, ya[-1], f"A  {ya[-1]:.1f}", va="center", fontsize=9, color=METIN)
ax.text(X[-1] + 3, yb[-1], f"B  {yb[-1]:.1f}", va="center", fontsize=9, color=METIN)
ax.set_xlim(-5, 165)
ax.set_ylim(50, 90)
ax.set_xticks([0] + X)
ax.set_xlabel("Eğitimdeki gerçek etiketli kare sayısı (13 maç; 3 / 5 / 8 / tümü kare/maç)", color=IKINCIL, fontsize=9)
ax.set_ylabel("PCK@10 (%) — önerisiz test (11+12)", color=IKINCIL, fontsize=9)
ax.set_title("Öğrenme eğrisi: sentetik veri, ~140 gerçek kareye bedel", color=METIN, fontsize=11, loc="left")
ax.grid(axis="y", color="#e4e3df", lw=0.8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#c9c8c3")
ax.tick_params(colors=IKINCIL, labelsize=8)
ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=METIN)
fig.tight_layout()
out = PROJE / "results" / "ogrenme_egrisi.png"
fig.savefig(out, facecolor=ZEMIN)
print(out)
