"""
ogrenme_egrisi_ciz.py — Öğrenme eğrisi grafiği (results\\guven_araligi.json'dan)
===============================================================================

Sol panel  : önerisiz test (11 + 12) PCK@10, eğitimdeki GERÇEK etiketli kare sayısına göre.
    A : sentetik ön-eğitimli modelden başla + sentetik/gerçek karışık eğitim
    B : sentetik yok (ImageNet kodlayıcı + sadece gerçek)
    Kesikli yatay çizgi: sadece sentetikle eğitilmiş model (0 gerçek kare). Bu model A/B
    koşularıyla AYNI tarif ve hesap bütçesiyle eğitilmediği için eğriye BAĞLANMAZ; referanstır.
Sağ panel  : eşleştirilmiş (paired) bootstrap ile model farkları ve %95 güven aralıkları.
    Tek tek modellerin güven aralıkları çok geniştir (~±12 puan) çünkü kareler arası zorluk
    farkı büyüktür; ama bu zorluk TÜM modeller için ortaktır. Karşılaştırmada doğru soru
    "aynı karelerde fark ne?" olduğundan sağ panel eşleştirilmiş farkları gösterir.

Kullanım:  .venv\\Scripts\\python.exe -X utf8 src\\ogrenme_egrisi_ciz.py            # İngilizce
           .venv\\Scripts\\python.exe -X utf8 src\\ogrenme_egrisi_ciz.py --dil tr   # Türkçe
Çıktı   :  results\\learning_curve.png  /  results\\ogrenme_egrisi.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJE = Path(__file__).resolve().parent.parent
X = [39, 65, 89, 143]
A = ["unet_kp_egri_A3", "unet_kp_egri_A5", "unet_kp_egri_A8", "unet_kp_v2"]
B = ["unet_kp_egri_B3", "unet_kp_egri_B5", "unet_kp_egri_B8", "unet_kp_egri_Bhep"]
REF = "unet_kp_sentetik_v1"
# Sağ panelde gösterilen farklar: (a, b, İngilizce etiket, Türkçe etiket)
FARKLAR = [
    ("unet_kp_egri_A3", "unet_kp_egri_B3", "A − B at 39 real frames", "A − B, 39 gerçek kare"),
    ("unet_kp_v2", "unet_kp_egri_Bhep", "A − B at 143 real frames", "A − B, 143 gerçek kare"),
    ("unet_kp_v2", "unet_kp_egri_A3", "A: 143 − 39 real frames", "A: 143 − 39 gerçek kare"),
    ("unet_kp_egri_Bhep", "unet_kp_egri_B3", "B: 143 − 39 real frames", "B: 143 − 39 gerçek kare"),
    ("unet_kp_egri_A3", "unet_kp_sentetik_v1", "A (39) − synthetic only", "A (39) − sadece sentetik"),
    ("unet_kp_sentetik_v1", "unet_kp_egri_Bhep", "synthetic only − B (143)", "sadece sentetik − B (143)"),
]

METIN = {
    "en": {"A": "A: synthetic pre-training + mixed synthetic/real",
           "B": "B: real only (ImageNet encoder, no synthetic data)",
           "ref": "synthetic only, 0 real frames",
           "x": "Real labelled training frames\n(13 matches; at most 3 / 5 / 8 / all per match)",
           "y": "PCK@10 (%), suggestion-free test set (11+12)",
           "t1": "(a) Learning curve",
           "t2": "(b) Paired differences, 95% bootstrap CI",
           "x2": "Difference in PCK@10 (percentage points)",
           "out": "learning_curve.png"},
    "tr": {"A": "A: sentetik ön-eğitim + sentetik/gerçek karışık",
           "B": "B: sadece gerçek (ImageNet kodlayıcı, sentetik yok)",
           "ref": "sadece sentetik, 0 gerçek kare",
           "x": "Eğitimdeki gerçek etiketli kare sayısı\n(13 maç; maç başına en fazla 3 / 5 / 8 / tümü)",
           "y": "PCK@10 (%), önerisiz test (11+12)",
           "t1": "(a) Öğrenme eğrisi",
           "t2": "(b) Eşleştirilmiş farklar, %95 bootstrap GA",
           "x2": "PCK@10 farkı (yüzde puan)",
           "out": "ogrenme_egrisi.png"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dil", choices=["en", "tr"], default="en")
    dil = ap.parse_args().dil
    t = METIN[dil]
    j = json.loads((PROJE / "results" / "guven_araligi.json").read_text(encoding="utf-8"))
    d = j["modeller"]
    karsi = {(c["a"], c["b"]): c for c in j["karsilastirmalar"] if c["metrik"] == "PCK@10"}

    MAVI, TURUNCU, GRI = "#2a78d6", "#eb6834", "#52514e"
    METN, IKINCIL, ZEMIN, IZGARA, CERCEVE = "#0b0b0b", "#52514e", "#ffffff", "#e4e3df", "#c9c8c3"
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=200, facecolor=ZEMIN,
                                 gridspec_kw={"width_ratios": [1.25, 1]})

    # --- (a) öğrenme eğrisi ---------------------------------------------------------------
    ref = d[REF]["PCK@10"]["deger"] * 100
    ax.hlines(ref, 30, 143, color=GRI, lw=1.2, ls=(0, (5, 3)), zorder=1)
    ax.text(31, ref + 0.7, f"{t['ref']}: {ref:.1f}", fontsize=7.5, color=IKINCIL, va="bottom")
    for adlar, renk, etiket, kayma in ((A, MAVI, t["A"], 0), (B, TURUNCU, t["B"], -1.2)):
        y = [d[m]["PCK@10"]["deger"] * 100 for m in adlar]
        ax.plot(X, y, color=renk, lw=2, marker="o", ms=6, zorder=3, label=etiket)
        ax.text(X[-1] + 4, y[-1] + kayma, f"{y[-1]:.1f}", va="center", fontsize=8.5, color=METN)
    ax.set_xlim(28, 160)
    ax.set_ylim(55, 85)
    ax.set_xticks(X)
    ax.set_xlabel(t["x"], color=IKINCIL, fontsize=8)
    ax.set_ylabel(t["y"], color=IKINCIL, fontsize=8)
    ax.set_title(t["t1"], color=METN, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right", labelcolor=METN)

    # --- (b) eşleştirilmiş farklar (forest plot) -------------------------------------------
    for i, (a, b, en, tr) in enumerate(FARKLAR):
        c = karsi[(a, b)]
        f, lo, hi = c["fark"] * 100, c["ci95"][0] * 100, c["ci95"][1] * 100
        yi = len(FARKLAR) - 1 - i
        bx.hlines(yi, lo, hi, color=METN, lw=1.5)
        bx.plot(f, yi, "o", color=METN if c["anlamli"] else ZEMIN, mec=METN, ms=6, zorder=3)
        bx.text(30, yi, f"{f:+.1f}  [{lo:+.1f}, {hi:+.1f}]", va="center", fontsize=7.5, color=IKINCIL)
    bx.axvline(0, color=GRI, lw=1, ls=(0, (5, 3)))
    bx.set_yticks(range(len(FARKLAR)))
    bx.set_yticklabels([(en if dil == "en" else tr) for _, _, en, tr in FARKLAR][::-1], fontsize=7.5,
                       color=METN)
    bx.set_xlim(-8, 46)
    bx.set_xticks([0, 10, 20, 30])
    bx.set_ylim(-0.7, len(FARKLAR) - 0.3)
    bx.set_xlabel(t["x2"], color=IKINCIL, fontsize=8)
    bx.set_title(t["t2"], color=METN, fontsize=10, loc="left")

    for e in (ax, bx):
        e.set_facecolor(ZEMIN)
        e.grid(axis="x" if e is bx else "y", color=IZGARA, lw=0.8)
        for s in ("top", "right"):
            e.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            e.spines[s].set_color(CERCEVE)
        e.tick_params(colors=IKINCIL, labelsize=7.5)
    fig.tight_layout()
    out = PROJE / "results" / t["out"]
    fig.savefig(out, facecolor=ZEMIN)
    print(out)


if __name__ == "__main__":
    main()
