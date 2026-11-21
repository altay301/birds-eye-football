"""
guven_araligi.py — Test sonuçları için %95 güven aralıkları (kare düzeyinde bootstrap)
===================================================================================

Neden?  Önerisiz test kümesi (11 + 12) 41 kare / 397 noktadan oluşur. Aynı karedeki noktalar
birbirinden bağımsız değildir (aynı kamera, aynı ışık), bu yüzden örnekleme birimi NOKTA değil
KAREDİR. Kareler yerine konularak yeniden örneklenir (B = 10 000) ve her örnekte metrik yeniden
hesaplanır; dağılımın %2.5 ve %97.5 yüzdelikleri güven aralığıdır.

Model KARŞILAŞTIRMALARI eşleştirilmiş (paired) bootstrap ile yapılır: her örnekte iki model aynı
karelerle ölçülür, farkın dağılımına bakılır. Aralık 0'ı içeriyorsa fark istatistiksel olarak
anlamlı değildir.

Metrikler keypoint_tahmin.degerlendir() ile BİREBİR aynı tanımlanır (her kare için ayrı çağrılır):
    PCK@10   = 10 px içinde bulunan nokta / tüm etiketli noktalar   (oranların değil, toplamların oranı)
    başarı   = homografi kuruldu VE ortalama hata < 1 m olan kare oranı

Kullanım:  .venv\\Scripts\\python.exe -X utf8 src\\guven_araligi.py
Çıktı   :  results\\guven_araligi.json
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np

from deney_degerlendir import ciftler
from keypoint_tahmin import degerlendir, model_yukle

PROJE = Path(__file__).resolve().parent.parent
CIKTI = PROJE / "results" / "guven_araligi.json"
GRUP = "test_onerisiz"
B = 10_000
MODELLER = ["unet_kp_sentetik_v1", "unet_kp_egri_A3", "unet_kp_egri_A5", "unet_kp_egri_A8", "unet_kp_v2",
            "unet_kp_egri_B3", "unet_kp_egri_B5", "unet_kp_egri_B8", "unet_kp_egri_Bhep",
            "unet_kp_gercek_v1", "saha_kp_gercek_v2"]
# (a, b): "a − b" farkı test edilir
KARSILASTIRMALAR = [
    ("unet_kp_v2", "unet_kp_egri_A3"),            # A: 39 → 143 gerçek kare kazandırıyor mu?
    ("unet_kp_egri_A3", "unet_kp_sentetik_v1"),   # 39 gerçek kare sadece-sentetiğe ne ekliyor?
    ("unet_kp_egri_Bhep", "unet_kp_egri_B3"),     # B: 39 → 143 gerçek kare
    ("unet_kp_egri_A3", "unet_kp_egri_B3"),       # 39 karede sentetiğin katkısı
    ("unet_kp_v2", "unet_kp_egri_Bhep"),          # 143 karede sentetiğin katkısı
    ("unet_kp_sentetik_v1", "unet_kp_egri_Bhep"), # sadece sentetik ≈ sadece 143 gerçek?
    ("unet_kp_gercek_v1", "saha_kp_gercek_v2"),   # U-Net vs YOLO-pose
]


def kare_kare(model_adi: str, pairs) -> np.ndarray:
    """Her kare için [etiketli nokta, ≤5 px, ≤10 px, homografi sayıldı mı, < 1 m başarı]."""
    yol = PROJE / "models" / f"{model_adi}.pt"
    model = model_yukle(str(yol), None, 0.5)
    satirlar = []
    for p in pairs:
        with contextlib.redirect_stdout(io.StringIO()):
            s = degerlendir(model, [p])
        if s is None:                      # atlanmış / boş kare
            continue
        n = s["etiketli_nokta"]
        satirlar.append([n, round(s["PCK@5px"] * n), round(s["PCK@10px"] * n),
                         s.get("homografi_kare", 0), round(s.get("basari_1m_oran", 0) * s.get("homografi_kare", 0))])
    return np.array(satirlar, dtype=float)


def metrikler(m: np.ndarray) -> dict:
    return {"PCK@5": m[:, 1].sum() / m[:, 0].sum(), "PCK@10": m[:, 2].sum() / m[:, 0].sum(),
            "basari_1m": m[:, 4].sum() / m[:, 3].sum()}


def main():
    pairs = ciftler(GRUP)
    veri = {}
    for ad in MODELLER:
        veri[ad] = kare_kare(ad, pairs)
        print(f"✓ {ad}: {len(veri[ad])} kare", flush=True)
    n_kare = len(next(iter(veri.values())))
    assert all(len(v) == n_kare for v in veri.values()), "modeller farklı sayıda kare değerlendirdi"

    rng = np.random.default_rng(0)
    idx = rng.integers(0, n_kare, size=(B, n_kare))     # tüm modeller AYNI örneklerle (eşleştirilmiş)
    boot = {}
    for ad, m in veri.items():
        s = m[idx]                                       # (B, n_kare, 5)
        boot[ad] = {"PCK@5": s[..., 1].sum(1) / s[..., 0].sum(1),
                    "PCK@10": s[..., 2].sum(1) / s[..., 0].sum(1),
                    "basari_1m": s[..., 4].sum(1) / np.maximum(s[..., 3].sum(1), 1)}

    def ci(x):
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    sonuc = {"grup": GRUP, "kare": n_kare, "nokta": int(veri[MODELLER[0]][:, 0].sum()), "B": B,
             "modeller": {}, "karsilastirmalar": []}
    for ad in MODELLER:
        nokta = metrikler(veri[ad])
        sonuc["modeller"][ad] = {k: {"deger": float(v), "ci95": ci(boot[ad][k])} for k, v in nokta.items()}
    for a, b in KARSILASTIRMALAR:
        for k in ("PCK@10", "basari_1m"):
            fark = boot[a][k] - boot[b][k]
            d = metrikler(veri[a])[k] - metrikler(veri[b])[k]
            sonuc["karsilastirmalar"].append({"a": a, "b": b, "metrik": k, "fark": float(d), "ci95": ci(fark),
                                               "anlamli": bool(ci(fark)[0] > 0 or ci(fark)[1] < 0)})
    CIKTI.write_text(json.dumps(sonuc, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{GRUP}: {n_kare} kare, {sonuc['nokta']} nokta, B = {B}")
    for ad, s in sonuc["modeller"].items():
        p, b_ = s["PCK@10"], s["basari_1m"]
        print(f"{ad:22s} PCK@10 {p['deger']:.1%} [{p['ci95'][0]:.1%}, {p['ci95'][1]:.1%}]   "
              f"<1 m {b_['deger']:.1%} [{b_['ci95'][0]:.1%}, {b_['ci95'][1]:.1%}]")
    print()
    for c in sonuc["karsilastirmalar"]:
        print(f"{c['a']:20s} − {c['b']:20s} {c['metrik']:9s} {c['fark']:+.1%} "
              f"[{c['ci95'][0]:+.1%}, {c['ci95'][1]:+.1%}] {'ANLAMLI' if c['anlamli'] else 'anlamlı değil'}")


if __name__ == "__main__":
    main()
