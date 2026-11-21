"""
deney_degerlendir.py — Eğitilen modelleri AYNI kümelerde ölçüp tek tabloda toplar
==============================================================================

Gruplar (bolme.json'a göre):
    val          : 02, 10, 19   (model SEÇİMİNDE kullanıldı → iyimser)
    test_04      : 04           (model önerisiyle etiketlendi → öneriyi yapan modeli kayırır)
    test_onerisiz: 11, 12       (önerisiz etiketlendi → EN DÜRÜST sonuç, raporda ana sonuç)

Sonuçlar results\\deneyler.json'a yazılır (model adı → grup → metrikler); tekrar
çalıştırınca sadece verilen modeller güncellenir.

Kullanım:
    .venv\\Scripts\\python.exe -X utf8 src\\deney_degerlendir.py models\\unet_kp_v2.pt models\\unet_kp_egri_A3.pt
    .venv\\Scripts\\python.exe -X utf8 src\\deney_degerlendir.py --tablo     # sadece tabloyu yazdır
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

from keypoint_tahmin import KARE_KLASOR, ETIKET_KLASOR, degerlendir, model_yukle

PROJE = Path(__file__).resolve().parent.parent
SONUC = PROJE / "results" / "deneyler.json"
GRUPLAR = {"val": {"02", "10", "19"}, "test_04": {"04"}, "test_onerisiz": {"11", "12"}}


def ciftler(grup: str):
    out = []
    for e in sorted(ETIKET_KLASOR.glob("*/*.json")):
        if e.parent.name[:2] in GRUPLAR[grup]:
            g = KARE_KLASOR / e.parent.name / f"{e.stem}.jpg"
            if g.exists():
                out.append((g, e))
    return out


def tablo(sonuclar: dict):
    print(f"{'model':28s} {'grup':14s} {'bulunan':>8s} {'PCK@5':>7s} {'PCK@10':>7s} {'medyan m':>9s} {'<1 m':>6s}")
    for ad, gruplar in sonuclar.items():
        for g, s in gruplar.items():
            m = s.get("metre_hata_medyan")
            print(f"{ad:28s} {g:14s} {s['bulunan_oran']:8.1%} {s['PCK@5px']:7.1%} {s['PCK@10px']:7.1%} "
                  f"{(f'{m:.2f}' if m is not None else '-'):>9s} {s.get('basari_1m_oran', 0):6.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modeller", nargs="*")
    ap.add_argument("--tablo", action="store_true")
    args = ap.parse_args()
    sonuclar = json.loads(SONUC.read_text(encoding="utf-8")) if SONUC.exists() else {}
    for m in args.modeller:
        model = model_yukle(m, None, 0.5)
        ad = Path(m).stem
        sonuclar[ad] = {}
        for g in GRUPLAR:
            with contextlib.redirect_stdout(io.StringIO()):   # degerlendir() JSON basıyor; sustur
                s = degerlendir(model, ciftler(g))
            sonuclar[ad][g] = s
        SONUC.write_text(json.dumps(sonuclar, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"✓ {ad}", flush=True)
    tablo(sonuclar)


if __name__ == "__main__":
    main()
