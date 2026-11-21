"""
bolme.py — Maçları train / val / test kümelerine ayırır (MAÇA göre, kareye göre DEĞİL)
====================================================================================

Neden maça göre? Aynı maçın kareleri hem eğitimde hem testte olursa model o
stadyumu/ışığı/formaları "ezberler" ve test sonucu olduğundan iyi çıkar (veri sızıntısı).

Kısıtlar (kaynaklar.csv'deki notlardan):
  * 14 bir DERLEME (PL 2026/27 golleri) → 15, 16, 18 ile aynı kümede (train)
  * 06 ve 08 aynı stadyum (Montjuïc) → aynı kümede
  * 16 (Anfield ~101x68), 17 (Goodison ~100x68) → train (test ölçümü saha boyutundan etkilenmesin)
  * 01 → train (ornek1/ornek2 bu maçtan; eşikleri onlarla ayarladık → testte olursa yanlı)
  * Test: farklı lig + farklı yayıncı + bilerek bir ZOR durum (11: sarımsı çim)

Çalıştırınca:
  * data\\raw_videos\\kaynaklar.csv'deki 'kume' sütununu doldurur
  * data\\bolme.json yazar (kod buradan okur)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

PROJE = Path(__file__).resolve().parent.parent
KAYNAK = PROJE / "data" / "raw_videos" / "kaynaklar.csv"

BOLME = {
    "train": ["01", "03", "05", "06", "07", "08", "09", "13", "14", "15", "16", "17", "18"],
    "val":   ["02", "10", "19"],
    "test":  ["04", "11", "12"],
}


def video_kumesi() -> dict[str, str]:
    """Video klasör adı (ör. '07_h68azhf5Ik0') → küme. kaynaklar.csv'den okunur."""
    sonuc = {}
    with open(KAYNAK, encoding="utf-8-sig", newline="") as f:
        for satir in csv.DictReader(f, delimiter=";"):
            stem = Path(satir["dosya_adi"]).stem
            sonuc[stem] = satir["kume"]
    return sonuc


def main():
    no_to_kume = {no: k for k, nolar in BOLME.items() for no in nolar}
    with open(KAYNAK, encoding="utf-8-sig", newline="") as f:
        okuyucu = csv.DictReader(f, delimiter=";")
        alanlar = okuyucu.fieldnames
        satirlar = list(okuyucu)
    for s in satirlar:
        s["kume"] = no_to_kume[s["dosya_adi"][:2]]
    with open(KAYNAK, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=alanlar, delimiter=";")
        w.writeheader()
        w.writerows(satirlar)
    (PROJE / "data" / "bolme.json").write_text(
        json.dumps({Path(s["dosya_adi"]).stem: s["kume"] for s in satirlar}, indent=2), encoding="utf-8")
    for k in BOLME:
        maclar = [s["mac"] for s in satirlar if s["kume"] == k]
        print(f"{k:5s} ({len(maclar):2d}): {', '.join(maclar)}")


if __name__ == "__main__":
    main()
