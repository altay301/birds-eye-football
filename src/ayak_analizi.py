"""
ayak_analizi.py — Ayak noktası mı, kutu merkezi mi? (rapor 4.6) + etiketsiz tespit istatistikleri (4.4)
=====================================================================================================

Homografi SADECE zemindeki noktalar için doğrudur. Kutu merkezi oyuncunun belinde (~0.9 m yukarıda)
olduğundan sahaya taşınınca kameradan UZAĞA kayar. Bu script bu kaymayı gerçek karelerde ölçer:

  * H: ELLE etiketlenmiş noktalardan (model değil) → "doğru" homografi (ort. hata < 0.5 m olanlar)
  * Kutular: hazır YOLOv8m tespitleri (data\\labels\\kutular_oneri)
  * Her oyuncu için ayak = (x orta, y2), merkez = (x orta, y orta) → ikisi de sahaya → fark (m)
  * Makullük: sahaya düşen konumların ne kadarı saha sınırları içinde (±3 m pay)?

Kutu etiketi (insan) olmadığından tespit doğruluğu (mAP) ÖLÇÜLEMEZ; bunun yerine etiketsiz
istatistikler verilir: kare başına tespit sayısı, topun bulunduğu kare oranı.

Kullanım:  .venv\\Scripts\\python.exe -X utf8 src\\ayak_analizi.py
Çıktı   :  results\\ayak_analizi.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import pitch
from homografi import homografi_hesapla, noktalari_donustur

PROJE = Path(__file__).resolve().parent.parent
ETIKET = PROJE / "data" / "labels" / "noktalar"
KUTU = PROJE / "data" / "labels" / "kutular_oneri"
PAY = 3.0


def icinde(p: np.ndarray) -> np.ndarray:
    return ((p[:, 0] > -PAY) & (p[:, 0] < pitch.SAHA_UZUNLUK + PAY) &
            (p[:, 1] > -PAY) & (p[:, 1] < pitch.SAHA_GENISLIK + PAY))


def main():
    farklar, ayak_ic, merkez_ic, kamera_uzak = [], [], [], []
    kare_sayisi, top_kare, oyuncu_sayilari = 0, 0, []
    for e in sorted(ETIKET.glob("*/*.json")):
        d = json.loads(e.read_text(encoding="utf-8"))
        if d.get("durum") != "etiketli" or len(d["noktalar"]) < 5:
            continue
        k = KUTU / e.parent.name / e.name
        if not k.exists():
            continue
        g = np.array([n["px"] for n in d["noktalar"]])
        s = pitch.NOKTA_KOORD[[n["id"] for n in d["noktalar"]]]
        try:
            r = homografi_hesapla(g, s)
        except ValueError:
            continue
        if r.hata_metre > 0.5:
            continue
        kutular = json.loads(k.read_text(encoding="utf-8"))["kutular"]
        oyuncular = np.array([b["xyxy"] for b in kutular if b["sinif"] != "top"]).reshape(-1, 4)
        kare_sayisi += 1
        top_kare += any(b["sinif"] == "top" for b in kutular)
        oyuncu_sayilari.append(len(oyuncular))
        if not len(oyuncular):
            continue
        xo = (oyuncular[:, 0] + oyuncular[:, 2]) / 2
        ayak = noktalari_donustur(r.H, np.c_[xo, oyuncular[:, 3]])
        merkez = noktalari_donustur(r.H, np.c_[xo, (oyuncular[:, 1] + oyuncular[:, 3]) / 2])
        ok = np.isfinite(ayak).all(1) & np.isfinite(merkez).all(1)
        ayak, merkez = ayak[ok], merkez[ok]
        farklar += list(np.linalg.norm(merkez - ayak, axis=1))
        ayak_ic += list(icinde(ayak))
        merkez_ic += list(icinde(merkez))

    f = np.array(farklar)
    sonuc = {
        "kare": kare_sayisi,
        "oyuncu": int(len(f)),
        "merkez_ayak_farki_m": {"medyan": float(np.median(f)), "ortalama": float(f.mean()),
                                "p90": float(np.percentile(f, 90)), "en_cok": float(f.max())},
        "saha_icinde_oran": {"ayak": float(np.mean(ayak_ic)), "merkez": float(np.mean(merkez_ic))},
        "kare_basina_oyuncu_medyan": float(np.median(oyuncu_sayilari)),
        "top_bulunan_kare_orani": top_kare / kare_sayisi,
    }
    (PROJE / "results" / "ayak_analizi.json").write_text(json.dumps(sonuc, indent=2, ensure_ascii=False),
                                                         encoding="utf-8")
    print(json.dumps(sonuc, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
