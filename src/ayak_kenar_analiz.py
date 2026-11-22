"""
ayak_kenar_analiz.py — Ayak noktası yanlılığı ve kenarda kesilen oyuncular (ölçülebilir hale getirilmiş)
=======================================================================================================

Haritadaki konum, kutunun ALT KENARININ ortasından (ayak) hesaplanır. İki olası hata kaynağı:

1) AYAK NOKTASI YANLILIĞI: tespit modelinin kutuları, etiketlerin kutularına göre sistematik olarak
   aşağıda/yukarıda bitiyorsa (bol ya da kısa kutu) oyuncular sistematik olarak kayar.
   Ölçüm: eşleşen kutularda  dy = (y2_tahmin − y2_etiket) / yükseklik_etiket  (etiket protokolünde alt kenar
   ayağa oturtuldu). Medyan dy'nin anlamlı olup olmadığı bootstrap ile; anlamlıysa düzeltme
   y2' = y2 − medyan_dy · h  VAL'de tahmin edilir, TEST'te metre hatası önce/sonra ölçülür.

2) KENARDA KESİLEN OYUNCU: görüntünün alt kenarı ayakları keserse "alt kenar" ayak değil kesim çizgisidir
   → oyuncu kameraya olduğundan yakın görünür. Gerçek ayak bilinmediği için BENZETİM: tam görünen etiketli
   oyuncuların kutusu yüksekliğin %60/75/90'ında yapay olarak kesilir; ayak tahmin yöntemleri:
     (a) saf      : kesim çizgisi (şu anki davranış)
     (b) oran     : karedeki tam kutuların medyan yükseklik/genişlik oranı × genişlik
     (c) komşu    : üst kenarı en yakın 3 tam kutunun medyan yüksekliği (perspektif: aynı derinlikte aynı boy)
   Gerçek ayakla metre farkı (elle etiketli homografiyle). Yöntem VAL'de seçilir, TEST'te raporlanır.

Kullanım:  .venv\\Scripts\\python.exe -X utf8 src\\ayak_kenar_analiz.py
Çıktı   :  results\\ayak_kenar.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from homografi import noktalari_donustur
from tespit_degerlendir import KISI, eslestir, tahmin_et, test_kareleri

PROJE = Path(__file__).resolve().parent.parent
MODEL = PROJE / "models" / "tespit_futbol_v3_1536.pt"
ESIK = 0.25


def ayak(k: np.ndarray) -> np.ndarray:
    return np.c_[(k[:, 0] + k[:, 2]) / 2, k[:, 3]]


def yanlilik_verisi(kareler, tahminler):
    """Her eşleşme: (kare no, dy oranı, gerçek kutu, tahmin kutu)."""
    out = []
    for i, (k, tah) in enumerate(zip(kareler, tahminler)):
        r = np.array([b["xyxy"] for b in k["kutular"] if b["sinif"] in KISI], float).reshape(-1, 4)
        t = [p for p in tah if p["sinif"] in KISI and p["guven"] >= ESIK]
        tx = np.array([p["xyxy"] for p in t]).reshape(-1, 4)
        d, esi = eslestir(tx, np.array([p["guven"] for p in t]), r, 0.5)
        for j in np.where(d)[0]:
            g, p = r[esi[j]], tx[j]
            out.append((i, (p[3] - g[3]) / max(g[3] - g[1], 1), g, p))
    return out


def metre_hatasi(kareler, veri, duzeltme: float = 0.0):
    e = []
    for i, _, g, p in veri:
        H = kareler[i]["H"]
        if H is None:
            continue
        p2 = p.copy()
        p2[3] = p[3] - duzeltme * (p[3] - p[1])
        a = noktalari_donustur(H, ayak(p2[None]))[0]
        b = noktalari_donustur(H, ayak(g[None]))[0]
        if np.isfinite(a).all() and np.isfinite(b).all():
            e.append(float(np.linalg.norm(a - b)))
    return np.array(e)


def kenar_benzetimi(kareler, kesimler=(0.6, 0.75, 0.9)):
    """Tam görünen etiketli oyuncuları yapay olarak keser; yöntemlerin metre hatası."""
    sonuc = {y: [] for y in ("saf", "oran", "komsu")}
    for k in kareler:
        H = k["H"]
        if H is None:
            continue
        kut = np.array([b["xyxy"] for b in k["kutular"] if b["sinif"] in ("oyuncu", "kaleci")], float).reshape(-1, 4)
        if len(kut) < 4:
            continue
        h = kut[:, 3] - kut[:, 1]
        w = kut[:, 2] - kut[:, 0]
        for i in range(len(kut)):
            digerleri = np.delete(np.arange(len(kut)), i)
            oran = float(np.median(h[digerleri] / np.maximum(w[digerleri], 1)))
            yakin = digerleri[np.argsort(np.abs(kut[digerleri, 1] - kut[i, 1]))[:3]]
            boy_komsu = float(np.median(h[yakin]))
            gercek = noktalari_donustur(H, ayak(kut[i][None]))[0]
            if not np.isfinite(gercek).all():
                continue
            for c in kesimler:
                kesik = kut[i].copy()
                kesik[3] = kut[i, 1] + c * h[i]
                tahmin = {"saf": kesik[3],
                          "oran": kut[i, 1] + max(oran * w[i], c * h[i]),        # kesimden yukarıda olamaz
                          "komsu": kut[i, 1] + max(boy_komsu, c * h[i])}
                for y, y2 in tahmin.items():
                    q = kesik.copy()
                    q[3] = y2
                    p = noktalari_donustur(H, ayak(q[None]))[0]
                    if np.isfinite(p).all():
                        sonuc[y].append((c, float(np.linalg.norm(p - gercek))))
    return {y: np.array(v) for y, v in sonuc.items()}


def main():
    rapor = {}
    tah = {}
    kk = {s: test_kareleri(s) for s in ("val", "test")}
    for s in kk:
        tah[s] = tahmin_et(MODEL, kk[s])

    # --- 1) Ayak noktası yanlılığı ---
    val = yanlilik_verisi(kk["val"], tah["val"])
    dy = np.array([v[1] for v in val])
    rng = np.random.default_rng(0)
    boot = [np.median(rng.choice(dy, len(dy))) for _ in range(5000)]
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    med = float(np.median(dy))
    anlamli = ci[0] > 0 or ci[1] < 0
    print(f"[1] VAL: {len(dy)} eşleşme, alt kenar farkı medyan {med * 100:+.2f}% kutu boyu, %95 GA "
          f"[{ci[0] * 100:+.2f}, {ci[1] * 100:+.2f}] → {'ANLAMLI' if anlamli else 'anlamlı değil'}")
    test = yanlilik_verisi(kk["test"], tah["test"])
    once = metre_hatasi(kk["test"], test)
    sonra = metre_hatasi(kk["test"], test, med)
    print(f"    TEST: ayak noktası farkı medyan {np.median(once):.3f} m → düzeltmeyle {np.median(sonra):.3f} m "
          f"(ortalama {once.mean():.3f} → {sonra.mean():.3f})")
    rapor["ayak_yanliligi"] = {"val_eslesme": len(dy), "medyan_dy": med, "ci95": ci, "anlamli": anlamli,
                               "test_once_medyan_m": float(np.median(once)), "test_sonra_medyan_m": float(np.median(sonra)),
                               "test_once_ort_m": float(once.mean()), "test_sonra_ort_m": float(sonra.mean())}

    # --- 2) Kenarda kesilen oyuncu benzetimi ---
    for s in ("val", "test"):
        b = kenar_benzetimi(kk[s])
        rapor[f"kenar_{s}"] = {}
        print(f"[2] {s.upper()} kesilmiş oyuncu benzetimi (metre hatası medyan; kesim = boyun %60 / %75 / %90'ı):")
        for y, v in b.items():
            satir = []
            for c in (0.6, 0.75, 0.9):
                e = v[v[:, 0] == c, 1]
                satir.append(float(np.median(e)))
            rapor[f"kenar_{s}"][y] = dict(zip(("0.60", "0.75", "0.90"), satir))
            print(f"    {y:6s} " + "  ".join(f"{x:.2f} m" for x in satir) + f"   (n={len(v) // 3})")
    (PROJE / "results" / "ayak_kenar.json").write_text(json.dumps(rapor, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
