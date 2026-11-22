"""
tespit_degerlendir.py — Hazır COCO YOLOv8m ile kendi kutularımızla ince ayarlanan modeli karşılaştırır
====================================================================================================

Test kümesi: test maçlarından (04, 11, 12) 15 kare, elle etiketli kutular. Eğitimde hiç kullanılmadı.

COCO modeli "kişi" ve "top" bilir; oyuncu / kaleci / hakem ayrımı yapmaz. Bu yüzden:
  (1) ORTAK GÖREVLER (iki model): KİŞİ (oyuncu + kaleci + hakem tek sınıf) ve TOP
        AP50, AP50-95 (COCO tarzı, tüm noktalı interpolasyon) ve çalışma noktasında (güven ≥ 0.25)
        duyarlılık (recall) / kesinlik (precision)
  (2) SADECE YENİ MODEL: 4 sınıfın AP50'si, hakem ayrımı (hakemlerin kaçı "hakem" dendi, kaç oyuncu
        yanlışlıkla "hakem" dendi)
  (3) KONUM (metre): eşleşen kişi kutularında tahmin edilen ayak noktası ile etiketin ayak noktası
        arasındaki fark; homografi ELLE etiketlenen saha noktalarından kurulur → saf tespit hatası.

Belirsizlik: 15 kare üzerinde eşleştirilmiş bootstrap (B = 5000), farkın %95 güven aralığı.

Kullanım:  .venv\\Scripts\\python.exe -X utf8 src\\tespit_degerlendir.py --model models\\tespit_futbol_v1.pt
Çıktı   :  results\\tespit.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import pitch
from homografi import homografi_hesapla, noktalari_donustur

PROJE = Path(__file__).resolve().parent.parent
KARE = PROJE / "data" / "frames"
KUTU = PROJE / "data" / "labels" / "kutular"
NOKTA = PROJE / "data" / "labels" / "noktalar"
KISI = {"oyuncu", "kaleci", "hakem"}
ESIK = 0.25
B = 5000


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (n, 4), b: (m, 4) xyxy → (n, m)"""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    kes = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    alan = lambda k: (k[:, 2] - k[:, 0]) * (k[:, 3] - k[:, 1])
    return kes / (alan(a)[:, None] + alan(b)[None, :] - kes + 1e-9)


def eslestir(tahmin: np.ndarray, guven: np.ndarray, gercek: np.ndarray, esik: float):
    """Açgözlü eşleştirme (yüksek güvenden başlayarak). Döner: her tahmin için doğru mu, eşleşen gerçek indeksi."""
    sira = np.argsort(-guven)
    dogru, esi = np.zeros(len(tahmin), bool), np.full(len(tahmin), -1)
    if len(tahmin) and len(gercek):
        m = iou(tahmin, gercek)
        kullanildi = np.zeros(len(gercek), bool)
        for i in sira:
            adaylar = np.where(~kullanildi & (m[i] >= esik))[0]
            if len(adaylar):
                j = adaylar[np.argmax(m[i, adaylar])]
                kullanildi[j], dogru[i], esi[i] = True, True, j
    return dogru, esi


def ap(kareler: list, esik: float) -> float:
    """kareler: [(tahmin xyxy, güven, gerçek xyxy)] → AP (tüm noktalı interpolasyon)."""
    tum_g, tum_d, n_gercek = [], [], 0
    for t, g, r in kareler:
        d, _ = eslestir(t, g, r, esik)
        tum_g += list(g); tum_d += list(d); n_gercek += len(r)
    if n_gercek == 0:
        return float("nan")
    sira = np.argsort(-np.array(tum_g))
    d = np.array(tum_d, float)[sira]
    tp, fp = np.cumsum(d), np.cumsum(1 - d)
    rec, prec = tp / n_gercek, tp / np.maximum(tp + fp, 1e-9)
    prec = np.concatenate([[0], prec, [0]]); rec = np.concatenate([[0], rec, [rec[-1] if len(rec) else 0]])
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    return float(np.sum((rec[1:] - rec[:-1]) * prec[1:]))


def ap_50_95(kareler):
    return float(np.mean([ap(kareler, e) for e in np.arange(0.5, 0.96, 0.05)]))


def calisma_noktasi(kareler):
    tp = n_t = n_r = 0
    for t, g, r in kareler:
        s = g >= ESIK
        d, _ = eslestir(t[s], g[s], r, 0.5)
        tp += d.sum(); n_t += s.sum(); n_r += len(r)
    return {"recall": tp / max(n_r, 1), "precision": tp / max(n_t, 1)}


YENI_MACLAR = {"20_deneme"}   # sonradan eklenen bağımsız test maçı: eski test sonuçlarına karışmasın


def test_kareleri(kume: str = "test"):
    """kume: train / val / test (eski test maçları) / yeni (sonradan eklenen bağımsız maç)."""
    bolme = json.loads((PROJE / "data" / "bolme.json").read_text(encoding="utf-8"))
    out = []
    for e in sorted(KUTU.glob("*/*.json")):
        if kume == "yeni":
            if e.parent.name not in YENI_MACLAR:
                continue
        elif bolme.get(e.parent.name) != kume or e.parent.name in YENI_MACLAR:
            continue
        d = json.loads(e.read_text(encoding="utf-8"))
        n = json.loads((NOKTA / e.parent.name / e.name).read_text(encoding="utf-8"))["noktalar"]
        H = None
        if len(n) >= 4:
            try:
                H = homografi_hesapla(np.array([p["px"] for p in n]), pitch.NOKTA_KOORD[[p["id"] for p in n]]).H
            except ValueError:
                pass
        out.append({"goruntu": KARE / e.parent.name / f"{e.stem}.jpg", "kutular": d["kutular"], "H": H})
    return out


def tahmin_et(model_yolu: Path, kareler) -> list[list[dict]]:
    from ultralytics import YOLO
    model = YOLO(str(model_yolu))
    adlar = model.names
    # Her model EĞİTİLDİĞİ çözünürlükte çalıştırılır (COCO modeli için 1280)
    imgsz = max(1280, (getattr(model, "ckpt", None) or {}).get("train_args", {}).get("imgsz", 1280))
    sonuc = []
    for k in kareler:
        img = cv2.imdecode(np.fromfile(str(k["goruntu"]), np.uint8), cv2.IMREAD_COLOR)
        r = model.predict(img, conf=0.001, imgsz=imgsz, verbose=False, max_det=300)[0]
        liste = []
        for xyxy, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            ad = adlar[int(s)]
            ad = {"person": "kisi", "sports ball": "top"}.get(ad, ad)
            if ad in KISI | {"kisi", "top"}:
                liste.append({"xyxy": xyxy, "guven": float(c), "sinif": ad})
        sonuc.append(liste)
    return sonuc


def gorev(kareler, tahminler, gorev_adi: str, idx=None):
    """gorev_adi: 'kisi' (oyuncu+kaleci+hakem, sınıftan bağımsız), 'top', ya da tek bir rol sınıfı."""
    idx = range(len(kareler)) if idx is None else idx
    out = []
    for i in idx:
        if gorev_adi == "kisi":
            r = [b["xyxy"] for b in kareler[i]["kutular"] if b["sinif"] in KISI]
            t = [p for p in tahminler[i] if p["sinif"] in KISI | {"kisi"}]
        else:
            r = [b["xyxy"] for b in kareler[i]["kutular"] if b["sinif"] == gorev_adi]
            t = [p for p in tahminler[i] if p["sinif"] == gorev_adi]
        out.append((np.array([p["xyxy"] for p in t]).reshape(-1, 4), np.array([p["guven"] for p in t]),
                    np.array(r, float).reshape(-1, 4)))
    return out


def ayak_hatasi(kareler, tahminler, idx=None) -> list[float]:
    """Eşleşen kişi kutularında (güven ≥ 0.25, IoU ≥ 0.5) ayak noktası farkı, metre (elle homografi)."""
    out = []
    for i in (range(len(kareler)) if idx is None else idx):
        H = kareler[i]["H"]
        if H is None:
            continue
        r = np.array([b["xyxy"] for b in kareler[i]["kutular"] if b["sinif"] in KISI], float).reshape(-1, 4)
        t = [p for p in tahminler[i] if p["sinif"] in KISI | {"kisi"} and p["guven"] >= ESIK]
        tx = np.array([p["xyxy"] for p in t]).reshape(-1, 4)
        d, esi = eslestir(tx, np.array([p["guven"] for p in t]), r, 0.5)
        if not d.any():
            continue
        ayak = lambda k: np.c_[(k[:, 0] + k[:, 2]) / 2, k[:, 3]]
        a = noktalari_donustur(H, ayak(tx[d])); b = noktalari_donustur(H, ayak(r[esi[d]]))
        e = np.linalg.norm(a - b, axis=1)
        out += list(e[np.isfinite(e)])
    return out


def hakem_ayrimi(kareler, tahminler) -> dict:
    """Yeni model: kişi tespitleriyle eşleşen gerçek hakem/oyuncu/kalecilerin tahmin edilen sınıfları."""
    karisiklik = {g: {"oyuncu": 0, "kaleci": 0, "hakem": 0} for g in ("oyuncu", "kaleci", "hakem")}
    for k, tah in zip(kareler, tahminler):
        r = [b for b in k["kutular"] if b["sinif"] in KISI]
        t = [p for p in tah if p["sinif"] in KISI and p["guven"] >= ESIK]
        rx = np.array([b["xyxy"] for b in r], float).reshape(-1, 4)
        tx = np.array([p["xyxy"] for p in t]).reshape(-1, 4)
        d, esi = eslestir(tx, np.array([p["guven"] for p in t]), rx, 0.5)
        for i in np.where(d)[0]:
            karisiklik[r[esi[i]]["sinif"]][t[i]["sinif"]] += 1
    return karisiklik


def val_secimi(modeller: list[str]):
    """Varyant seçimi SADECE val maçlarında (02, 10, 19): test maçlarına bakılmaz."""
    kareler = test_kareleri("val")
    print(f"VAL: {len(kareler)} kare")
    print(f"{'model':28s} {'kişi':>6s} {'oyuncu':>7s} {'kaleci':>7s} {'hakem':>6s} {'top':>6s} {'top R@.25':>9s} {'ort.4':>6s}")
    sonuc = {}
    for m in modeller:
        tah = tahmin_et(Path(m), kareler)
        s = {g: ap(gorev(kareler, tah, g), 0.5) for g in ("kisi", "oyuncu", "kaleci", "hakem", "top")}
        s["top_R"] = calisma_noktasi(gorev(kareler, tah, "top"))["recall"]
        s["ort4"] = float(np.mean([s[g] for g in ("oyuncu", "kaleci", "hakem", "top")]))
        sonuc[Path(m).stem] = s
        print(f"{Path(m).stem:28s} {s['kisi']:6.3f} {s['oyuncu']:7.3f} {s['kaleci']:7.3f} {s['hakem']:6.3f} "
              f"{s['top']:6.3f} {s['top_R']:9.2f} {s['ort4']:6.3f}")
    (PROJE / "results" / "tespit_val_secimi.json").write_text(json.dumps(sonuc, indent=2), encoding="utf-8")


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--model", default=str(PROJE / "models" / "tespit_futbol_v3_1536.pt"))
    ap_.add_argument("--temel", default=str(PROJE / "models" / "yolov8m.pt"))
    ap_.add_argument("--secim", nargs="+", help="VAL maçlarında model seçimi: verilen modellerin sınıf AP50'leri")
    args = ap_.parse_args()
    if args.secim:
        val_secimi(args.secim)
        return
    kareler = test_kareleri()
    tah = {"coco": tahmin_et(Path(args.temel), kareler), "yeni": tahmin_et(Path(args.model), kareler)}
    sonuc = {"kare": len(kareler), "esik": ESIK, "modeller": {}, "farklar": {}}
    for m in tah:
        s = {}
        for g in ("kisi", "top"):
            k = gorev(kareler, tah[m], g)
            s[g] = {"AP50": ap(k, 0.5), "AP50_95": ap_50_95(k), **calisma_noktasi(k)}
        e = ayak_hatasi(kareler, tah[m])
        s["ayak_hatasi_m"] = {"medyan": float(np.median(e)), "p90": float(np.percentile(e, 90)), "n": len(e)}
        sonuc["modeller"][m] = s
    sonuc["modeller"]["yeni"]["sinif_AP50"] = {g: ap(gorev(kareler, tah["yeni"], g), 0.5)
                                               for g in ("oyuncu", "kaleci", "hakem", "top")}
    sonuc["modeller"]["yeni"]["hakem_ayrimi"] = hakem_ayrimi(kareler, tah["yeni"])

    # Eşleştirilmiş bootstrap: aynı yeniden örneklenmiş karelerde yeni − coco
    rng = np.random.default_rng(0)
    olc = {
        "kisi_AP50": lambda m, ix: ap(gorev(kareler, tah[m], "kisi", ix), 0.5),
        "top_AP50": lambda m, ix: ap(gorev(kareler, tah[m], "top", ix), 0.5),
        "top_recall": lambda m, ix: calisma_noktasi(gorev(kareler, tah[m], "top", ix))["recall"],
        "ayak_medyan_m": lambda m, ix: float(np.median(ayak_hatasi(kareler, tah[m], ix) or [np.nan])),
    }
    for ad, f in olc.items():
        farklar = []
        for _ in range(B):
            ix = rng.integers(0, len(kareler), len(kareler))
            farklar.append(f("yeni", ix) - f("coco", ix))
        farklar = np.array(farklar)
        farklar = farklar[np.isfinite(farklar)]
        nokta = f("yeni", range(len(kareler))) - f("coco", range(len(kareler)))
        ci = [float(np.percentile(farklar, 2.5)), float(np.percentile(farklar, 97.5))]
        sonuc["farklar"][ad] = {"fark": float(nokta), "ci95": ci, "anlamli": bool(ci[0] > 0 or ci[1] < 0)}
    (PROJE / "results" / "tespit.json").write_text(json.dumps(sonuc, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Test: {len(kareler)} kare (maç 04, 11, 12)\n")
    print(f"{'':6s} {'kişi AP50':>10s} {'AP50-95':>8s} {'R@.25':>6s} {'P@.25':>6s} | {'top AP50':>9s} {'R@.25':>6s} {'P@.25':>6s} | ayak hatası med/p90")
    for m, s in sonuc["modeller"].items():
        k, t, a = s["kisi"], s["top"], s["ayak_hatasi_m"]
        print(f"{m:6s} {k['AP50']:10.3f} {k['AP50_95']:8.3f} {k['recall']:6.2f} {k['precision']:6.2f} | "
              f"{t['AP50']:9.3f} {t['recall']:6.2f} {t['precision']:6.2f} | {a['medyan']:.2f} / {a['p90']:.2f} m")
    print("\nYeni model sınıf AP50:", {k: round(v, 3) for k, v in sonuc["modeller"]["yeni"]["sinif_AP50"].items()})
    print("Hakem ayrımı (satır: gerçek, sütun: tahmin):")
    for g, r in sonuc["modeller"]["yeni"]["hakem_ayrimi"].items():
        print(f"  {g:7s} {r}")
    print("\nFark (yeni − coco), %95 GA:")
    for ad, f in sonuc["farklar"].items():
        print(f"  {ad:14s} {f['fark']:+.3f} [{f['ci95'][0]:+.3f}, {f['ci95'][1]:+.3f}] {'ANLAMLI' if f['anlamli'] else 'anlamlı değil'}")


if __name__ == "__main__":
    main()
