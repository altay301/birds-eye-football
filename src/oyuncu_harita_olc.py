"""
oyuncu_harita_olc.py — Kuş bakışı haritadaki OYUNCULARIN doğruluğu (uçtan uca, video hattıyla)
==============================================================================================

Projenin amacı oyuncuların sahadaki konumu. Bu script, video hattının (takip, yumuşatma, rol ve takım
kuralları dahil) haritaya koyduğu noktaları ELLE ETİKETLİ gerçekle karşılaştırır.

Yöntem
  * Kutuları elle etiketlenmiş val ve test kareleri (her biri bir maç videosunun belirli saniyesi).
  * Video o saniyeden ~4 sn önce başlatılır → takip/yumuşatma "ısınmış" olarak tam o kareye gelinir
    (işlenen son kare, etiketli kareyle piksel farkı kontrol edilerek doğrulanır).
  * GERÇEK konumlar: elle etiketlenen kutuların ayak noktası, elle etiketlenen saha noktalarından
    kurulan homografiyle sahaya taşınır (model hiç kullanılmaz).
  * TAHMİN: video hattının haritaya koyduğu noktalar (hakem ve top hariç; soluk noktalar dahil).
  * Eşleştirme: Macar algoritması, en fazla 3 m.

Ölçütler
  oyuncu bulma (recall)  : gerçek oyuncu + kalecilerin haritada bir noktayla eşleşen oranı
  takım renginde         : bulunan oyuncuların soluk değil, takım/kaleci rengiyle gösterilen oranı
  hakem görünme          : gerçek hakemlerin haritada (gizlenmemiş) bir noktayla eşleşen oranı (düşük iyi)
  sahte nokta            : haritadaki noktaların hiçbir gerçek kişiyle eşleşmeyen oranı (düşük iyi)
  konum hatası           : eşleşen oyuncularda metre (medyan)

Yapılandırmalar (aynı kareler):
  A  önceki sistem : hazır COCO YOLOv8m, kare bazında takım ataması
  C  kendi model   : tespit_futbol_v3_1536, kare bazında takım ataması (rol yok)
  B  şimdiki sistem: tespit_futbol_v3_1536 + iz bazında rol/takım kuralları + hıza duyarlı yumuşatma
     (B_sabit_yumusatma: aynısı, sabit katsayılı yumuşatmayla — hıza duyarlılığın etkisini ayırmak için)

Kullanım:  .venv\\Scripts\\python.exe -X utf8 src\\oyuncu_harita_olc.py
Çıktı   :  results\\oyuncu_harita.json
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

import pitch
from homografi import homografi_hesapla, noktalari_donustur
from keypoint_tahmin import model_yukle, oku
from sistem import VideoHatti, olcek_k, video_kareleri

PROJE = Path(__file__).resolve().parent.parent
KUTU = PROJE / "data" / "labels" / "kutular"
NOKTA = PROJE / "data" / "labels" / "noktalar"
VIDEO = PROJE / "data" / "raw_videos"
ISINMA_SN, KAPI_M, HEDEF_FPS = 4.0, 3.0, 12
YENI_MACLAR = {"20_deneme"}
KUMELER = {"val", "test"}          # --yeni ile {"yeni"}: sadece sonradan eklenen bağımsız maç
GELISTIRMEDE_GORULEN = {"12_gxsTs2qtV8A_t00090.0", "11_gsDyRUL4KX0_t00106.0"}   # video kuralı kliplerine düşen kareler
YAPILANDIRMA = {"A_onceki": ("yolov8m.pt", False, None), "C_kendi_model": ("tespit_futbol_v3_1536.pt", False, None),
                "B_sabit_yumusatma": ("tespit_futbol_v3_1536.pt", True, None),
                "B_simdiki": ("tespit_futbol_v3_1536.pt", True, 1.0)}   # (tespit modeli, iz kuralları, hız eşiği m)


def kareler():
    bolme = json.loads((PROJE / "data" / "bolme.json").read_text(encoding="utf-8"))
    out = []
    for e in sorted(KUTU.glob("*/*.json")):
        kume = bolme[e.parent.name]
        if e.parent.name in YENI_MACLAR:
            kume = "yeni"                    # sonradan eklenen bağımsız maç: ayrı raporlanır
        if kume not in KUMELER:
            continue
        nd = json.loads((NOKTA / e.parent.name / e.name).read_text(encoding="utf-8"))
        # Sadece iki etiketi de ONAYLANMIŞ kareler: atlanan karede kalmış öneri noktaları gerçek sayılmaz
        if nd.get("durum", "etiketli") != "etiketli" or \
                json.loads(e.read_text(encoding="utf-8")).get("durum", "etiketli") != "etiketli":
            continue
        n = nd["noktalar"]
        try:
            H = homografi_hesapla(np.array([p["px"] for p in n]), pitch.NOKTA_KOORD[[p["id"] for p in n]]).H
        except (ValueError, IndexError):
            continue
        kut = json.loads(e.read_text(encoding="utf-8"))["kutular"]
        def saha(siniflar):
            k = np.array([b["xyxy"] for b in kut if b["sinif"] in siniflar], float).reshape(-1, 4)
            p = noktalari_donustur(H, np.c_[(k[:, 0] + k[:, 2]) / 2, k[:, 3]]) if len(k) else np.empty((0, 2))
            ic = np.isfinite(p).all(1) & (p[:, 0] > -2) & (p[:, 0] < 107) & (p[:, 1] > -2) & (p[:, 1] < 70)
            return p[ic]
        out.append({"ad": e.stem, "video": e.parent.name, "kume": kume, "t": float(e.stem.rsplit("_t", 1)[1]),
                    "goruntu": PROJE / "data" / "frames" / e.parent.name / f"{e.stem}.jpg",
                    "oyuncu": saha({"oyuncu", "kaleci"}), "hakem": saha({"hakem"})})
    return out


def hatti_calistir(model, k, yolo, kararli, hiz_m=1.0):
    """Videoyu etiketli kareden ~4 sn önce başlatıp tam o kareye kadar işler; son karenin çıktısını döner."""
    cap = cv2.VideoCapture(str(VIDEO / f"{k['video']}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.release()
    adim = max(1, round(fps / HEDEF_FPS))
    hedef = round(k["t"] * fps)
    K = round(ISINMA_SN * fps / adim)
    bas_kare = max(0, hedef - adim * K)
    K = (hedef - bas_kare) // adim
    hat = VideoHatti(model, yolo, yumusat=True, kararli_takim=kararli, hiz_m=hiz_m,
                     fps=12.0 * olcek_k(VIDEO / f"{k['video']}.mp4", HEDEF_FPS))
    son = None
    for img, kesme, _ in video_kareleri(VIDEO / f"{k['video']}.mp4", (bas_kare + 0.01) / fps,
                                        (adim * K + 1.5) / fps, HEDEF_FPS):
        son = (img, *hat.isle(img, kesme))
    img, kal, _, oyuncular = son
    ref = oku(k["goruntu"])
    fark = float(np.abs(cv2.resize(img, (320, 180)).astype(float) - cv2.resize(ref, (320, 180)).astype(float)).mean())
    return kal, oyuncular, fark


def eslestir(tahmin: np.ndarray, gercek: np.ndarray):
    if not len(tahmin) or not len(gercek):
        return []
    d = np.linalg.norm(tahmin[:, None] - gercek[None], axis=2)
    r, c = linear_sum_assignment(d)
    return [(i, j, d[i, j]) for i, j in zip(r, c) if d[i, j] <= KAPI_M]


def main():
    import argparse
    global KUMELER, HEDEF_FPS, YAPILANDIRMA
    ap = argparse.ArgumentParser()
    ap.add_argument("--yeni", action="store_true", help="sadece sonradan eklenen bağımsız maç (maç 20)")
    ap.add_argument("--fps", type=float, default=12, help="işlenecek kare hızı (ayarlar süreye göre ölçeklenir)")
    ap.add_argument("--sadece", nargs="+", help="sadece bu yapılandırmalar (ör. B_simdiki)")
    args = ap.parse_args()
    if args.yeni:
        KUMELER = {"yeni"}
    HEDEF_FPS = args.fps
    if args.sadece:
        YAPILANDIRMA = {k: v for k, v in YAPILANDIRMA.items() if k in args.sadece}
    model = model_yukle(str(PROJE / "models" / "unet_kp_gercek_v1.pt"))
    kk = kareler()
    sonuc = {"kareler": [], "ozet": {}}
    for ad_y, (yolo, kararli, hiz_m) in YAPILANDIRMA.items():
        for k in kk:
            kal, oyuncular, fark = hatti_calistir(model, k, yolo, kararli, hiz_m)
            gosterilen = [o for o in oyuncular if o["sinif"] not in ("hakem", "top")]
            p = np.array([[o["x"], o["y"]] for o in gosterilen]).reshape(-1, 2)
            gercek = np.vstack([k["oyuncu"], k["hakem"]]) if len(k["hakem"]) else k["oyuncu"]
            n_oy = len(k["oyuncu"])
            es = eslestir(p, gercek) if kal is not None else []
            oy_es = [(i, j, d) for i, j, d in es if j < n_oy]
            sonuc["kareler"].append({
                "yapilandirma": ad_y, "kare": k["ad"], "kume": k["kume"], "piksel_farki": fark,
                "gelistirmede": k["ad"] in GELISTIRMEDE_GORULEN, "homografi": kal is not None,
                "gercek_oyuncu": n_oy, "gercek_hakem": len(k["hakem"]), "gosterilen": len(p),
                "bulunan_oyuncu": len(oy_es),
                "takim_renginde": sum(gosterilen[i]["sinif"] not in pitch.HARITADA_SOLUK for i, _, _ in oy_es),
                "gorunen_hakem": len(es) - len(oy_es), "eslesen": len(es),
                "hatalar": [float(d) for _, _, d in oy_es]})
            print(f"{ad_y:14s} {k['ad']:28s} fark {fark:5.1f} H {'✓' if kal is not None else '✗'} "
                  f"oyuncu {len(oy_es)}/{n_oy} hakem görünen {len(es) - len(oy_es)}/{len(k['hakem'])} "
                  f"gösterilen {len(p)}", flush=True)

    def ozet(satirlar):
        go, bo = sum(s["gercek_oyuncu"] for s in satirlar), sum(s["bulunan_oyuncu"] for s in satirlar)
        gh = sum(s["gercek_hakem"] for s in satirlar)
        g, e = sum(s["gosterilen"] for s in satirlar), sum(s["eslesen"] for s in satirlar)
        h = [x for s in satirlar for x in s["hatalar"]]
        return {"kare": len(satirlar), "homografi": sum(s["homografi"] for s in satirlar),
                "oyuncu_bulma": bo / max(go, 1),
                "takim_renginde": sum(s["takim_renginde"] for s in satirlar) / max(bo, 1),
                "hakem_gorunme": sum(s["gorunen_hakem"] for s in satirlar) / max(gh, 1),
                "sahte_nokta": (g - e) / max(g, 1), "konum_hatasi_medyan_m": float(np.median(h)) if h else None,
                "gercek_oyuncu": go, "gercek_hakem": gh}

    print(f"\n{'yapılandırma':14s} {'küme':18s} {'kare':>4s} {'H':>3s} {'oyuncu bulma':>12s} {'takım renginde':>14s} "
          f"{'hakem görünme':>13s} {'sahte nokta':>11s} {'konum (m)':>9s}")
    hizali = [s for s in sonuc["kareler"] if s["piksel_farki"] < 20]
    for ad_y in YAPILANDIRMA:
        for etiket, filtre in (("val+test", lambda s: True), ("test", lambda s: s["kume"] == "test"),
                               ("bağımsız (geliştirmede görülmemiş)", lambda s: not s["gelistirmede"])):
            sat = [s for s in hizali if s["yapilandirma"] == ad_y and filtre(s)]
            o = ozet(sat)
            sonuc["ozet"][f"{ad_y}/{etiket}"] = o
            print(f"{ad_y:14s} {etiket[:18]:18s} {o['kare']:4d} {o['homografi']:3d} {o['oyuncu_bulma']:12.1%} "
                  f"{o['takim_renginde']:14.1%} {o['hakem_gorunme']:13.1%} {o['sahte_nokta']:11.1%} "
                  f"{(o['konum_hatasi_medyan_m'] or float('nan')):9.2f}")
    print(f"\nHizalı kare: {len({s['kare'] for s in hizali})} / {len(kk)} (piksel farkı < 20)")
    ad = "oyuncu_harita_yeni.json" if KUMELER == {"yeni"} else "oyuncu_harita.json"
    if HEDEF_FPS != 12:
        ad = ad.replace(".json", f"_fps{HEDEF_FPS:g}.json")
    elif args.sadece:
        ad = ad.replace(".json", "_" + "_".join(args.sadece) + ".json")
    (PROJE / "results" / ad).write_text(json.dumps(sonuc, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
