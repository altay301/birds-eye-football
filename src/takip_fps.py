"""
takip_fps.py — İşlenen kare hızının (fps) takibe etkisi: 12 vs 25 fps
=====================================================================

Bütün kare başına ayarlar 12 fps'de seçildi; başka fps'de aynı SÜREYE denk gelecek şekilde ölçeklenir
(takip.ema_olcekle, sistem.olcek_k). Bu betik aynı klipleri iki hızda işler ve SÜREYE göre normalize
edilmiş takip ölçütlerini karşılaştırır (kare başına ölçütler fps'e göre haksız olurdu):

    kimlik / dk          : haritada görünen (gizli olmayan) farklı iz kimliği sayısı, oyun dakikası başına.
                           Aynı oyuncu kopup yeni kimlik alırsa artar → düşük iyi (parçalanma)
    iz ömrü (sn, medyan) : bir kimliğin haritada görünür kaldığı süre → yüksek iyi
    etiket değişimi / dk : bir izin gösterilen etiketinin (takım/rol) değişme sayısı, iz-dakika başına
                           → düşük iyi
    işlem süresi / sn    : video saniyesi başına işlem süresi

Klipler: val (karar için) ve test (rapor için) maçları; video_titreme.py ile aynı.
Kullanım:  .venv\\Scripts\\python.exe -X utf8 src\\takip_fps.py
Çıktı   :  results\\takip_fps.json
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from keypoint_tahmin import model_yukle
from sistem import VideoHatti, islenen_fps, olcek_k, video_kareleri

PROJE = Path(__file__).resolve().parent.parent
VIDEO = PROJE / "data" / "raw_videos"
KLIPLER = {"val": [("10_8_WZSknXW98", 60, 45), ("02_nagW6n39jCs", 120, 45)],
           "test": [("12_gxsTs2qtV8A", 60, 60), ("11_gsDyRUL4KX0", 90, 60), ("04_UDsByIrG9GE", 120, 60)]}
YOLO = "tespit_futbol_v3_1536.pt"
GIZLI = {"hakem", "top"}


def olc(model, ad: str, bas: float, sure: float, hedef_fps: float) -> dict:
    yol = VIDEO / f"{ad}.mp4"
    fps = islenen_fps(yol, hedef_fps)
    hat = VideoHatti(model, YOLO, yumusat=True, fps=12.0 * olcek_k(yol, hedef_fps))
    gorunme = defaultdict(int)                 # iz → haritada görünür olduğu kare sayısı
    son, degisim, iz_kare, oyun_kare = {}, 0, 0, 0
    t0 = time.time()
    for img, kesme, _ in video_kareleri(yol, bas, sure, hedef_fps):
        if kesme:
            son.clear()
        kal, _, oyuncular = hat.isle(img, kesme)
        if kal is None:
            continue
        oyun_kare += 1
        for o in oyuncular:
            if o["sinif"] in GIZLI or o.get("id") is None:
                continue
            anahtar = (ad, o["id"])
            gorunme[anahtar] += 1
            iz_kare += 1
            if anahtar in son and son[anahtar] != o["sinif"]:
                degisim += 1
            son[anahtar] = o["sinif"]
    sure_isl = time.time() - t0
    oyun_dk = oyun_kare / fps / 60
    omur = np.array(list(gorunme.values())) / fps
    return {"fps": fps, "oyun_sn": oyun_kare / fps, "kimlik": len(gorunme),
            "kimlik_dk": len(gorunme) / max(oyun_dk, 1e-9),
            "iz_omru_medyan_sn": float(np.median(omur)) if len(omur) else 0.0,
            "etiket_degisimi_dk": degisim / max(iz_kare / fps / 60, 1e-9),
            "islem_sn_video_sn": sure_isl / sure}


def main():
    model = model_yukle(str(PROJE / "models" / "unet_kp_gercek_v1.pt"))
    sonuc = {}
    print(f"{'küme':5s} {'klip':16s} {'fps':>5s} {'oyun sn':>8s} {'kimlik/dk':>10s} {'iz ömrü sn':>11s} "
          f"{'değişim/dk':>11s} {'işlem/sn':>9s}", flush=True)
    for kume, klipler in KLIPLER.items():
        for ad, bas, sure in klipler:
            for hedef in (12, 25):
                s = olc(model, ad, bas, sure, hedef)
                sonuc[f"{kume}/{ad}@{bas}s/{hedef}"] = s
                print(f"{kume:5s} {ad[:16]:16s} {s['fps']:5.1f} {s['oyun_sn']:8.1f} {s['kimlik_dk']:10.1f} "
                      f"{s['iz_omru_medyan_sn']:11.1f} {s['etiket_degisimi_dk']:11.2f} {s['islem_sn_video_sn']:9.2f}",
                      flush=True)
    for kume in KLIPLER:
        for hedef in (12, 25):
            s = [v for k, v in sonuc.items() if k.startswith(kume + "/") and k.endswith(f"/{hedef}")]
            print(f"{kume:5s} ORTALAMA {hedef:>9d} fps: kimlik/dk {np.mean([x['kimlik_dk'] for x in s]):.1f}  "
                  f"iz ömrü {np.mean([x['iz_omru_medyan_sn'] for x in s]):.1f} sn  "
                  f"değişim/dk {np.mean([x['etiket_degisimi_dk'] for x in s]):.2f}  "
                  f"işlem/sn {np.mean([x['islem_sn_video_sn'] for x in s]):.2f}")
    (PROJE / "results" / "takip_fps.json").write_text(json.dumps(sonuc, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
