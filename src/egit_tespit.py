"""
egit_tespit.py — Oyuncu / kaleci / hakem / top tespit modelini kendi kutularımızla eğitir (ince ayar)
===================================================================================================

Başlangıç: COCO'da eğitilmiş YOLOv8m (transfer öğrenme). COCO modeli "kişi" ve "top" bilir ama
oyuncu / kaleci / hakem ayrımı yapmaz ve futbol topunu küçük/bulanık halde sık kaçırır.
İnce ayarla 4 rol sınıfı öğretilir.

* imgsz 1280: top görüntüde ~5–15 px; 640'a küçültülünce kaybolur.
* Model seçimi val maçlarına (02, 10, 19) göre (Ultralytics: 0.1·mAP50 + 0.9·mAP50-95); test maçlarına
  (04, 11, 12) eğitim boyunca dokunulmaz → değerlendirme: tespit_degerlendir.py
* RAM sınırı (15 GB): workers 2, önbellek yok.

Kullanım:  .venv\\Scripts\\python.exe -X utf8 src\\egit_tespit.py --ad futbol_v1
Çıktı   :  models\\tespit_<ad>.pt
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJE = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ad", default="futbol_v1")
    ap.add_argument("--temel", default="yolov8m.pt", help="başlangıç ağırlıkları (models\\ içinde)")
    ap.add_argument("--epoch", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--isci", type=int, default=2)
    ap.add_argument("--veri", default="kutu.yaml", help="data\\kutu_yolo içindeki yaml (kutu_nadir.yaml: nadir sınıf çoğaltmalı)")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(str(PROJE / "models" / args.temel))
    model.train(data=str(PROJE / "data" / "kutu_yolo" / args.veri), epochs=args.epoch, imgsz=args.imgsz,
                batch=args.batch, workers=args.isci, cache=False, seed=0, deterministic=False,
                project=str(PROJE / "runs" / "tespit"), name=args.ad, exist_ok=True, plots=True,
                close_mosaic=10, patience=0, verbose=False)
    en_iyi = PROJE / "runs" / "tespit" / args.ad / "weights" / "best.pt"
    hedef = PROJE / "models" / f"tespit_{args.ad}.pt"
    shutil.copy2(en_iyi, hedef)
    print("→", hedef)


if __name__ == "__main__":
    main()
