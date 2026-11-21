"""
kutu_oneri.py — Tüm kareler için hazır YOLO ile kutu ÖNERİLERİ üretir
=====================================================================

Etiketlemeyi hızlandırmak için (model-destekli etiketleme): kutuları sıfırdan
çizmek yerine, hazır modelin bulduklarını öneri olarak kutu_etiketle.py'ye veririz.
Sen sadece yanlışları düzeltir, eksikleri eklersin.

Hazır (COCO) model sadece "person" ve "sports ball" bilir → herkes "oyuncu" olarak
gelir; kaleci/hakem sınıfını sen tıklayarak değiştirirsin.

Daha büyük model (yolov8m) kullanıyoruz: öneri ne kadar iyi olursa düzeltme o kadar az.

Kullanım:
    .venv\\Scripts\\python.exe src\\kutu_oneri.py
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

PROJE = Path(__file__).resolve().parent.parent
KARE = PROJE / "data" / "frames"
ONERI = PROJE / "data" / "labels" / "kutular_oneri"


def main():
    from ultralytics import YOLO
    model = YOLO(str(PROJE / "models" / "yolov8m.pt"))
    kareler = sorted(KARE.glob("*/*.jpg"))
    say = {"oyuncu": 0, "top": 0}
    for k in kareler:
        img = cv2.imdecode(np.fromfile(str(k), np.uint8), cv2.IMREAD_COLOR)
        s = model.predict(img, classes=[0, 32], conf=0.3, imgsz=1280, verbose=False)[0]
        kutular = []
        for xyxy, c in zip(s.boxes.xyxy.cpu().numpy(), s.boxes.cls.cpu().numpy()):
            sinif = "top" if int(c) == 32 else "oyuncu"
            say[sinif] += 1
            kutular.append({"sinif": sinif, "xyxy": [round(float(v), 1) for v in xyxy]})
        yol = ONERI / k.parent.name / f"{k.stem}.json"
        yol.parent.mkdir(parents=True, exist_ok=True)
        yol.write_text(json.dumps({"goruntu": k.name, "kutular": kutular}, ensure_ascii=False), encoding="utf-8")
    print(f"{len(kareler)} kare: {say['oyuncu']} kişi, {say['top']} top önerisi → {ONERI}")


if __name__ == "__main__":
    main()
