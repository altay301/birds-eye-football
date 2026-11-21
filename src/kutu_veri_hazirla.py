"""
kutu_veri_hazirla.py — Elle etiketlenen kutuları YOLO tespit (detection) formatına çevirir
========================================================================================

Bölme saha noktalarıyla AYNI (data\\bolme.json, maça göre): aynı maçın kareleri asla iki kümeye
dağılmaz. Sınıflar (rol tabanlı):  0 oyuncu · 1 kaleci · 2 hakem · 3 top

YOLO etiket satırı:  sınıf  x_merkez  y_merkez  genişlik  yükseklik   (hepsi görüntü boyutuna oranla, 0..1)

Kullanım:  .venv\\Scripts\\python.exe -X utf8 src\\kutu_veri_hazirla.py
Çıktı   :  data\\kutu_yolo\\{images,labels}\\{train,val,test}  +  data\\kutu_yolo\\kutu.yaml
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

PROJE = Path(__file__).resolve().parent.parent
ETIKET = PROJE / "data" / "labels" / "kutular"
KARE = PROJE / "data" / "frames"
CIKTI = PROJE / "data" / "kutu_yolo"
SINIFLAR = ["oyuncu", "kaleci", "hakem", "top"]


def main():
    bolme = json.loads((PROJE / "data" / "bolme.json").read_text(encoding="utf-8"))
    if CIKTI.exists():
        shutil.rmtree(CIKTI)
    say = {}
    for e in sorted(ETIKET.glob("*/*.json")):
        d = json.loads(e.read_text(encoding="utf-8"))
        if d.get("durum") != "etiketli":
            continue
        kume = bolme[e.parent.name]
        g = KARE / e.parent.name / f"{e.stem}.jpg"
        h, w = cv2.imdecode(np.fromfile(str(g), np.uint8), cv2.IMREAD_COLOR).shape[:2]
        satirlar = []
        for b in d["kutular"]:
            x1, y1, x2, y2 = b["xyxy"]
            x1, x2 = max(0.0, min(x1, x2)), min(float(w), max(x1, x2))
            y1, y2 = max(0.0, min(y1, y2)), min(float(h), max(y1, y2))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            satirlar.append(f"{SINIFLAR.index(b['sinif'])} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                            f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        for alt in ("images", "labels"):
            (CIKTI / alt / kume).mkdir(parents=True, exist_ok=True)
        ad = f"{e.parent.name}__{e.stem}"
        shutil.copy2(g, CIKTI / "images" / kume / f"{ad}.jpg")
        (CIKTI / "labels" / kume / f"{ad}.txt").write_text("\n".join(satirlar) + "\n", encoding="utf-8")
        say[kume] = say.get(kume, 0) + 1
    yaml = (f"path: {CIKTI.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
            f"names:\n" + "".join(f"  {i}: {s}\n" for i, s in enumerate(SINIFLAR)))
    (CIKTI / "kutu.yaml").write_text(yaml, encoding="utf-8")
    print(say, "→", CIKTI / "kutu.yaml")


if __name__ == "__main__":
    main()
