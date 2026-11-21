"""
veri_hazirla.py — Elle etiketlenmiş saha noktalarını YOLO-pose eğitim setine çevirir
===================================================================================

Girdi : data\\labels\\noktalar\\<video>\\<kare>.json   (etiketle.py çıktısı)
        data\\bolme.json                               (video → train/val/test)
Çıktı : data\\real_kp\\images\\{train,val,test}\\*.jpg
        data\\real_kp\\labels\\{train,val,test}\\*.txt
        data\\real_kp\\saha_gercek.yaml

Etiket satırı (sentetik.py ile aynı format):
    0 0.5 0.5 1 1  x1 y1 v1 ... x35 y35 v35      (0-1 arası; işaretlenmeyen nokta → 0 0 0)

--sentetik N : Train kümesine N sentetik görüntü de ekler (gerçek + sentetik karışık eğitim).
--cogalt K   : Gerçek train karelerini K kez kopyalar (oversampling; sentetik eklenmeden ÖNCE).
--sinir N    : Train kümesinde video başına en fazla N kare (ÖĞRENME EĞRİSİ deneyi için:
               --sinir 3, 5, 10 ... ile farklı miktarda gerçek veriyle eğitip karşılaştır).

Kullanım:
    .venv\\Scripts\\python.exe src\\veri_hazirla.py
    .venv\\Scripts\\python.exe src\\veri_hazirla.py --sentetik 2000 --ad karisik
    .venv\\Scripts\\python.exe src\\veri_hazirla.py --sinir 5 --ad egri_5
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

import pitch

PROJE = Path(__file__).resolve().parent.parent
ETIKET = PROJE / "data" / "labels" / "noktalar"
KARE = PROJE / "data" / "frames"
SENTETIK = PROJE / "data" / "synthetic"


def etiket_satiri(noktalar: list[dict], w: int, h: int) -> str:
    v = np.zeros((pitch.NOKTA_SAYISI, 3))
    for n in noktalar:
        v[n["id"]] = [n["px"][0] / w, n["px"][1] / h, 2]
    return "0 0.5 0.5 1 1 " + " ".join(f"{a:.6f} {b:.6f} {int(c)}" for a, b, c in v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ad", default="gercek", help="çıktı klasörü adı: data\\real_kp_<ad>")
    ap.add_argument("--sentetik", type=int, default=0, help="train'e eklenecek sentetik görüntü sayısı")
    ap.add_argument("--sinir", type=int, default=0, help="train'de video başına en fazla kare (0 = hepsi)")
    ap.add_argument("--cogalt", type=int, default=1, help="gerçek train karelerini K kez kopyala (oversampling)")
    args = ap.parse_args()

    bolme = json.loads((PROJE / "data" / "bolme.json").read_text(encoding="utf-8"))
    cikti = PROJE / "data" / f"real_kp_{args.ad}"
    if cikti.exists():
        shutil.rmtree(cikti)
    sayac = {"train": 0, "val": 0, "test": 0}
    rnd = random.Random(0)
    for video, kume in bolme.items():
        dosyalar = sorted((ETIKET / video).glob("*.json"))
        etiketli = []
        for d in dosyalar:
            e = json.loads(d.read_text(encoding="utf-8"))
            if e.get("durum") == "etiketli" and e["noktalar"]:
                etiketli.append((d, e))
        if kume == "train" and args.sinir:
            rnd.shuffle(etiketli)
            etiketli = etiketli[:args.sinir]
        for d, e in etiketli:
            kaynak = KARE / video / f"{d.stem}.jpg"
            if not kaynak.exists():
                continue
            h, w = cv2.imdecode(np.fromfile(str(kaynak), np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_2).shape
            h, w = h * 2, w * 2
            (cikti / "images" / kume).mkdir(parents=True, exist_ok=True)
            (cikti / "labels" / kume).mkdir(parents=True, exist_ok=True)
            shutil.copy(kaynak, cikti / "images" / kume / kaynak.name)
            (cikti / "labels" / kume / f"{d.stem}.txt").write_text(etiket_satiri(e["noktalar"], w, h) + "\n")
            sayac[kume] += 1

    if args.cogalt > 1:
        # OVERSAMPLING: gerçek train karelerini K kez kopyala. 1500 sentetiğin yanında 100-150
        # gerçek kare "kaybolur"; kopyalayınca her epoch'ta gerçek kareler K kez görülür.
        img_k, lab_k = cikti / "images" / "train", cikti / "labels" / "train"
        for g in list(img_k.glob("*.jpg")):
            for k in range(1, args.cogalt):
                shutil.copy(g, img_k / f"{g.stem}_k{k}.jpg")
                shutil.copy(lab_k / f"{g.stem}.txt", lab_k / f"{g.stem}_k{k}.txt")
        print(f"Gerçek train kareleri x{args.cogalt} → {sayac['train'] * args.cogalt}")

    if args.sentetik:
        sg = sorted((SENTETIK / "images" / "train").glob("*.jpg"))
        rnd.shuffle(sg)
        (cikti / "images" / "train").mkdir(parents=True, exist_ok=True)
        (cikti / "labels" / "train").mkdir(parents=True, exist_ok=True)
        for g in sg[:args.sentetik]:
            shutil.copy(g, cikti / "images" / "train" / g.name)
            shutil.copy(SENTETIK / "labels" / "train" / f"{g.stem}.txt", cikti / "labels" / "train" / f"{g.stem}.txt")
        print(f"Train'e {min(args.sentetik, len(sg))} sentetik görüntü eklendi")

    yaml = cikti / "saha.yaml"
    cikti.mkdir(parents=True, exist_ok=True)
    yaml.write_text(f"path: {cikti.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                    f"kpt_shape: [{pitch.NOKTA_SAYISI}, 3]\nflip_idx: {pitch.FLIP_IDX}\nnames:\n  0: saha\n",
                    encoding="utf-8")
    print(f"Gerçek etiketli kare → train {sayac['train']}, val {sayac['val']}, test {sayac['test']}")
    print("Veri seti tanımı:", yaml)


if __name__ == "__main__":
    main()
