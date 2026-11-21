"""
egit_keypoint.py — Saha anahtar noktası modelini (YOLO-pose) eğitir
===================================================================

YOLO-POSE NEDİR?
  Normalde insan iskeleti bulur: her kişi için bir kutu + 17 eklem noktası.
  Biz ona şunu öğretiyoruz:  nesne = SAHA,  eklem noktaları = 35 SAHA NOKTASI.
  Aynı mimari, farklı problem → buna "yeniden amaçlandırma" denebilir.

  Her nokta için model iki şey tahmin eder:
     (x, y)  : noktanın yeri
     güven   : "bu nokta görüntüde görünüyor mu?" olasılığı (0-1)

EĞİTİM AYARLARI (neden böyle?)
  mosaic=0   : Mozaik çoğaltma 4 görüntüyü birleştirir; saha tek bütün bir sahne
               olduğu için anlamsız → kapalı.
  fliplr=0.5 : Yarı olasılıkla aynala. flip_idx (veri seti yaml'ında) sol ↔ sağ
               noktaları doğru değiştirir (bkz. pitch.FLIP_IDX).
  degrees=0  : Döndürme yok — yayın kamerası neredeyse hiç yan yatmaz.
  imgsz=960  : 640'ta 1 piksel hata ≈ orijinalde 2 piksel; daha hassas olsun diye büyük.

Kullanım:
    .venv\\Scripts\\python.exe src\\egit_keypoint.py --veri data\\synthetic\\saha_sentetik.yaml --ad sentetik_v1
    .venv\\Scripts\\python.exe src\\egit_keypoint.py --veri <gerçek.yaml> --model models\\saha_kp_sentetik_v1.pt --ad ince_ayar_v1

Çıktılar:
    runs\\<ad>\\                 Ultralytics eğitim klasörü (grafikler, results.csv, weights)
    models\\saha_kp_<ad>.pt      en iyi model (kopyası)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJE = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description="Saha keypoint modeli eğit (YOLO-pose)")
    ap.add_argument("--veri", required=True, help="veri seti yaml dosyası")
    ap.add_argument("--model", default="yolov8s-pose.pt",
                    help="başlangıç ağırlıkları (hazır COCO-pose ya da önceki model)")
    ap.add_argument("--ad", required=True, help="bu eğitimin adı")
    ap.add_argument("--epoch", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--sabir", type=int, default=0,
                    help="erken durdurma sabrı; 0 = KAPALI (Ultralytics'in pose metriği bu problemde "
                         "anlamsız olduğu için erken durdurma yanlış karar veriyor)")
    ap.add_argument("--isci", type=int, default=2, help="veri yükleyici işçi sayısı (RAM'e dikkat)")
    ap.add_argument("--devam", action="store_true",
                    help="yarıda kalan eğitimi runs\\<ad>\\weights\\last.pt'den sürdür (tüm ayarlar oradan gelir)")
    args = ap.parse_args()

    from ultralytics import YOLO
    if args.devam:
        # Eğitim kesilirse (elektrik, bellek hatası...) her epoch sonunda kaydedilen last.pt'den
        # optimizer durumu ve epoch sayısıyla birlikte KALDIĞI YERDEN devam edilir.
        # workers devam ederken değiştirilebilir: her işçi ayrı süreç ve ~600 MB RAM;
        # Ultralytics doğrulama için işçi sayısının 2 katını daha açar → 4 işçi ≈ 12 süreç ≈ 7.5 GB
        YOLO(str(PROJE / "runs" / args.ad / "weights" / "last.pt")).train(
            resume=True, workers=args.isci, patience=args.sabir)
        son = PROJE / "runs" / args.ad / "weights" / "last.pt"
        shutil.copy(son, PROJE / "models" / f"saha_kp_{args.ad}.pt")
        print("Model (son epoch):", PROJE / "models" / f"saha_kp_{args.ad}.pt")
        return

    model_yolu = Path(args.model)
    if not model_yolu.exists() and not model_yolu.is_absolute():
        model_yolu = PROJE / "models" / args.model   # hazır model buraya indirilir
    model = YOLO(str(model_yolu))
    model.train(
        data=str(Path(args.veri).resolve()),
        epochs=args.epoch,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.sabir,
        mosaic=0.0,
        fliplr=0.5,
        degrees=0.0,
        scale=0.3,
        translate=0.1,
        workers=args.isci,
        project=str(PROJE / "runs"),
        name=args.ad,
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    # best.pt DEĞİL last.pt: Ultralytics "en iyi"yi OKS-mAP ile seçer; bu metrik insan
    # iskeleti için tasarlandı (tolerans = nesne boyutu). Bizde "nesne" tüm görüntü olduğundan
    # anlamsız çıkıyor → best.pt 1. epoch'ta kaldı. Model seçimini kendi PCK'mızla yapıyoruz.
    son = PROJE / "runs" / args.ad / "weights" / "last.pt"
    hedef = PROJE / "models" / f"saha_kp_{args.ad}.pt"
    if son.exists():
        shutil.copy(son, hedef)
        print("Model (son epoch):", hedef)


if __name__ == "__main__":   # Windows'ta veri yükleyici işçileri için bu satır ŞART
    main()
