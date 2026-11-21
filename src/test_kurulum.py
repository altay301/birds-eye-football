"""
Kurulum testi
-------------
1) PyTorch GPU'yu (RTX 4070) görüyor mu?
2) Hazır YOLO modeli örnek maç görüntülerinde oyuncuları bulabiliyor mu?

Çalıştırma (proje klasöründe):
    .venv\\Scripts\\python.exe src\\test_kurulum.py
Sonuçlar: results\\test_kurulum\\ klasörüne kaydedilir.
"""
from pathlib import Path

import torch
from ultralytics import YOLO

PROJE = Path(__file__).resolve().parent.parent

# --- 1) GPU kontrolü ---------------------------------------------------------
print("PyTorch sürümü :", torch.__version__)
print("CUDA var mı?   :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU            :", torch.cuda.get_device_name(0))
    # Küçük bir matris çarpımını GPU'da yaparak gerçekten çalıştığını doğrula
    a = torch.randn(2000, 2000, device="cuda")
    print("GPU hesap testi:", float((a @ a).sum()) != 0.0)

# --- 2) Hazır YOLO ile oyuncu tespiti -----------------------------------------
# yolov8n.pt: COCO veri setinde eğitilmiş küçük model. "person" = sınıf 0, "sports ball" = 32
model = YOLO(str(PROJE / "models" / "yolov8n.pt"))

ornekler = sorted((PROJE / "data" / "ornekler").glob("*.*"))
sonuclar = model.predict(
    source=[str(p) for p in ornekler],
    classes=[0, 32],          # sadece insan ve top
    conf=0.25,
    device=0 if torch.cuda.is_available() else "cpu",
    save=True,
    project=str(PROJE / "results"),
    name="test_kurulum",
    exist_ok=True,
    verbose=False,
)

for yol, s in zip(ornekler, sonuclar):
    insan = int((s.boxes.cls == 0).sum())
    top = int((s.boxes.cls == 32).sum())
    print(f"{yol.name}: {insan} kişi, {top} top bulundu")

print("Kutulu görüntüler:", PROJE / "results" / "test_kurulum")
