"""
test_homografi.py — Homografi kodunun doğruluğunu CEVABI BİLİNEN bir durumda test eder.

Fikir:
  1) Gerçek bir kamera açısını taklit eden bir H_gercek (metre → piksel) seçiyoruz.
  2) Kuş bakışı sahayı bu açıdan bakılıyormuş gibi büküyoruz → sahte yayın görüntüsü.
  3) Anahtar noktaların görüntüdeki yeri artık KESİN biliniyor; bunları "tıklanmış"
     sayıyoruz ve gerçekçi olsun diye ±1.5 piksel el titremesi ekliyoruz.
  4) Bizim kodumuz H'yi hesaplıyor; bilinen yerlere koyduğumuz sahte oyuncuları
     geri dönüştürüp kaç metre saptığına bakıyoruz.
  5) Bir de bilerek KÖTÜ nokta seçimi (hepsi aynı doğruda) deniyoruz: kod bunu yakalamalı.

Çalıştırma:  .venv\\Scripts\\python.exe src\\test_homografi.py
"""
from pathlib import Path

import cv2
import numpy as np

import pitch
from homografi import cizgileri_goruntuye_ciz, homografi_hesapla, noktalari_donustur

PROJE = Path(__file__).resolve().parent.parent
CIKTI = PROJE / "results" / "test_homografi"
CIKTI.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)  # sabit tohum → her çalıştırmada aynı sonuç

# --- 1) Sahte kamera: sahanın sol yarısına yukarıdan-yandan bakan yayın kamerası ---
# Sahadaki 4 köşe (metre) görüntüde bir yamuğa (trapez) denk gelsin:
#   uzak kenar (y=0) dar ve yukarıda, yakın kenar (y=68) geniş ve aşağıda → perspektif
saha_dortgen = np.float32([[0, 0], [60, 0], [60, 68], [0, 68]])
goruntu_dortgen = np.float32([[330, 140], [1150, 120], [1500, 700], [-150, 760]])
H_gercek = cv2.getPerspectiveTransform(saha_dortgen, goruntu_dortgen)  # metre → piksel
W, Hh = 1280, 720

# --- 2) Sahte yayın görüntüsünü oluştur ---
ayar = pitch.HaritaAyar()
harita = pitch.saha_ciz(ayar)
# mini harita pikseli → metre (ölçek ve kenar boşluğunu geri al)
piksel_to_metre = np.array([[1 / ayar.olcek, 0, -ayar.kenar / ayar.olcek],
                            [0, 1 / ayar.olcek, -ayar.kenar / ayar.olcek],
                            [0, 0, 1]])
sahte_yayin = cv2.warpPerspective(harita, H_gercek @ piksel_to_metre, (W, Hh),
                                  borderValue=(40, 40, 40))

# --- 3) "Tıklanan" noktalar: görüntüde görünen anahtar noktalar + el titremesi ---
id_listesi = [0, 1, 8, 9, 10, 11, 13, 12, 14, 15, 16, 19]   # sol yarı + orta çizgi
saha_np = pitch.NOKTA_KOORD[id_listesi]
tiklanan = noktalari_donustur(H_gercek, saha_np) + rng.normal(0, 1.5, (len(id_listesi), 2))
# Görüntünün dışına düşenleri at (gerçekte onları göremezdik)
gorunur = ((tiklanan[:, 0] >= 0) & (tiklanan[:, 0] < W) &
           (tiklanan[:, 1] >= 0) & (tiklanan[:, 1] < Hh))
print(f"Görünen nokta sayısı: {gorunur.sum()} / {len(id_listesi)}")

# --- 4) Bizim kodumuzla homografi hesapla ---
sonuc = homografi_hesapla(tiklanan[gorunur], saha_np[gorunur])
print(f"Kalibrasyon hatası : {sonuc.hata_metre:.3f} m  |  {sonuc.hata_piksel:.2f} piksel")

# Sahte oyuncular: rastgele saha konumları (metre) → görüntüye → bizim H ile geri
oyuncu_gercek = np.column_stack([rng.uniform(2, 58, 15), rng.uniform(5, 65, 15)])
oyuncu_piksel = noktalari_donustur(H_gercek, oyuncu_gercek) + rng.normal(0, 1.5, (15, 2))
oyuncu_tahmin = noktalari_donustur(sonuc.H, oyuncu_piksel)
hata = np.linalg.norm(oyuncu_tahmin - oyuncu_gercek, axis=1)
print(f"Oyuncu konum hatası: ortalama {hata.mean():.3f} m, en kötü {hata.max():.3f} m")

# Görsel kontrol: hesaplanan çizgiler (sarı) gerçek çizgilerin (beyaz) üstüne oturmalı
kontrol = cizgileri_goruntuye_ciz(sahte_yayin, sonuc.H_ters)
for p in tiklanan[gorunur]:
    cv2.circle(kontrol, tuple(np.round(p).astype(int)), 6, (0, 0, 255), -1)
cv2.imwrite(str(CIKTI / "sahte_yayin.png"), sahte_yayin)
cv2.imwrite(str(CIKTI / "kalibrasyon_kontrol.png"), kontrol)

# --- 5) Kötü durum testi: sadece orta çizgi üzerindeki noktalar ---
kotu_id = [14, 15, 16, 17]
try:
    homografi_hesapla(noktalari_donustur(H_gercek, pitch.NOKTA_KOORD[kotu_id]),
                      pitch.NOKTA_KOORD[kotu_id])
    print("HATA: dejenere durum yakalanmadı!")
except ValueError as e:
    print("Kötü nokta seçimi doğru şekilde yakalandı →", e)

assert hata.mean() < 0.5, "Ortalama konum hatası beklenenden büyük"
print("TEST BAŞARILI. Görseller:", CIKTI)
