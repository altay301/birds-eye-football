"""
unet_keypoint.py — KENDİ YAZDIĞIMIZ saha noktası modeli: ResNet-18 + U-Net + ısı haritası
========================================================================================

Neden? YOLO-pose hazır bir mimari. Bu dosyada modeli PyTorch ile kendimiz kuruyoruz:
mimari, kayıp fonksiyonu, veri çoğaltma ve eğitim döngüsü tamamen bizim → raporda
"YOLO-pose (hazır) vs U-Net (kendi modelimiz)" karşılaştırması.

MİMARİ
    girdi 640x352 ─► ResNet-18 KODLAYICI (ImageNet'te önceden eğitilmiş)
                       c1: 1/4  (64 kanal)   ← "nerede" bilgisi (ince ayrıntı)
                       c2: 1/8  (128)
                       c3: 1/16 (256)
                       c4: 1/32 (512)        ← "ne" bilgisi (anlam)
                   ─► U-Net ÇÖZÜCÜ: büyüt + aynı boyuttaki kodlayıcı haritasıyla BİRLEŞTİR
                       (skip connection — U-Net'i U-Net yapan şey)
                   ─► 35 kanallı ISI HARİTASI, 1/4 boyut (160x88)

ISI HARİTASI (heatmap) nedir?
    Her saha noktası için ayrı bir "görüntü". Noktanın olduğu yerde Gauss tepesi (1.0),
    uzaklaştıkça 0. Nokta görünmüyorsa harita tamamen 0.
    Tahmin: her kanalın EN PARLAK pikseli = nokta; parlaklık = güven.
    Neden koordinatı doğrudan tahmin etmek yerine bu? Evrişimli ağlar (CNN) "nerede"
    sorusuna uzamsal haritalarla daha iyi cevap verir; ayrıca "görünmüyor" durumu
    doğal olarak ifade edilir (karanlık harita).

KAYIP FONKSİYONU: ağırlıklı ortalama kare hata (MSE)
    kayıp = ortalama( (tahmin − hedef)² · (1 + 20·hedef) )
    Haritanın %99'u 0 → düz MSE'de model "her yere 0 de" diyerek kolayca düşük kayıp
    alır. (1 + 20·hedef) çarpanı tepe çevresindeki hatayı 21 kat önemli yapar.

Kullanım:
    .venv\\Scripts\\python.exe src\\unet_keypoint.py --veri data\\synthetic --ad sentetik_v1 --epoch 15
    (veri klasörü YOLO-pose formatında olmalı: images/{train,val}, labels/{train,val})
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import pitch

PROJE = Path(__file__).resolve().parent.parent
GIRDI_W, GIRDI_H = 640, 352          # 32'ye bölünebilir olmalı (ResNet 5 kez yarıya böler)
ADIM = 4                              # ısı haritası girdinin 1/4'ü
HM_W, HM_H = GIRDI_W // ADIM, GIRDI_H // ADIM
SIGMA = 2.0                           # Gauss tepesinin genişliği (ısı haritası pikseli)
N = pitch.NOKTA_SAYISI
ORT = np.array([0.485, 0.456, 0.406], np.float32)   # ImageNet normalizasyonu (kodlayıcı
STD = np.array([0.229, 0.224, 0.225], np.float32)   # bu istatistiklerle eğitildi)


# ---------------------------------------------------------------------------
# VERİ
# ---------------------------------------------------------------------------
def etiket_oku(yol: Path) -> np.ndarray:
    """YOLO-pose satırı → (35, 3) [x, y, v], x/y 0-1 arası."""
    p = yol.read_text().split()
    return np.array(p[5:], np.float32).reshape(N, 3)


class SahaVeri(Dataset):
    def __init__(self, klasor: Path, kume: str, egitim: bool):
        self.gorseller = sorted((klasor / "images" / kume).glob("*.jpg"))
        self.etiketler = [klasor / "labels" / kume / f"{g.stem}.txt" for g in self.gorseller]
        self.egitim = egitim
        ys, xs = np.mgrid[0:HM_H, 0:HM_W].astype(np.float32)
        self.ys, self.xs = ys, xs

    def __len__(self):
        return len(self.gorseller)

    def __getitem__(self, i):
        img = cv2.imread(str(self.gorseller[i]))
        img = cv2.resize(img, (GIRDI_W, GIRDI_H), interpolation=cv2.INTER_AREA)
        kp = etiket_oku(self.etiketler[i])
        if self.egitim:
            img, kp = self.cogalt(img, kp)
        x = ((img[..., ::-1].astype(np.float32) / 255 - ORT) / STD).transpose(2, 0, 1)
        hm = np.zeros((N, HM_H, HM_W), np.float32)
        for j, (u, v, g) in enumerate(kp):
            if g > 0:
                cx, cy = u * HM_W, v * HM_H
                hm[j] = np.exp(-((self.xs - cx) ** 2 + (self.ys - cy) ** 2) / (2 * SIGMA ** 2))
        return torch.from_numpy(x.copy()), torch.from_numpy(hm), torch.from_numpy(kp)

    @staticmethod
    def cogalt(img, kp):
        """Veri çoğaltma: renk/parlaklık + yatay aynalama (flip_idx ile)."""
        rng = np.random
        img = cv2.convertScaleAbs(img, alpha=rng.uniform(0.7, 1.3), beta=rng.uniform(-30, 30))
        if rng.rand() < 0.5:   # renk tonu kaydırma (farklı çim renkleri)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
            hsv[..., 0] = (hsv[..., 0] + rng.randint(-8, 9)) % 180
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        if rng.rand() < 0.5:   # AYNALAMA: sol kale sağ kale olur → ID'ler de yer değiştirir
            img = img[:, ::-1]
            yeni = np.zeros_like(kp)
            for j in range(N):
                if kp[j, 2] > 0:
                    yeni[pitch.FLIP_IDX[j]] = [1 - kp[j, 0], kp[j, 1], kp[j, 2]]
            kp = yeni
        return np.ascontiguousarray(img), kp


# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------
class Yukari(nn.Module):
    """Çözücü bloğu: 2 kat büyüt → kodlayıcıdan gelen haritayla birleştir → 2 evrişim."""

    def __init__(self, giris, atlama, cikis):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(giris + atlama, cikis, 3, padding=1, bias=False), nn.BatchNorm2d(cikis), nn.ReLU(inplace=True),
            nn.Conv2d(cikis, cikis, 3, padding=1, bias=False), nn.BatchNorm2d(cikis), nn.ReLU(inplace=True))

    def forward(self, x, atla):
        x = F.interpolate(x, size=atla.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, atla], dim=1))   # ← skip connection


class SahaUNet(nn.Module):
    def __init__(self, onegitimli: bool = True):
        super().__init__()
        import torchvision
        r = torchvision.models.resnet18(weights="IMAGENET1K_V1" if onegitimli else None)
        self.giris = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)   # 1/4
        self.k1, self.k2, self.k3, self.k4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.y3 = Yukari(512, 256, 256)   # 1/32 → 1/16
        self.y2 = Yukari(256, 128, 128)   # 1/16 → 1/8
        self.y1 = Yukari(128, 64, 64)     # 1/8  → 1/4
        self.cikis = nn.Conv2d(64, N, 1)  # 35 ısı haritası

    def forward(self, x):
        c1 = self.k1(self.giris(x))
        c2 = self.k2(c1)
        c3 = self.k3(c2)
        c4 = self.k4(c3)
        d = self.y3(c4, c3)
        d = self.y2(d, c2)
        d = self.y1(d, c1)
        return self.cikis(d)


def kayip_fonksiyonu(tahmin, hedef):
    return (((tahmin - hedef) ** 2) * (1 + 20 * hedef)).mean()


# ---------------------------------------------------------------------------
# ISI HARİTASINDAN NOKTAYA
# ---------------------------------------------------------------------------
def haritadan_nokta(hm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    (35, H, W) ısı haritası → (35, 2) nokta (0-1 arası) ve (35,) güven.

    ALT-PİKSEL HASSASİYET: En parlak piksel tek başına kaba (1/4 ölçekte 1 piksel =
    orijinalde ~8 piksel). Gauss eğrisinin LOGARİTMASI tam bir paraboldür → en parlak
    piksel (c) ve iki komşusundan (l, r) parabolün tepesi kesin bulunur:
        kayma = 0.5 · (ln l − ln r) / (ln l − 2·ln c + ln r)
    (İlk denemedeki 5x5 ağırlıklı ortalama, pencere Gauss'u kırptığı için tepeye doğru
    çekiyordu: 2.4 px hata. Log-parabol ile ~0.)
    """
    n, h, w = hm.shape
    xy = np.zeros((n, 2), np.float32)
    guven = np.zeros(n, np.float32)

    def kayma(l, c, r):
        l, c, r = np.log(max(l, 1e-6)), np.log(max(c, 1e-6)), np.log(max(r, 1e-6))
        payda = l - 2 * c + r
        return 0.0 if payda >= -1e-9 else float(np.clip(0.5 * (l - r) / payda, -0.5, 0.5))

    for j in range(n):
        idx = int(hm[j].argmax())
        y, x = divmod(idx, w)
        guven[j] = hm[j, y, x]
        c = hm[j, y, x]
        dx = kayma(hm[j, y, x - 1], c, hm[j, y, x + 1]) if 0 < x < w - 1 else 0.0
        dy = kayma(hm[j, y - 1, x], c, hm[j, y + 1, x]) if 0 < y < h - 1 else 0.0
        x, y = x + dx, y + dy
        # Hedef haritada nokta (u·W, v·H) konumuna çizildi (piksel merkezi = tam sayı),
        # geri dönüşüm de aynı kuralla: x / W.  (+0.5 eklemek yarım piksel kaydırırdı →
        # 1/4 ölçekte bu orijinalde ~4-5 piksel hata demek; birim testte yakalandı.)
        xy[j] = [x / w, y / h]
    return xy, guven


class UNetSahaModeli:
    """keypoint_tahmin.SahaModeli ile AYNI arayüz → sistem.py ikisini de kullanabilir."""

    def __init__(self, yol: str, esik: float = 0.5):
        self.cihaz = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SahaUNet(onegitimli=False).to(self.cihaz).eval()
        self.model.load_state_dict(torch.load(yol, map_location=self.cihaz))
        self.esik = esik

    @torch.no_grad()
    def tahmin(self, img: np.ndarray) -> list[dict]:
        h, w = img.shape[:2]
        x = cv2.resize(img, (GIRDI_W, GIRDI_H), interpolation=cv2.INTER_AREA)
        x = ((x[..., ::-1].astype(np.float32) / 255 - ORT) / STD).transpose(2, 0, 1)
        hm = self.model(torch.from_numpy(x.copy())[None].to(self.cihaz))[0].float().cpu().numpy()
        xy, g = haritadan_nokta(hm)
        return [{"id": j, "px": [round(float(xy[j, 0] * w), 1), round(float(xy[j, 1] * h), 1)],
                 "guven": round(float(g[j]), 3)} for j in range(N) if g[j] >= self.esik]


# ---------------------------------------------------------------------------
# EĞİTİM
# ---------------------------------------------------------------------------
@torch.no_grad()
def dogrula(model, yukleyici, cihaz, esik=0.5):
    """Val kaybı + PCK@10px (1280 genişliğe göre) + görünürlük doğruluğu."""
    model.eval()
    toplam, n_batch, dogru, etiketli, gor_dogru, gor_toplam = 0.0, 0, 0, 0, 0, 0
    for x, hm, kp in yukleyici:
        x, hm = x.to(cihaz), hm.to(cihaz)
        with torch.autocast(cihaz, dtype=torch.float16, enabled=cihaz == "cuda"):
            t = model(x)
        toplam += kayip_fonksiyonu(t.float(), hm).item()
        n_batch += 1
        for b in range(len(x)):
            xy, g = haritadan_nokta(t[b].float().cpu().numpy())
            gercek = kp[b].numpy()
            gor = gercek[:, 2] > 0
            gor_dogru += int(((g >= esik) == gor).sum())
            gor_toplam += N
            etiketli += int(gor.sum())
            hata = np.hypot((xy[:, 0] - gercek[:, 0]) * 1280, (xy[:, 1] - gercek[:, 1]) * 720)
            dogru += int(((hata <= 10) & gor & (g >= esik)).sum())
    model.train()
    return toplam / max(n_batch, 1), dogru / max(etiketli, 1), gor_dogru / max(gor_toplam, 1)


def main():
    ap = argparse.ArgumentParser(description="U-Net saha keypoint modeli eğit")
    ap.add_argument("--veri", required=True, help="YOLO-pose formatında veri klasörü")
    ap.add_argument("--ad", required=True)
    ap.add_argument("--epoch", type=int, default=15)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--baslangic", help="önceki model ağırlıkları (ince ayar için)")
    ap.add_argument("--isci", type=int, default=3, help="veri yükleyici işçi sayısı (RAM'e dikkat)")
    args = ap.parse_args()

    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    klasor = Path(args.veri)
    # İşçi sayısı bilerek düşük: her işçi ayrı bir Python süreci ve görüntüleri RAM'e açar.
    # 15 GB RAM'li dizüstünde başka işlerle birlikte 6 işçi belleği doldurup çöktü.
    egit = DataLoader(SahaVeri(klasor, "train", True), batch_size=args.bs, shuffle=True,
                      num_workers=args.isci, pin_memory=True, persistent_workers=True, drop_last=True)
    val = DataLoader(SahaVeri(klasor, "val", False), batch_size=args.bs, shuffle=False, num_workers=1)
    model = SahaUNet().to(cihaz)
    if args.baslangic:
        model.load_state_dict(torch.load(args.baslangic, map_location=cihaz))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Öğrenme oranı: kosinüs eğrisiyle yavaşça 0'a iner (sonlara doğru ince ayar)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epoch * len(egit))
    olcekleyici = torch.amp.GradScaler(enabled=cihaz == "cuda")   # karışık hassasiyet (hız)

    kayit_yolu = PROJE / "runs" / f"unet_{args.ad}"
    kayit_yolu.mkdir(parents=True, exist_ok=True)
    en_iyi = -1.0
    with open(kayit_yolu / "sonuclar.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "egitim_kaybi", "val_kaybi", "val_PCK10", "val_gorunurluk_dogrulugu", "sure_sn"])
        for ep in range(1, args.epoch + 1):
            t0, toplam = time.time(), 0.0
            for i, (x, hm, _) in enumerate(egit):
                x, hm = x.to(cihaz, non_blocking=True), hm.to(cihaz, non_blocking=True)
                with torch.autocast(cihaz, dtype=torch.float16, enabled=cihaz == "cuda"):
                    tahmin = model(x)
                kayip = kayip_fonksiyonu(tahmin.float(), hm)
                opt.zero_grad(set_to_none=True)
                olcekleyici.scale(kayip).backward()        # geri yayılım (backpropagation)
                olcekleyici.step(opt)                      # ağırlıkları güncelle
                olcekleyici.update()
                sched.step()
                toplam += kayip.item()
            vk, pck, gor = dogrula(model, val, cihaz)
            sure = time.time() - t0
            w.writerow([ep, f"{toplam / len(egit):.5f}", f"{vk:.5f}", f"{pck:.4f}", f"{gor:.4f}", f"{sure:.0f}"])
            f.flush()
            print(f"epoch {ep:2d}/{args.epoch} | eğitim kaybı {toplam / len(egit):.5f} | val kaybı {vk:.5f} "
                  f"| PCK@10 {pck:.3f} | görünürlük {gor:.3f} | {sure:.0f} sn", flush=True)
            torch.save(model.state_dict(), kayit_yolu / "son.pt")
            if pck > en_iyi:
                en_iyi = pck
                torch.save(model.state_dict(), PROJE / "models" / f"unet_kp_{args.ad}.pt")
    print(f"En iyi val PCK@10: {en_iyi:.3f} → models\\unet_kp_{args.ad}.pt")


if __name__ == "__main__":
    main()
