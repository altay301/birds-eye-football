"""
sentetik.py — Sentetik yayın görüntüsü + hatasız saha noktası etiketi üretir
===========================================================================

FİKİR
  Elle etiketleme yavaş (~2-3 dk/kare). Bilgisayarda "sanal bir yayın kamerası"
  kurup sahayı fotoğraflarsak, 35 noktanın görüntüdeki yeri MATEMATİKTEN kesin
  olarak bilinir → sınırsız, hatasız etiketli veri. Model saha geometrisini burada
  öğrenir; gerçek yayının nasıl göründüğünü ise az sayıdaki gerçek etiketten öğrenir.

3D KAMERA MODELİ (iğne deliği / pinhole kamera)
  Dünya koordinatı (metre): x = saha uzunluğu (0-105), y = genişlik (0-68, kameraya
  doğru artar), z = yükseklik. Zemin z = 0.
  Bir 3D nokta X'in görüntüdeki yeri:
        p ~ K · R · (X - C)
     C : kameranın yeri          R : kameranın yönü (3x3 dönme)
     K : iç parametreler (odak uzaklığı f = zoom, görüntü merkezi)
  Zemindeki noktalar için z = 0 olduğundan R'nin 3. sütunu düşer:
        H = K · [r1  r2  -R·C]      ← HOMOGRAFİ, 3D kameranın zemine indirgenmiş hali!

ÜRETİM ADIMLARI (her görüntü için)
  1) Rastgele yayın kamerası: yakın taç çizgisinin 15-45 m gerisi, 10-30 m yükseklik,
     sahada rastgele bir noktaya bakıyor, görüş açısı 10-60° (zoom)
  2) Arka plan: tribün benzeri renkli gürültü
  3) Saha dokusu (kuş bakışı çizilir, sonra kameraya göre bükülür): çim rengi,
     biçme şeritleri, ışık farkı, aşınmış beyaz çizgiler
  4) Reklam panoları (3D dikey düzlemler), oyuncular (3D boylu figürler + gölge)
  5) Bozulmalar: bulanıklık, gürültü, parlaklık, JPEG sıkıştırma, skor tabelası
  6) Etiket: 35 noktayı kameradan geçir → görüntü içindekiler "görünür"

ÇIKTI (Ultralytics YOLO-pose formatı)
  data\\synthetic\\images\\{train,val}\\s000001.jpg
  data\\synthetic\\labels\\{train,val}\\s000001.txt
     "0 cx cy w h  x1 y1 v1  x2 y2 v2 ... x35 y35 v35"   (hepsi 0-1 arası)
     Kutu = tüm görüntü (saha tek "nesne"); v=2 görünür, v=0 görünmez.

Kullanım:
    .venv\\Scripts\\python.exe src\\sentetik.py --ornek 12          # birkaç örnek + önizleme
    .venv\\Scripts\\python.exe src\\sentetik.py --train 6000 --val 600
"""
from __future__ import annotations

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

import pitch

PROJE = Path(__file__).resolve().parent.parent
CIKTI = PROJE / "data" / "synthetic"
W, H = 1280, 720
DOKU_OLCEK = 20          # saha dokusu: 1 metre = 20 piksel
DOKU_KENAR = 6.0         # sahanın etrafındaki çim payı (metre)


# ---------------------------------------------------------------------------
# 1) KAMERA
# ---------------------------------------------------------------------------
def rastgele_kamera(rng: np.random.Generator):
    """Yayın kamerasına benzeyen rastgele bir kamera: (K, R, C)."""
    L, Wd = pitch.SAHA_UZUNLUK, pitch.SAHA_GENISLIK
    # Ana kamera çoğunlukla orta sahada; bazen ceza sahası hizasında (yan kameralar)
    cx = rng.normal(L / 2, 12) if rng.random() < 0.7 else rng.uniform(10, L - 10)
    C = np.array([np.clip(cx, -5, L + 5), Wd + rng.uniform(15, 45), rng.uniform(10, 30)])
    # Bakılan nokta: sahada rastgele (kameranın x'inden çok uzaklaşmasın)
    hedef = np.array([np.clip(C[0] + rng.normal(0, 25), 0, L), rng.uniform(4, Wd - 4), 0.0])
    fov = math.radians(rng.uniform(10, 60))                 # yatay görüş açısı
    f = (W / 2) / math.tan(fov / 2)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])

    # Bakış yönünden dönme matrisi ("look-at"):
    #   z_kam = ileri (hedefe doğru), x_kam = görüntüde sağ, y_kam = görüntüde aşağı
    ileri = hedef - C
    ileri /= np.linalg.norm(ileri)
    yukari = np.array([0.0, 0.0, 1.0])
    sag = np.cross(yukari, ileri)          # sahamız "x sağa, y kameraya doğru" olduğundan bu sıra
    sag /= np.linalg.norm(sag)
    asagi = -np.cross(ileri, sag)
    asagi /= np.linalg.norm(asagi)
    R = np.stack([sag, asagi, ileri])     # satırlar: kamera eksenleri
    # Küçük yuvarlanma (kamera hafif yamuk)
    a = math.radians(rng.normal(0, 1.0))
    Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    return K, Rz @ R, C


def izdusur(K, R, C, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """3D noktaları (N,3) görüntüye taşır. Döner: (N,2) piksel, (N,) derinlik."""
    kam = (np.asarray(X, dtype=np.float64) - C) @ R.T
    derinlik = kam[:, 2]
    p = kam @ K.T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = p[:, :2] / p[:, 2:3]
    return uv, derinlik


def zemin_homografisi(K, R, C) -> np.ndarray:
    """H = K [r1 r2 t]  (t = -R C): zemin (metre) → görüntü (piksel)."""
    t = -R @ C
    return K @ np.column_stack([R[:, 0], R[:, 1], t])


# ---------------------------------------------------------------------------
# 2) ÇİZİM PARÇALARI
# ---------------------------------------------------------------------------
def arka_plan(rng) -> np.ndarray:
    """Tribün benzeri: koyu zemin üstünde çok sayıda küçük renkli leke (seyirci)."""
    taban = rng.integers(20, 90, 3)
    img = np.full((H, W, 3), taban, np.uint8)
    kucuk = rng.integers(0, 255, (H // 6, W // 6, 3), dtype=np.uint8)
    kucuk = (kucuk * rng.uniform(0.3, 0.8)).astype(np.uint8)
    img = cv2.addWeighted(img, 0.5, cv2.resize(kucuk, (W, H), interpolation=cv2.INTER_NEAREST), 0.5, 0)
    return cv2.GaussianBlur(img, (3, 3), 0)


def saha_dokusu(rng) -> tuple[np.ndarray, np.ndarray]:
    """Kuş bakışı saha dokusu + doku pikseli → metre dönüşümü (3x3)."""
    L, Wd, k, s = pitch.SAHA_UZUNLUK, pitch.SAHA_GENISLIK, DOKU_KENAR, DOKU_OLCEK
    dw, dh = int((L + 2 * k) * s), int((Wd + 2 * k) * s)
    # Çim rengi (HSV): ton 30-60 (sarımsıdan maviye yakın yeşile), farklı doygunluk/parlaklık
    # (Parlaklık alt sınırı 95: gece maçları bile projektörle aydınlatılır, yayında karanlık olmaz)
    hsv = np.array([rng.uniform(30, 60), rng.uniform(80, 200), rng.uniform(95, 180)])
    cim = cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)
    doku = np.empty((dh, dw, 3), np.float32)
    doku[:] = cim
    # Biçme şeritleri: x boyunca (bazen damalı)
    serit_m = rng.uniform(4.5, 7.0)
    fark = rng.uniform(0.04, 0.14)
    xs = (np.arange(dw) / s - k) // serit_m
    maske = (xs % 2 == 0).astype(np.float32)[None, :]
    if rng.random() < 0.2:
        ys = (np.arange(dh) / s - k) // serit_m
        maske = np.abs(maske - (ys % 2 == 0).astype(np.float32)[:, None])
    doku *= (1 + fark * (maske[..., None] * 2 - 1))
    # Işık farkı (bir taraf gölgede) + ince gürültü
    gx = np.linspace(rng.uniform(0.75, 1.0), rng.uniform(0.9, 1.15), dw, dtype=np.float32)
    doku *= gx[None, :, None]
    doku += rng.normal(0, 6, (dh // 4, dw // 4, 1)).astype(np.float32).repeat(4, 0).repeat(4, 1)[:dh, :dw]
    doku = np.clip(doku, 0, 255).astype(np.uint8)

    # Çizgiler: pitch.saha_cizgileri() metre cinsinden; dokuya çiz
    kalinlik = max(2, int(round(rng.uniform(0.10, 0.16) * s)))
    beyaz = int(rng.uniform(200, 255))
    for c in pitch.saha_cizgileri(adim=0.25):
        p = np.round((c + k) * s).astype(np.int32)
        cv2.polylines(doku, [p], False, (beyaz,) * 3, kalinlik, cv2.LINE_AA)
    for xm in (pitch.PENALTI_MESAFE, L - pitch.PENALTI_MESAFE, L / 2):
        cv2.circle(doku, (int((xm + k) * s), int((Wd / 2 + k) * s)), kalinlik, (beyaz,) * 3, -1)
    # Aşınma: çizgileri yer yer soluklaştır
    if rng.random() < 0.6:
        asinma = cv2.resize(rng.uniform(0.6, 1.0, (dh // 40, dw // 40)).astype(np.float32), (dw, dh))
        cim_img = np.empty_like(doku)
        cim_img[:] = cim.astype(np.uint8)
        doku = (doku * asinma[..., None] + cim_img * (1 - asinma[..., None])).astype(np.uint8)
    # doku pikseli → metre:  m = px / s - k
    T = np.array([[1 / s, 0, -k], [0, 1 / s, -k], [0, 0, 1]])
    return doku, T


def panolar(img, K, R, C, rng):
    """Uzak taç çizgisi ve kale arkalarında 3D dikey reklam panoları."""
    L, Wd = pitch.SAHA_UZUNLUK, pitch.SAHA_GENISLIK
    yuk = rng.uniform(0.8, 1.2)
    uzak = rng.uniform(2.5, 5.0)
    hatlar = [((-uzak, -uzak), (L + uzak, -uzak)),            # uzak taç çizgisi arkası
              ((-uzak, -uzak), (-uzak, Wd + uzak)),           # sol kale arkası
              ((L + uzak, -uzak), (L + uzak, Wd + uzak))]     # sağ kale arkası
    for (x1, y1), (x2, y2) in hatlar:
        n = int(np.hypot(x2 - x1, y2 - y1) // rng.uniform(6, 12)) + 1
        for j in range(n):
            a, b = j / n, (j + 1) / n
            p1 = (x1 + (x2 - x1) * a, y1 + (y2 - y1) * a)
            p2 = (x1 + (x2 - x1) * b, y1 + (y2 - y1) * b)
            X = np.array([[*p1, 0], [*p2, 0], [*p2, yuk], [*p1, yuk]])
            uv, d = izdusur(K, R, C, X)
            if (d <= 0.5).any() or np.abs(uv).max() > 1e5:
                continue
            renk = tuple(int(v) for v in rng.integers(0, 255, 3))
            cv2.fillPoly(img, [np.round(uv).astype(np.int32)], renk, cv2.LINE_AA)
            if rng.random() < 0.5:   # pano üstünde "yazı" şeridi
                X2 = np.array([[*p1, yuk * 0.35], [*p2, yuk * 0.35], [*p2, yuk * 0.65], [*p1, yuk * 0.65]])
                uv2, _ = izdusur(K, R, C, X2)
                cv2.fillPoly(img, [np.round(uv2).astype(np.int32)],
                             tuple(255 - c for c in renk), cv2.LINE_AA)


def oyuncular(img, K, R, C, rng):
    """Basit 3D oyuncu figürleri: gölge + bacak + forma + kafa (uzaktakiler önce çizilir)."""
    L, Wd = pitch.SAHA_UZUNLUK, pitch.SAHA_GENISLIK
    renkler = [tuple(int(v) for v in rng.integers(0, 255, 3)) for _ in range(2)]
    hakem = (20, 20, 20) if rng.random() < 0.5 else tuple(int(v) for v in rng.integers(0, 255, 3))
    # Oyuncular bakılan bölgede kümelenir (gerçek maçta da top etrafında)
    merkez = np.array([np.clip(C[0] + rng.normal(0, 20), 5, L - 5), rng.uniform(10, Wd - 10)])
    n = int(rng.integers(6, 23))
    poz = merkez + rng.normal(0, [18, 12], (n, 2))
    poz = np.clip(poz, [-1, -1], [L + 1, Wd + 1])
    sira = np.argsort(-np.linalg.norm(poz - C[:2], axis=1))   # uzaktan yakına
    ten = [(80, 110, 160), (60, 80, 120), (40, 55, 80), (120, 150, 200)]
    for idx in sira:
        x, y = poz[idx]
        boy = rng.uniform(1.7, 1.95)
        uv, d = izdusur(K, R, C, np.array([[x, y, 0], [x, y, boy]]))
        if (d <= 1).any():
            continue
        (ax, ay), (bx, by) = uv
        h_px = ay - by
        if not (2 < h_px < 600) or not (-50 < ax < W + 50 and -50 < ay < H + 50):
            continue
        g = max(1, int(h_px * 0.28))   # genişlik
        renk = hakem if idx == 0 else renkler[idx % 2]
        ayak = (int(ax), int(ay))
        cv2.ellipse(img, ayak, (int(g * 1.3), max(1, int(g * 0.35))), 0, 0, 360, (30, 50, 30), -1)  # gölge
        cv2.rectangle(img, (int(ax - g * 0.45), int(ay - h_px * 0.48)), (int(ax + g * 0.45), int(ay)),
                      tuple(int(c * 0.4) for c in renk), -1)                         # bacak/şort
        cv2.rectangle(img, (int(ax - g * 0.55), int(ay - h_px * 0.85)), (int(ax + g * 0.55), int(ay - h_px * 0.45)),
                      renk, -1)                                                      # forma
        cv2.circle(img, (int(ax), int(ay - h_px * 0.92)), max(1, int(g * 0.4)), ten[idx % 4], -1)  # kafa


def bozulmalar(img, rng) -> np.ndarray:
    """Gerçek yayın kusurlarını taklit et: bulanıklık, renk, gürültü, tabela, JPEG."""
    if rng.random() < 0.4:   # hareket bulanıklığı (yatay pan)
        n = int(rng.integers(3, 12))
        ker = np.zeros((n, n), np.float32)
        ker[n // 2, :] = 1 / n
        img = cv2.filter2D(img, -1, ker)
    elif rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.5))
    img = cv2.convertScaleAbs(img, alpha=rng.uniform(0.75, 1.25), beta=rng.uniform(-25, 25))
    if rng.random() < 0.5:
        img = np.clip(img.astype(np.int16) + rng.normal(0, rng.uniform(2, 8), img.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.6:   # skor tabelası / kanal logosu
        x0, y0 = int(rng.uniform(20, 300)), int(rng.uniform(15, 60))
        cv2.rectangle(img, (x0, y0), (x0 + int(rng.uniform(150, 320)), y0 + int(rng.uniform(25, 45))),
                      tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(45, 95))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# 3) TEK ÖRNEK
# ---------------------------------------------------------------------------
def ornek_uret(tohum: int):
    """Döner: (görüntü, noktalar (35,2) piksel, görünür (35,) bool) ya da None (kötü kamera)."""
    rng = np.random.default_rng(tohum)
    for _ in range(20):   # sahanın yeterince göründüğü bir kamera bulana kadar dene
        K, R, C = rastgele_kamera(rng)
        Hz = zemin_homografisi(K, R, C)
        X = np.column_stack([pitch.NOKTA_KOORD, np.zeros(pitch.NOKTA_SAYISI)])
        uv, d = izdusur(K, R, C, X)
        gorunur = (d > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        # Görüntünün alt yarısı sahada mı? (ufuk/tribün çok fazla olmasın)
        alt = np.array([[W * 0.1, H * 0.95], [W * 0.9, H * 0.95], [W * 0.5, H * 0.6]])
        zemin = cv2.perspectiveTransform(alt.reshape(-1, 1, 2), np.linalg.inv(Hz)).reshape(-1, 2)
        sahada = ((zemin[:, 0] > -3) & (zemin[:, 0] < 108) & (zemin[:, 1] > -3) & (zemin[:, 1] < 71)).all()
        if gorunur.sum() >= 2 and sahada:
            break
    else:
        return None

    img = arka_plan(rng)
    doku, T = saha_dokusu(rng)
    # doku → görüntü:  piksel = Hz · T · doku_pikseli
    M = Hz @ T
    sahamaske = cv2.warpPerspective(np.full(doku.shape[:2], 255, np.uint8), M, (W, H))
    saha_img = cv2.warpPerspective(doku, M, (W, H), flags=cv2.INTER_LINEAR)
    # Ufkun üstündeki pikseller zemine "kameranın arkasından" eşlenir (homojen w'nin işareti
    # ters döner) ve warpPerspective oralara saha dokusunun AYNA görüntüsünü koyar.
    # Referans: görüntünün alt ortası kesinlikle zeminde → onun w işaretiyle aynı olmayanları at.
    ys, xs = np.mgrid[0:H, 0:W]
    ters = np.linalg.inv(Hz)
    w_ = ters[2, 0] * xs + ters[2, 1] * ys + ters[2, 2]
    referans = np.sign(ters[2, 0] * (W / 2) + ters[2, 1] * (H - 1) + ters[2, 2])
    sahamaske[np.sign(w_) != referans] = 0
    img[sahamaske > 0] = saha_img[sahamaske > 0]
    panolar(img, K, R, C, rng)
    oyuncular(img, K, R, C, rng)
    img = bozulmalar(img, rng)
    return img, uv, gorunur


def etiket_satiri(uv, gorunur) -> str:
    parcalar = ["0", "0.5", "0.5", "1", "1"]   # sınıf 0, kutu = tüm görüntü
    for (u, v), g in zip(uv, gorunur):
        if g:
            parcalar += [f"{u / W:.6f}", f"{v / H:.6f}", "2"]
        else:
            parcalar += ["0", "0", "0"]
    return " ".join(parcalar)


def _isci(is_):
    kume, no = is_
    sonuc = ornek_uret(no * 7919 + (0 if kume == "train" else 10_000_000))
    if sonuc is None:
        return 0
    img, uv, gor = sonuc
    ad = f"s{no:06d}"
    cv2.imwrite(str(CIKTI / "images" / kume / f"{ad}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    (CIKTI / "labels" / kume / f"{ad}.txt").write_text(etiket_satiri(uv, gor) + "\n")
    return 1


def veri_seti_yaml():
    """Ultralytics'in okuyacağı veri seti tanımı."""
    yol = CIKTI / "saha_sentetik.yaml"
    yol.write_text(
        f"path: {CIKTI.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"kpt_shape: [{pitch.NOKTA_SAYISI}, 3]\n"
        f"flip_idx: {pitch.FLIP_IDX}\n"
        "names:\n  0: saha\n", encoding="utf-8")
    return yol


def onizleme(adet: int, cikti: Path):
    """İlk birkaç örneği etiket noktalarıyla birlikte bir panoda göster."""
    kucuk = []
    for t in range(adet):
        s = ornek_uret(1000 + t)
        if s is None:
            continue
        img, uv, gor = s
        for i in np.where(gor)[0]:
            p = tuple(np.round(uv[i]).astype(int))
            cv2.circle(img, p, 6, (0, 0, 255), -1)
            cv2.putText(img, str(i), (p[0] + 7, p[1] - 5), 0, 0.7, (0, 255, 255), 2)
        kucuk.append(cv2.resize(img, (480, 270)))
    while len(kucuk) % 3:
        kucuk.append(np.zeros_like(kucuk[0]))
    pano = np.vstack([np.hstack(kucuk[i:i + 3]) for i in range(0, len(kucuk), 3)])
    cikti.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(cikti), pano)
    print("Önizleme:", cikti)


def main():
    ap = argparse.ArgumentParser(description="Sentetik saha görüntüsü üret")
    ap.add_argument("--ornek", type=int, default=0, help="sadece N örneklik önizleme panosu üret")
    ap.add_argument("--train", type=int, default=6000)
    ap.add_argument("--val", type=int, default=600)
    ap.add_argument("--isci", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    if args.ornek:
        onizleme(args.ornek, PROJE / "results" / "sentetik_onizleme.jpg")
        return
    for kume in ("train", "val"):
        (CIKTI / "images" / kume).mkdir(parents=True, exist_ok=True)
        (CIKTI / "labels" / kume).mkdir(parents=True, exist_ok=True)
    isler = [("train", i) for i in range(args.train)] + [("val", i) for i in range(args.val)]
    with ProcessPoolExecutor(args.isci) as ex:
        uretilen = 0
        for j, r in enumerate(ex.map(_isci, isler, chunksize=16)):
            uretilen += r
            if (j + 1) % 500 == 0:
                print(f"  {j + 1}/{len(isler)}", flush=True)
    print(f"Üretilen: {uretilen} / {len(isler)}  →  {CIKTI}")
    print("Veri seti tanımı:", veri_seti_yaml())


if __name__ == "__main__":
    main()
