"""
kare_cikar.py — Maç videolarından etiketlenecek kareleri otomatik seçer
=======================================================================

Bir 10 dakikalık özet ≈ 15.000 kare. Çoğu işe yaramaz (ardışık kareler aynı,
tribün çekimleri, bulanık tekrarlar). Bu script İKİ AŞAMADA çalışır:

  1. AŞAMA — ADAY TOPLAMA (videoyu sık tarar, her 1 sn'de 1 kare)
     a) SAHA FİLTRESİ   : görüntünün alt 2/3'ünde yeşil oranı düşükse → saha yok
     b) BULANIKLIK      : Laplace varyansı düşükse → bulanık
     c) TEKRAR FİLTRESİ : önceki bir adaya çok benzerse → tekrar
     Filtreden geçenler "aday" olur. Bellekte sadece ZAMANLARI tutulur, görüntüleri değil
     (15 dk HD videonun karelerini bellekte tutmak ~5 GB olurdu).

  2. AŞAMA — DENGELİ SEÇİM
     Aday sayısı maç başına sınırı (varsayılan 25) aşıyorsa, adaylar arasından
     ZAMANA EŞİT YAYILMIŞ olanlar seçilir. Sonra video bir kez daha hızlıca
     taranıp sadece seçilen kareler diske yazılır.

  Neden? Videolar 2–15 dk arası değişiyor. Sabit aralıkla kısa video çok az,
  uzun video çok fazla kare verirdi. Bu yöntemle:
     * kısa video → elindeki tüm iyi kareleri verir
     * uzun video → tüm süresine yayılmış dengeli bir örnek verir
     * hiçbir maç veri setine hâkim olmaz

Kullanım (proje klasöründe):
    .venv\\Scripts\\python.exe src\\kare_cikar.py                    # tüm videolar
    .venv\\Scripts\\python.exe src\\kare_cikar.py --max-kare 30      # maç başına 30
    .venv\\Scripts\\python.exe src\\kare_cikar.py --video mac01.mp4  # tek video
    .venv\\Scripts\\python.exe src\\kare_cikar.py --yeniden          # daha önce işlenenleri de yeniden yap

GÜVENLİK: Kareleri daha önce çıkarılmış bir video ATLANIR (etiketlediğin kareler
yanlışlıkla silinmesin diye). Yeniden yapmak için --yeniden kullan.

Çıktılar:
    data\\frames\\<video_adı>\\<video_adı>_t0123.0.jpg    seçilen kareler
    data\\frames\\kare_kaydi.csv                          incelenen her kare için karar + sebep
    data\\frames\\video_ozeti.csv                         video başına özet
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJE = Path(__file__).resolve().parent.parent
VIDEO_KLASOR = PROJE / "data" / "raw_videos"
KARE_KLASOR = PROJE / "data" / "frames"
VIDEO_UZANTI = {".mp4", ".mkv", ".webm", ".avi", ".mov"}
# İndirme sürerken oluşan yarım dosyalar: ad.f616.mp4, ad.f251.webm, ad.temp.mp4, ad.mp4.part
YARIM_DOSYA = re.compile(r"\.(f\d+|temp|part)\.|\.part$", re.IGNORECASE)


def videolari_listele() -> list[Path]:
    videolar, yarim = [], []
    for p in sorted(VIDEO_KLASOR.iterdir()):
        if YARIM_DOSYA.search(p.name):
            yarim.append(p.name)
        elif p.suffix.lower() in VIDEO_UZANTI:
            videolar.append(p)
    if yarim:
        print(f"Yarım (inmekte olan) dosyalar atlandı: {', '.join(yarim)}")
    return videolar


# ---------------------------------------------------------------------------
# FİLTRE a: Saha var mı? — yeşil piksel oranı
# ---------------------------------------------------------------------------
def yesil_orani(kare: np.ndarray) -> float:
    """
    Görüntünün alt 2/3'ündeki yeşil piksel oranı (0-1).
    Üst kısmı saymıyoruz çünkü geniş açıda orası tribün olur — sahayı alt kısım gösterir.
    HSV renk uzayı: H = renk tonu (OpenCV'de 0-180; yeşil ≈ 35-85),
                    S = doygunluk (gri/beyaz düşük), V = parlaklık.
    Aralık 25-90 tutuldu: yaz maçlarında / bazı yayınlarda çim SARIMSI yeşil görünür
    (ör. 11 numaralı video: H ≈ 32-35). 35'ten başlatınca bu maçın %92'si "saha yok" sayıldı.
    """
    alt = kare[kare.shape[0] // 3:]
    hsv = cv2.cvtColor(alt, cv2.COLOR_BGR2HSV)
    yesil = (hsv[..., 0] > 25) & (hsv[..., 0] < 90) & (hsv[..., 1] > 50) & (hsv[..., 2] > 40)
    return float(yesil.mean())


# ---------------------------------------------------------------------------
# FİLTRE b: Bulanık mı? — Laplace varyansı
# ---------------------------------------------------------------------------
def keskinlik(kare: np.ndarray) -> float:
    """
    Laplace filtresi görüntüdeki ani parlaklık değişimlerini (kenarları) bulur.
    Keskin görüntüde çok kenar vardır → sonuçların varyansı YÜKSEK.
    Bulanık görüntüde kenarlar yumuşar → varyans DÜŞÜK.
    Karşılaştırılabilir olsun diye önce sabit boyuta (640 genişlik) küçültüyoruz.
    """
    gri = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)
    olcek = 640 / gri.shape[1]
    gri = cv2.resize(gri, None, fx=olcek, fy=olcek, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gri, cv2.CV_64F).var())


# ---------------------------------------------------------------------------
# FİLTRE c: Tekrar mı? — algısal hash (dHash)
# ---------------------------------------------------------------------------
def dhash(kare: np.ndarray, boyut: int = 16) -> np.ndarray:
    """
    Algısal hash: görüntünün "parmak izi". Benzer görüntülerin parmak izleri de benzer.
      1) Gri yap, (boyut+1) x boyut piksele küçült  → ayrıntılar gider, genel yapı kalır
      2) Her pikseli sağındakiyle karşılaştır: sağdaki daha parlaksa 1, değilse 0
      3) Sonuç: boyut*boyut bitlik (256 bit) bir dizi
    İki karenin kaç bitinin farklı olduğu (Hamming uzaklığı) ne kadar farklı olduklarını söyler.
    """
    gri = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)
    kucuk = cv2.resize(gri, (boyut + 1, boyut), interpolation=cv2.INTER_AREA)
    return (kucuk[:, 1:] > kucuk[:, :-1]).flatten()


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


# ---------------------------------------------------------------------------
# FİLTRE d: Geniş açı mı? — oyuncu sayısı ve boyutu (YOLO)
# ---------------------------------------------------------------------------
_yolo = None


def genis_aci_olcumu(kare: np.ndarray) -> tuple[int, float]:
    """
    Yeşil oranı sadece "çim var mı?" sorusunu cevaplar; yakın planlarda da arka plan
    çim olduğu için onları eleyemez (ör. yeşil kaleci forması, sadece ayaklar).
    Asıl soru: "Bu bir GENİŞ AÇI çekim mi?" Cevap oyuncuların boyutunda:
        geniş açı → çok sayıda KÜÇÜK oyuncu
        yakın plan → az sayıda BÜYÜK oyuncu (ya da hiç oyuncu yok)
    Döner: (kişi sayısı, kişi boylarının medyanı / görüntü yüksekliği)
    imgsz=1280: uzaktaki küçük oyuncuları kaçırmamak için yüksek çözünürlük.
    """
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO
        _yolo = YOLO(str(PROJE / "models" / "yolov8n.pt"))
    s = _yolo.predict(kare, classes=[0], conf=0.25, imgsz=1280, verbose=False)[0]
    if len(s.boxes) == 0:
        return 0, 0.0
    boylar = (s.boxes.xyxy[:, 3] - s.boxes.xyxy[:, 1]).cpu().numpy()
    return len(boylar), float(np.median(boylar) / kare.shape[0])


# ---------------------------------------------------------------------------
# AŞAMA 2'nin kalbi: zamana eşit yayılmış seçim
# ---------------------------------------------------------------------------
def esit_yayilmis_sec(n_aday: int, sinir: int) -> list[int]:
    """
    n_aday adaydan (zamana göre sıralı) 'sinir' kadarını eşit aralıklarla seçer.
    Örnek: 10 adaydan 4 → np.linspace(0, 9, 4) = [0, 3, 6, 9]
    İlk ve son aday her zaman dahil → videonun başı da sonu da temsil edilir.
    """
    if sinir <= 0 or n_aday <= sinir:
        return list(range(n_aday))
    return sorted(set(np.round(np.linspace(0, n_aday - 1, sinir)).astype(int).tolist()))


@dataclass
class Aday:
    kare_no: int
    zaman: float


# ---------------------------------------------------------------------------
# Ana işlem
# ---------------------------------------------------------------------------
def video_isle(video: Path, args, kayit: csv.writer) -> dict:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"  ! Açılamadı: {video.name}")
        return {}
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    toplam = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    adim = max(1, int(round(fps * args.aralik)))    # kaç karede bir inceleyeceğiz
    sure_dk = toplam / fps / 60
    print(f"  {video.name}: {sure_dk:.1f} dk, {fps:.0f} fps")

    # ---------------- 1. AŞAMA: aday toplama ----------------
    adaylar: list[Aday] = []
    aday_hashler: list[np.ndarray] = []
    sayac = {"saha_yok": 0, "bulanik": 0, "tekrar": 0, "yakin_plan": 0}
    satirlar = []           # CSV satırları; "aday" kararı 2. aşamada netleşecek
    i = 0
    while cap.grab():       # grab(): kareyi çözmeden atla (hızlı)
        if i % adim == 0:
            ok, kare = cap.retrieve()   # retrieve(): sadece incelenecek kareyi çöz (yavaş)
            if not ok:
                break
            t = i / fps
            yesil = yesil_orani(kare)
            keskin = keskinlik(kare)
            h = dhash(kare)
            en_yakin = min((hamming(h, s) for s in aday_hashler), default=999)
            kisi, boy = "", ""
            # Ucuz filtreler önce; pahalı YOLO sadece onlardan geçenlere uygulanır
            if yesil < args.min_yesil:
                karar = "saha_yok"
            elif keskin < args.min_keskinlik:
                karar = "bulanik"
            elif en_yakin < args.min_fark:
                karar = "tekrar"
            else:
                kisi, boy = genis_aci_olcumu(kare)
                if kisi < args.min_kisi or boy > args.max_boy:
                    karar = "yakin_plan"
                else:
                    karar = "aday"
                    adaylar.append(Aday(i, t))
                    aday_hashler.append(h)
                boy = f"{boy:.3f}"
            if karar != "aday":
                sayac[karar] += 1
            satirlar.append([video.name, f"{t:.1f}", f"{yesil:.3f}", f"{keskin:.1f}",
                             en_yakin if en_yakin != 999 else "", kisi, boy, karar, i])
        i += 1
    incelenen = len(satirlar)

    # ---------------- 2. AŞAMA: dengeli seçim + kaydetme ----------------
    secilen_idx = esit_yayilmis_sec(len(adaylar), args.max_kare)
    secilen_kare_no = {adaylar[j].kare_no for j in secilen_idx}

    hedef = KARE_KLASOR / video.stem
    hedef.mkdir(parents=True, exist_ok=True)
    kaydedilen = set()
    if secilen_kare_no:
        # Videonun başına dönmek için kapatıp yeniden açıyoruz. cap.set(...) ile "geri sarma"
        # bazı video formatlarında tam doğru kareye gitmiyor; yeniden açmak her zaman güvenli.
        cap.release()
        cap = cv2.VideoCapture(str(video))
        son = max(secilen_kare_no)
        i = 0
        while i <= son and cap.grab():
            if i in secilen_kare_no:
                ok, kare = cap.retrieve()
                if ok:
                    # Tutarlılık: tüm kareler en fazla 1920 px genişlikte (ör. 4K video → Full HD)
                    if kare.shape[1] > args.max_genislik:
                        o = args.max_genislik / kare.shape[1]
                        kare = cv2.resize(kare, None, fx=o, fy=o, interpolation=cv2.INTER_AREA)
                    ad = f"{video.stem}_t{i / fps:07.1f}.jpg"
                    # imencode + tofile: Türkçe karakterli yollarda da çalışır
                    cv2.imencode(".jpg", kare, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(str(hedef / ad))
                    kaydedilen.add(i)
            i += 1
    cap.release()

    # DOĞRULAMA: seçilen her kare gerçekten diske yazıldı mı?
    # (Bir gece 4K video ikinci okumada bellek yetmediği için 135. saniyede kesildi;
    #  script fark etmeden "25 seçildi" yazmıştı — gerçekte 13 kare kaydedilmişti.)
    eksik = secilen_kare_no - kaydedilen
    if eksik:
        print(f"    !!! UYARI: {len(eksik)} seçili kare KAYDEDİLEMEDİ (video okuma yarıda kesildi). "
              f"Yeniden dene: --video {video.name} --yeniden", flush=True)

    for s in satirlar:
        kare_no = s.pop()
        if s[-1] == "aday":
            s[-1] = ("secildi" if kare_no in kaydedilen else
                     "kaydedilemedi" if kare_no in secilen_kare_no else "sinir_disi")
        kayit.writerow(s)

    ozet = {"video": video.name, "sure_dk": round(sure_dk, 1), "incelenen": incelenen,
            **sayac, "aday": len(adaylar), "secilen": len(kaydedilen)}
    print(f"    incelenen {incelenen} | saha yok {sayac['saha_yok']} | bulanık {sayac['bulanik']} "
          f"| tekrar {sayac['tekrar']} | yakın plan {sayac['yakin_plan']} "
          f"| aday {len(adaylar)} → SEÇİLEN {len(kaydedilen)}", flush=True)
    return ozet


def main():
    ap = argparse.ArgumentParser(description="Videolardan etiketlenecek kareleri seç")
    ap.add_argument("--video", help="sadece bu videoyu işle (data/raw_videos içindeki adı)")
    ap.add_argument("--aralik", type=float, default=1.0, help="kaç saniyede bir kare incelensin (varsayılan 1)")
    ap.add_argument("--max-kare", type=int, default=25, help="maç başına en fazla kare (0 = sınırsız)")
    ap.add_argument("--min-yesil", type=float, default=0.40, help="alt 2/3'te en az yeşil oranı")
    ap.add_argument("--min-keskinlik", type=float, default=60.0, help="Laplace varyansı alt sınırı")
    ap.add_argument("--min-fark", type=int, default=40,
                    help="adaylardan en az kaç bit farklı olmalı (256 bitten)")
    ap.add_argument("--min-kisi", type=int, default=4, help="geniş açı için en az kişi sayısı")
    ap.add_argument("--max-boy", type=float, default=0.20,
                    help="geniş açı için kişi boylarının medyanı / ekran yüksekliği üst sınırı")
    ap.add_argument("--max-genislik", type=int, default=1920,
                    help="kaydedilen karelerin en fazla genişliği (4K → Full HD)")
    ap.add_argument("--yeniden", action="store_true",
                    help="kareleri daha önce çıkarılmış videoları da sil ve yeniden işle")
    args = ap.parse_args()

    if args.video:
        videolar = [VIDEO_KLASOR / args.video]
    else:
        videolar = videolari_listele()
    if not videolar:
        print(f"Video bulunamadı. Videoları şu klasöre koy: {VIDEO_KLASOR}")
        return

    KARE_KLASOR.mkdir(parents=True, exist_ok=True)
    kayit_yolu = KARE_KLASOR / "kare_kaydi.csv"
    ozet_yolu = KARE_KLASOR / "video_ozeti.csv"
    ozetler = []
    yeni = not kayit_yolu.exists()
    with open(kayit_yolu, "a", newline="", encoding="utf-8") as f:
        kayit = csv.writer(f)
        if yeni:
            kayit.writerow(["video", "zaman_sn", "yesil_orani", "keskinlik", "en_yakin_fark",
                            "kisi_sayisi", "medyan_boy", "karar"])
        for v in videolar:
            hedef = KARE_KLASOR / v.stem
            if hedef.exists() and any(hedef.iterdir()):
                if not args.yeniden:
                    print(f"  {v.name}: daha önce işlenmiş, atlandı (yeniden için --yeniden)")
                    continue
                shutil.rmtree(hedef)
            ozet = video_isle(v, args, kayit)
            if ozet:
                ozetler.append(ozet)

    if ozetler:
        yeni_ozet = not ozet_yolu.exists()
        with open(ozet_yolu, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(ozetler[0].keys()))
            if yeni_ozet:
                w.writeheader()
            w.writerows(ozetler)
        toplam = sum(o["secilen"] for o in ozetler)
        print(f"\nBu çalıştırmada {len(ozetler)} video işlendi, toplam {toplam} kare seçildi → {KARE_KLASOR}")
        az = [o["video"] for o in ozetler if o["secilen"] < args.max_kare // 2]
        if az:
            print(f"Dikkat — sınırın yarısından az kare veren videolar: {', '.join(az)}")
        print(f"Kayıtlar: {kayit_yolu.name}, {ozet_yolu.name}")


if __name__ == "__main__":
    main()
