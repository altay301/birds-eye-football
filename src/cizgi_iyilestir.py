"""
cizgi_iyilestir.py — Homografiyi görüntüdeki SAHA ÇİZGİLERİNE oturtarak iyileştirir
==================================================================================

Saha noktası modeli homografiyi birkaç noktadan kurar. Noktalar görüntünün bir bölgesinde
toplanmışsa dönüşüm o bölgede doğru, uzaklaştıkça bozuk olur. Çizgiler ise görüntünün her yerinde
bilgi taşır. Fikir:

  1) ÇİZGİ HARİTASI: Lab parlaklığına "top-hat" filtresi (zemine göre ince ve parlak yapılar) +
     çim bölgesi maskesi → beyaz çizgi pikselleri. Ardından MESAFE DÖNÜŞÜMÜ: her pikselin en yakın
     çizgi pikseline uzaklığı.
  2) ŞABLON: sahadaki tüm çizgiler (metre) 0.25 m aralıkla örneklenir.
  3) OPTİMİZASYON: H_ters (saha → görüntü) 8 parametresi, şablon noktalarının görüntüdeki
     yansımalarının çizgilere uzaklığı (kırpılmış, τ) en küçük olacak şekilde ayarlanır
     (scipy least_squares, dayanıklı "soft_l1" kaybı: oyuncu gibi çizgi olmayan parlak şeylere
     karşı). Modelin RANSAC'ın kabul ettiği noktaları da hafif bir "çapa" terimi olarak eklenir,
     böylece çözüm başka bir çizgiye kayamaz.
  4) KABUL: ortalama çizgi uzaklığı belirgin azaldıysa ve noktalar fazla kaymadıysa yeni H; yoksa eski.

Kullanım:  from cizgi_iyilestir import iyilestir   →   kal = iyilestir(img, kal, noktalar)
Değerlendirme: .venv\\Scripts\\python.exe -X utf8 src\\cizgi_iyilestir.py --kume val|test
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

import pitch
from homografi import HomografiSonuc, isareti_duzelt, noktalari_donustur

PROJE = Path(__file__).resolve().parent.parent
SABLON = np.vstack(pitch.saha_cizgileri(adim=0.25)).astype(np.float64)       # (N, 2) metre


def cizgi_haritasi(img: np.ndarray, olcek: float = 0.5, esik: float = 14.0):
    """Döner: (mesafe dönüşümü [küçük ölçek], çizgi maskesi, saha maskesi)."""
    k = cv2.resize(img, None, fx=olcek, fy=olcek, interpolation=cv2.INTER_AREA)
    L = cv2.cvtColor(k, cv2.COLOR_BGR2LAB)[..., 0]
    boy = max(5, int(round(22 * olcek)) | 1)
    tophat = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (boy, boy)))
    hsv = cv2.cvtColor(k, cv2.COLOR_BGR2HSV)
    cim = ((hsv[..., 0] > 25) & (hsv[..., 0] < 95) & (hsv[..., 1] > 35)).astype(np.uint8)
    buyuk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(41 * olcek) | 1, int(41 * olcek) | 1))
    saha = cv2.morphologyEx(cim, cv2.MORPH_CLOSE, buyuk)                    # çizgileri de içine alan çim bölgesi
    cizgi = ((tophat > esik) & (saha > 0)).astype(np.uint8)
    dt = cv2.distanceTransform((1 - cizgi).astype(np.uint8), cv2.DIST_L2, 3)
    return dt, cizgi, saha


def _yansit(G: np.ndarray, p: np.ndarray):
    q = np.c_[p, np.ones(len(p))] @ G.T
    w = q[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = q[:, :2] / w[:, None]
    return uv, w


def iyilestir(img: np.ndarray, kal: HomografiSonuc, noktalar: list[dict] | None = None,
              olcek: float = 0.5, tau: float = 8.0, capa: float = 0.5, en_az_iyilesme: float = 0.10):
    """
    kal: modelin noktalarından RANSAC ile kurulan homografi. Döner: (yeni kal, bilgi sözlüğü).
    İyileşme yetersizse ya da çözüm noktalardan fazla uzaklaşırsa eski kal döner.
    """
    dt, _, saha = cizgi_haritasi(img, olcek)
    h, w = dt.shape
    S = np.diag([olcek, olcek, 1.0])
    G0 = S @ kal.H_ters
    G0 = G0 / G0[2, 2]
    uv0, w0 = _yansit(G0, SABLON)
    ic = (w0 > 0) & np.isfinite(uv0).all(1)
    ic &= (uv0[:, 0] >= 2) & (uv0[:, 0] < w - 2) & (uv0[:, 1] >= 2) & (uv0[:, 1] < h - 2)
    if ic.sum() < 50:
        return kal, {"durum": "az görünen çizgi"}
    ui, vi = uv0[ic].astype(int).T
    ic_idx = np.where(ic)[0][saha[vi, ui] > 0]                              # sadece çim bölgesindekiler
    if len(ic_idx) < 50:
        return kal, {"durum": "az görünen çizgi"}
    sablon = SABLON[ic_idx]

    # Çapa: RANSAC'ın kabul ettiği model noktaları (görüntü ↔ saha), çözüm başka çizgiye kaymasın
    capa_g = capa_s = None
    if noktalar and kal.icerdeki is not None and len(noktalar) == len(kal.icerdeki):
        sec = [n for n, k in zip(noktalar, kal.icerdeki) if k]
        if sec:
            capa_g = np.array([n["px"] for n in sec], float) * olcek
            capa_s = pitch.NOKTA_KOORD[[n["id"] for n in sec]].astype(float)

    def artik(x):
        G = np.append(x, 1.0).reshape(3, 3)
        uv, ww = _yansit(G, sablon)
        gecersiz = ~((ww > 0) & np.isfinite(uv).all(1))
        uv = np.nan_to_num(uv)
        d = map_coordinates(dt, [uv[:, 1], uv[:, 0]], order=1, mode="constant", cval=tau)
        d = np.minimum(d, tau)
        d[gecersiz] = tau
        if capa_g is None:
            return d
        uvc, _ = _yansit(G, capa_s)
        return np.concatenate([d, capa * np.nan_to_num(np.linalg.norm(uvc - capa_g, axis=1), nan=50.0)])

    x0 = G0.flatten()[:8]
    once = float(np.mean(artik(x0)[:len(sablon)]))
    t0 = time.time()
    sonuc = least_squares(artik, x0, loss="soft_l1", f_scale=2.0, x_scale="jac", max_nfev=150)
    sonra = float(np.mean(artik(sonuc.x)[:len(sablon)]))
    bilgi = {"once_px": once / olcek, "sonra_px": sonra / olcek, "sure_sn": time.time() - t0,
             "nokta": len(sablon)}
    if sonra > (1 - en_az_iyilesme) * once:
        bilgi["durum"] = "iyileşme yetersiz"
        return kal, bilgi
    G = np.append(sonuc.x, 1.0).reshape(3, 3)
    H_ters = np.linalg.inv(S) @ G
    H = np.linalg.inv(H_ters)
    if noktalar:
        H = isareti_duzelt(H, np.array([n["px"] for n in noktalar], float))
        H_ters = np.linalg.inv(H)
    if capa_g is not None:                                       # noktalardan çok uzaklaştıysa reddet
        uvc, _ = _yansit(H_ters, capa_s)
        if np.nanmedian(np.linalg.norm(uvc - capa_g / olcek, axis=1)) > 15:
            bilgi["durum"] = "noktalardan uzaklaştı"
            return kal, bilgi
    bilgi["durum"] = "iyileştirildi"
    return HomografiSonuc(H, H_ters, kal.hata_metre, kal.hata_piksel, kal.nokta_hatalari, kal.icerdeki), bilgi


def degerlendir(kume: str, model_yolu: str, **ayar):
    """Elle etiketli kareler: model H'si ile çizgi-iyileştirilmiş H'nin metre hatası (keypoint_tahmin ile aynı tanım)."""
    from deney_degerlendir import ciftler, GRUPLAR
    from homografi import dejenere_mi
    from keypoint_tahmin import model_yukle, oku, ransac_homografi
    grup = {"val": "val", "test": "test_onerisiz"}[kume]
    model = model_yukle(model_yolu, None, 0.5)
    satirlar = []
    for g, e in ciftler(grup):
        gercek = json.loads(e.read_text(encoding="utf-8"))
        if gercek.get("durum") == "atlandi" or not gercek["noktalar"]:
            continue
        ids = [n["id"] for n in gercek["noktalar"]]
        if len(ids) < 4 or dejenere_mi(pitch.NOKTA_KOORD[ids]) is not None:
            continue
        img = oku(g)
        noktalar = model.tahmin(img)
        kal = ransac_homografi(noktalar)
        gp = np.array([n["px"] for n in gercek["noktalar"]])
        hedef = pitch.NOKTA_KOORD[ids]
        if kal is None:
            satirlar.append({"kare": g.stem, "once": None, "sonra": None, "durum": "homografi yok"})
            continue
        yeni, bilgi = iyilestir(img, kal, noktalar, **ayar)
        hata = lambda H: float(np.mean(np.linalg.norm(noktalari_donustur(H, gp) - hedef, axis=1)))
        satirlar.append({"kare": g.stem, "once": hata(kal.H), "sonra": hata(yeni.H), **bilgi})
        s = satirlar[-1]
        ek = ""
        if "once_px" in s:
            ek = f"  (çizgiye uzaklık {s['once_px']:.1f} → {s['sonra_px']:.1f} px, {s['sure_sn']:.1f} sn)"
        print(f"{g.stem:30s} {s['once']:6.2f} m → {s['sonra']:6.2f} m  {s['durum']}{ek}", flush=True)
    kurulan = [s for s in satirlar if s["once"] is not None]
    once = np.array([s["once"] for s in kurulan]); sonra = np.array([s["sonra"] for s in kurulan])
    ozet = {"kume": kume, "kare": len(satirlar), "kurulan": len(kurulan),
            "once": {"medyan": float(np.median(once)), "basari_1m": float(np.mean(once < 1)) * len(kurulan) / len(satirlar)},
            "sonra": {"medyan": float(np.median(sonra)), "basari_1m": float(np.mean(sonra < 1)) * len(kurulan) / len(satirlar)},
            "iyilesen": int(np.sum(sonra < once - 0.02)), "kotulesen": int(np.sum(sonra > once + 0.02)),
            "ayar": ayar, "kareler": satirlar}
    print(f"\n{kume}: {len(kurulan)}/{len(satirlar)} karede homografi | medyan {ozet['once']['medyan']:.3f} → "
          f"{ozet['sonra']['medyan']:.3f} m | <1 m başarı {ozet['once']['basari_1m']:.1%} → {ozet['sonra']['basari_1m']:.1%} "
          f"| iyileşen {ozet['iyilesen']}, kötüleşen {ozet['kotulesen']}")
    return ozet


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kume", choices=["val", "test"], default="val")
    ap.add_argument("--model", default=str(PROJE / "models" / "unet_kp_gercek_v1.pt"))
    ap.add_argument("--tau", type=float, default=8.0)
    ap.add_argument("--capa", type=float, default=0.5)
    ap.add_argument("--kaydet", action="store_true")
    a = ap.parse_args()
    oz = degerlendir(a.kume, a.model, tau=a.tau, capa=a.capa)
    if a.kaydet:
        (PROJE / "results" / f"cizgi_iyilestirme_{a.kume}.json").write_text(json.dumps(oz, indent=2, ensure_ascii=False),
                                                                           encoding="utf-8")
