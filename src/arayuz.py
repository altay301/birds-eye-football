"""
arayuz.py — Projenin tarayıcıda çalışan demo arayüzü (Gradio)
=============================================================

İki sekme:
  🎬 Maç özeti → kuş bakışı video (sistem.video_isle): video karelere ayrılır, her kare işlenir,
     sonuç tarayıcıda oynayan bir videoya yazılır (results\\sistem\\arayuz\\).
  🖼️ Tek kare: aşağıdaki adımlar + elle etiketli karelerde gerçek hata ölçümü.

Bir yayın karesi seç (hazır listeden ya da kendi görüntünü yükle) → sistem:
    1) saha keypoint modeli (U-Net / YOLO-pose) → saha noktaları
    2) RANSAC homografi → hatalı noktaları ayıklar
    3) YOLO → oyuncular ve top, forma rengi → takımlar
    4) ayak noktaları × homografi → kuş bakışı mini harita

Hazır listedeki kareler ELLE ETİKETLİ olduğu için, modelin homografisinin
gerçekte kaç metre şaştığı da gösterilir (elle tıklanan noktalar sahaya taşınıp
gerçek konumlarıyla karşılaştırılır).

Kullanım:  .venv\\Scripts\\python.exe -X utf8 src\\arayuz.py
           → tarayıcıda http://127.0.0.1:7860
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

import pitch
from homografi import noktalari_donustur
from keypoint_tahmin import model_yukle, oku, ransac_homografi
from manuel_kalibrasyon import sonuc_uret

PROJE = Path(__file__).resolve().parent.parent
KARELER = PROJE / "data" / "frames"
ETIKET = PROJE / "data" / "labels" / "noktalar"
BOLME = json.loads((PROJE / "data" / "bolme.json").read_text(encoding="utf-8"))
VIDEO_CIKTI = PROJE / "results" / "sistem" / "arayuz"

SAHA_MODELLERI = {
    "U-Net, gerçek + sentetik (en iyi) — unet_kp_gercek_v1": "unet_kp_gercek_v1.pt",
    "U-Net v2 — unet_kp_v2": "unet_kp_v2.pt",
    "U-Net, sadece sentetik — unet_kp_sentetik_v1": "unet_kp_sentetik_v1.pt",
    "U-Net, sentetiksiz (B, 143 kare) — unet_kp_egri_Bhep": "unet_kp_egri_Bhep.pt",
    "YOLO-pose, ince ayarlı — saha_kp_gercek_v2": "saha_kp_gercek_v2.pt",
}
OYUNCU_MODELLERI = {"YOLOv8m (daha iyi, yavaş)": "yolov8m.pt", "YOLOv8n (hızlı)": "yolov8n.pt"}
_onbellek: dict = {}


def saha_modeli(ad: str, esik: float):
    anahtar = (ad, esik)
    if anahtar not in _onbellek:
        _onbellek.clear()               # 8 GB GPU: aynı anda tek model tut
        _onbellek[anahtar] = model_yukle(str(PROJE / "models" / SAHA_MODELLERI[ad]), esik=esik)
    return _onbellek[anahtar]


def hazir_kareler() -> dict[str, Path]:
    """Elle etiketli test/val kareleri: 'görünen ad' → görüntü yolu."""
    sonuc = {}
    for kume in ("test", "val"):
        for e in sorted(ETIKET.glob("*/*.json")):
            if BOLME.get(e.parent.name) != kume:
                continue
            d = json.loads(e.read_text(encoding="utf-8"))
            g = KARELER / e.parent.name / f"{e.stem}.jpg"
            if d.get("durum") == "etiketli" and len(d["noktalar"]) >= 4 and g.exists():
                zaman = e.stem.rsplit("_", 1)[-1]
                sonuc[f"{kume} · maç {e.parent.name[:2]} · {zaman.removesuffix('.0')}"] = g
    return sonuc


HAZIR = hazir_kareler()


def noktalari_ciz(img: np.ndarray, noktalar: list[dict], kal) -> np.ndarray:
    """Model noktaları: yeşil = RANSAC kabul etti, kırmızı = attı."""
    out = img.copy()
    s = out.shape[1] / 1280                       # büyük görüntüde yazılar okunur kalsın
    r, f, k = int(9 * s), 0.8 * s, max(1, int(2 * s))
    ic = kal.icerdeki if kal is not None and kal.icerdeki is not None else np.ones(len(noktalar), bool)
    for n, kabul in zip(noktalar, ic):
        p = tuple(np.round(n["px"]).astype(int))
        renk = (0, 200, 0) if kabul else (0, 0, 255)
        cv2.circle(out, p, r, renk, -1, cv2.LINE_AA)
        cv2.circle(out, p, r, (0, 0, 0), k, cv2.LINE_AA)
        yazi = f"{n['id']} ({n['guven']:.2f})"
        q = (p[0] + r + 4, p[1] - r)
        cv2.putText(out, yazi, q, cv2.FONT_HERSHEY_SIMPLEX, f, (0, 0, 0), 3 * k, cv2.LINE_AA)
        cv2.putText(out, yazi, q, cv2.FONT_HERSHEY_SIMPLEX, f, (255, 255, 255), k, cv2.LINE_AA)
    return out


def gercek_hata(etiket_yolu: Path, H: np.ndarray) -> tuple[float, float, int] | None:
    """Elle tıklanmış noktalar model homografisiyle sahaya → gerçek konuma uzaklık (m)."""
    if not etiket_yolu.exists():
        return None
    d = json.loads(etiket_yolu.read_text(encoding="utf-8"))
    if len(d.get("noktalar", [])) < 4:
        return None
    g = np.array([n["px"] for n in d["noktalar"]])
    s = pitch.NOKTA_KOORD[[n["id"] for n in d["noktalar"]]]
    tahmin = noktalari_donustur(H, g)
    ok = np.isfinite(tahmin).all(1)
    if not ok.any():
        return None
    e = np.linalg.norm(tahmin[ok] - s[ok], axis=1)
    return float(e.mean()), float(e.max()), int(ok.sum())


def calistir(hazir_ad, yuklenen, model_ad, oyuncu_ad, esik):
    if yuklenen is not None:
        img = cv2.cvtColor(yuklenen, cv2.COLOR_RGB2BGR)
        etiket_yolu = None
    elif hazir_ad:
        g = HAZIR[hazir_ad]
        img = oku(g)
        etiket_yolu = ETIKET / g.parent.name / f"{g.stem}.json"
    else:
        return None, None, "Bir kare seç ya da görüntü yükle."

    model = saha_modeli(model_ad, round(float(esik), 2))
    noktalar = model.tahmin(img)
    kal = ransac_homografi(noktalar)
    nokta_gorsel = cv2.cvtColor(noktalari_ciz(img, noktalar, kal), cv2.COLOR_BGR2RGB)

    if kal is None:
        return nokta_gorsel, None, (f"### ❌ Homografi kurulamadı\n"
                                    f"Model {len(noktalar)} nokta buldu; en az 4 tutarlı nokta gerekir. "
                                    "Genelde yakın plan, tribün ya da çok az çizgi görünen karelerde olur.")

    birlesik, sahadakiler, kal = sonuc_uret(img, noktalar, kal=kal, yolo=OYUNCU_MODELLERI[oyuncu_ad])
    ic = int(kal.icerdeki.sum()) if kal.icerdeki is not None else len(noktalar)
    oyuncu = sum(t.sinif != "top" for t in sahadakiler)
    top = any(t.sinif == "top" for t in sahadakiler)

    satirlar = [
        "| | |", "|---|---|",
        f"| Modelin bulduğu saha noktası | **{len(noktalar)}** / 35 |",
        f"| RANSAC'ın kabul ettiği | **{ic}** (kırmızılar atıldı) |",
        f"| Homografi iç hatası | {kal.hata_metre:.2f} m · {kal.hata_piksel:.1f} px |",
        f"| Sahadaki oyuncu / hakem | **{oyuncu}** |",
        f"| Top | {'bulundu' if top else 'bulunamadı'} |",
    ]
    if etiket_yolu is not None:
        h = gercek_hata(etiket_yolu, kal.H)
        if h is not None:
            ort, en, n = h
            durum = "✅ başarılı (< 1 m)" if ort < 1 else "⚠️ başarısız (≥ 1 m)"
            satirlar.append(f"| **Gerçek hata** (elle etiketli {n} noktaya göre) | **{ort:.2f} m** "
                            f"(en çok {en:.2f} m) — {durum} |")
    else:
        satirlar.append("| Gerçek hata | yüklenen görüntünün elle etiketi yok → ölçülemez |")

    return nokta_gorsel, cv2.cvtColor(birlesik, cv2.COLOR_BGR2RGB), "\n".join(satirlar)


def video_sureleri() -> dict[str, Path]:
    """data\\raw_videos'taki maç özetleri: 'görünen ad' → yol."""
    sonuc = {}
    for v in sorted((PROJE / "data" / "raw_videos").glob("*.mp4")):
        cap = cv2.VideoCapture(str(v))
        sn = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 25)
        cap.release()
        sonuc[f"maç {v.stem[:2]} ({BOLME.get(v.stem, '?')}) · {int(sn // 60)}:{int(sn % 60):02d}"] = v
    return sonuc


VIDEOLAR = video_sureleri()


def video_calistir(hazir_ad, yuklenen, bas, sure, model_ad, oyuncu_ad, hedef_fps, gorunum,
                   ilerleme=gr.Progress()):
    from sistem import video_isle
    yol = Path(yuklenen) if yuklenen else VIDEOLAR.get(hazir_ad)
    if yol is None:
        return None, "Bir video seç ya da yükle."
    ilerleme(0, desc="Modeller yükleniyor…")
    model = saha_modeli(model_ad, 0.5)
    cikti = VIDEO_CIKTI / f"{yol.stem[:40]}_{int(bas)}s_{time.strftime('%H%M%S')}.mp4"
    s = video_isle(model, yol, cikti, bas=float(bas), sure=float(sure), hedef_fps=float(hedef_fps),
                   yolo=OYUNCU_MODELLERI[oyuncu_ad], sadece_harita=gorunum.startswith("Sadece"),
                   ilerleme=lambda o, m: ilerleme(o, desc=m))
    if not s["yol"]:
        return None, "Video okunamadı ya da seçilen aralık videonun dışında."
    oran = s["haritalanan"] / max(1, s["islenen"])
    ozet = "\n".join([
        "| | |", "|---|---|",
        f"| İşlenen video | {s['video_sn']:.0f} sn ({s['islenen']} kare, saniyede {hedef_fps:.0f}) |",
        f"| Haritası çıkarılan kare | **{s['haritalanan']}** (%{oran * 100:.0f}) — gerisi yakın plan / tribün / tekrar |",
        f"| Algılanan sahne kesmesi | {s['kesme']} |",
        f"| İşlem süresi | {s['sure_sn']:.0f} sn |",
        f"| Dosya | `{Path(s['yol']).relative_to(PROJE)}` |",
    ])
    return s["yol"], ozet


ACIKLAMA = """
# ⚽ Yayın görüntüsünden kuş bakışı oyuncu haritası
**Saha noktası modeli** sahadaki 35 referans noktasından görünenleri bulur →
**RANSAC homografi** kamera görüntüsünü saha planına bağlar → **YOLO** oyuncuları bulur →
oyuncuların **ayak noktaları** kuş bakışı haritaya taşınır.
"""

with gr.Blocks(title="Kuş Bakışı Futbol") as demo:
    gr.Markdown(ACIKLAMA)
    with gr.Tab("🎬 Maç özeti → kuş bakışı video"):
        gr.Markdown("Bir maç özeti yükle ya da hazır listeden seç. Video karelere ayrılır, her karede saha ve "
                    "oyuncular bulunur, sonuç **kuş bakışı video** olarak oynatılır. Yakın plan ve tekrar "
                    "görüntülerinde haritada *saha görünmüyor* yazar. 1 dakikalık video ≈ 1,5–2 dakikada işlenir.")
        with gr.Row():
            with gr.Column(scale=1):
                v_hazir = gr.Dropdown(list(VIDEOLAR), value=next((k for k in VIDEOLAR if k.startswith("maç 12")),
                                                                  next(iter(VIDEOLAR), None)),
                                      label="Hazır maç özeti (data\\raw_videos)")
                v_yuklenen = gr.Video(label="…ya da kendi videonu yükle (yüklersen bu kullanılır)", height=180)
                with gr.Row():
                    v_bas = gr.Number(value=60, minimum=0, label="Başlangıç (sn)")
                    v_sure = gr.Slider(5, 300, value=30, step=5, label="Süre (sn)")
                v_model = gr.Dropdown(list(SAHA_MODELLERI), value=next(iter(SAHA_MODELLERI)),
                                      label="Saha noktası modeli")
                v_oyuncu = gr.Radio(list(OYUNCU_MODELLERI), value="YOLOv8n (hızlı)", label="Oyuncu tespit modeli")
                v_fps = gr.Slider(4, 25, value=12, step=1, label="Saniyede işlenecek kare (az → hızlı, kesik)")
                v_gorunum = gr.Radio(["Yan yana (görüntü + harita)", "Sadece kuş bakışı harita"],
                                     value="Yan yana (görüntü + harita)", label="Çıktı")
                v_calis = gr.Button("▶ Videoyu işle", variant="primary")
                v_ozet = gr.Markdown()
            with gr.Column(scale=2):
                v_sonuc = gr.Video(label="Kuş bakışı video", autoplay=True)
        v_calis.click(video_calistir, [v_hazir, v_yuklenen, v_bas, v_sure, v_model, v_oyuncu, v_fps, v_gorunum],
                      [v_sonuc, v_ozet])

    with gr.Tab("🖼️ Tek kare (gerçek hata ölçümüyle)"):
        gr.Markdown("Hazır listedeki test / val kareleri eğitimde **hiç görülmedi** ve elle etiketli olduğu için "
                    "gerçek hata (metre) de hesaplanır. Test 11 ve 12 model önerisi olmadan etiketlendi "
                    "(en dürüst ölçüm).")
        with gr.Row():
            with gr.Column(scale=1):
                hazir = gr.Dropdown(list(HAZIR), value=next(iter(HAZIR), None), label="Hazır kare (elle etiketli)")
                yuklenen = gr.Image(label="…ya da kendi görüntünü yükle (yüklersen bu kullanılır)", type="numpy",
                                    height=200)
                model_ad = gr.Dropdown(list(SAHA_MODELLERI), value=next(iter(SAHA_MODELLERI)),
                                       label="Saha noktası modeli")
                oyuncu_ad = gr.Radio(list(OYUNCU_MODELLERI), value=next(iter(OYUNCU_MODELLERI)),
                                     label="Oyuncu tespit modeli")
                esik = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Nokta güven eşiği")
                with gr.Row():
                    rastgele = gr.Button("🎲 Rastgele kare")
                    calis = gr.Button("▶ Çalıştır", variant="primary")
                ozet = gr.Markdown()
            with gr.Column(scale=2):
                nokta_gorsel = gr.Image(label="1) Modelin bulduğu saha noktaları (numara + güven; "
                                              "yeşil = RANSAC kabul, kırmızı = atıldı)")
                sonuc = gr.Image(label="2) Saha çizgileri + oyuncular (sol) → kuş bakışı harita (sağ)")

        girdiler = [hazir, yuklenen, model_ad, oyuncu_ad, esik]
        ciktilar = [nokta_gorsel, sonuc, ozet]
        calis.click(calistir, girdiler, ciktilar)
        hazir.change(lambda: None, None, yuklenen).then(calistir, girdiler, ciktilar)
        rastgele.click(lambda: (str(np.random.choice(list(HAZIR))), None), None, [hazir, yuklenen])
        demo.load(calistir, girdiler, ciktilar)

if __name__ == "__main__":
    import sys
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser="--tarayici-acma" not in sys.argv,
                allowed_paths=[str(VIDEO_CIKTI)])
