"""
arayuz.py — Projenin tarayıcıda çalışan demo arayüzü (Gradio, İngilizce metinler)
================================================================================

İki sekme:
  * Match highlights → bird's-eye video (sistem.video_isle): video karelere ayrılır, her kare işlenir,
     sonuç tarayıcıda oynayan bir videoya yazılır (results\\sistem\\arayuz\\). Saha görünmeyen sahneler
     her zaman kırpılır, top haritada gösterilmez (projenin amacı oyuncu konumları).
  * Single frame: aşağıdaki adımlar + elle etiketli karelerde gerçek hata ölçümü.

Bir yayın karesi seç (hazır listeden ya da kendi görüntünü yükle) → sistem:
    1) saha keypoint modeli (U-Net / YOLO-pose) → saha noktaları
    2) RANSAC homografi → hatalı noktaları ayıklar
    3) YOLO → oyuncular, forma rengi → takımlar
    4) ayak noktaları × homografi → kuş bakışı mini harita

Renk paleti: siyah / beyaz / pembe (TEMA + CSS aşağıda).

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
    "U-Net · synthetic + real (best)": "unet_kp_gercek_v1.pt",
    "U-Net · synthetic + real, v2": "unet_kp_v2.pt",
    "U-Net · synthetic only": "unet_kp_sentetik_v1.pt",
    "U-Net · real only": "unet_kp_egri_Bhep.pt",
    "YOLOv8-pose · fine-tuned (baseline)": "saha_kp_gercek_v2.pt",
}
OYUNCU_MODELLERI = {"Fine-tuned YOLOv8m · player / goalkeeper / referee": "tespit_futbol_v3_1536.pt",
                    "YOLOv8m · COCO (baseline)": "yolov8m.pt", "YOLOv8n · COCO (fast)": "yolov8n.pt"}
OYUNCU_MODELLERI = {k: v for k, v in OYUNCU_MODELLERI.items() if (PROJE / "models" / v).exists()}
_onbellek: dict = {}


def saha_modeli(ad: str, esik: float):
    anahtar = (ad, esik)
    if anahtar not in _onbellek:
        _onbellek.clear()               # 8 GB GPU: aynı anda tek model tut
        _onbellek[anahtar] = model_yukle(str(PROJE / "models" / SAHA_MODELLERI[ad]), esik=esik)
    return _onbellek[anahtar]


def mac_adlari() -> dict[str, str]:
    """kaynaklar.csv: dosya adı (uzantısız) → 'Ev – Deplasman'."""
    yol = PROJE / "data" / "raw_videos" / "kaynaklar.csv"
    if not yol.exists():
        return {}
    out = {}
    for satir in yol.read_text(encoding="utf-8").splitlines()[1:]:
        s = satir.split(";")
        if len(s) > 3:
            out[Path(s[0]).stem] = s[1].replace(" - ", " – ")
    return out


ADLAR = mac_adlari()


def hazir_kareler() -> dict[str, Path]:
    """Elle etiketli, eğitimde görülmemiş (test/val) kareler: 'görünen ad' → görüntü yolu."""
    sonuc = {}
    for e in sorted(ETIKET.glob("*/*.json")):
        if BOLME.get(e.parent.name) not in ("test", "val"):
            continue
        d = json.loads(e.read_text(encoding="utf-8"))
        g = KARELER / e.parent.name / f"{e.stem}.jpg"
        if d.get("durum") == "etiketli" and len(d["noktalar"]) >= 4 and g.exists():
            sn = float(e.stem.rsplit("_t", 1)[-1])
            ad = ADLAR.get(e.parent.name, f"Match {e.parent.name[:2]}")
            sonuc[f"{ad} · {int(sn // 60)}:{int(sn % 60):02d}"] = g
    return sonuc


HAZIR = hazir_kareler()


def noktalari_ciz(img: np.ndarray, noktalar: list[dict], kal) -> np.ndarray:
    """Model noktaları: pembe = RANSAC kabul etti, gri = attı."""
    out = img.copy()
    s = out.shape[1] / 1280                       # büyük görüntüde yazılar okunur kalsın
    r, f, k = int(9 * s), 0.8 * s, max(1, int(2 * s))
    ic = kal.icerdeki if kal is not None and kal.icerdeki is not None else np.ones(len(noktalar), bool)
    for n, kabul in zip(noktalar, ic):
        p = tuple(np.round(n["px"]).astype(int))
        renk = (135, 45, 255) if kabul else (150, 150, 150)       # BGR: pembe / gri
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
        return None, None, log("frame", [("error", "select a frame or upload an image", "uyari")])
    kaynak = "upload" if yuklenen is not None else f'"{hazir_ad}"'

    model = saha_modeli(model_ad, round(float(esik), 2))
    noktalar = model.tahmin(img)
    kal = ransac_homografi(noktalar)
    nokta_gorsel = cv2.cvtColor(noktalari_ciz(img, noktalar, kal), cv2.COLOR_BGR2RGB)

    if kal is None:
        return nokta_gorsel, None, log(f"frame {kaynak}", [
            ("landmarks", f"{len(noktalar)}/35"),
            ("view", "not a main-camera view — try another frame", "uyari")])

    birlesik, sahadakiler, kal = sonuc_uret(img, noktalar, kal=kal, yolo=OYUNCU_MODELLERI[oyuncu_ad])
    ic = int(kal.icerdeki.sum()) if kal.icerdeki is not None else len(noktalar)
    oyuncu = sum(t.sinif != "top" for t in sahadakiler)

    satirlar = [("landmarks", f"{len(noktalar)}/35 detected · {ic} used"),
                ("players", f"{oyuncu} on the pitch")]
    if etiket_yolu is not None:
        h = gercek_hata(etiket_yolu, kal.H)
        if h is not None:
            satirlar.append(("error", f"{h[0]:.2f} m (vs. manual)", "vurgu"))
    return nokta_gorsel, cv2.cvtColor(birlesik, cv2.COLOR_BGR2RGB), log(f"frame {kaynak}", satirlar)


def video_sureleri() -> dict[str, Path]:
    """data\\raw_videos'taki maç özetleri: 'görünen ad' → yol. Eğitimde görülmemiş maçlar önce."""
    gorulmemis, egitim = {}, {}
    for v in sorted((PROJE / "data" / "raw_videos").glob("*.mp4")):
        if v.stem.startswith("14_"):            # derleme, tek maç değil
            continue
        cap = cv2.VideoCapture(str(v))
        sn = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 25)
        cap.release()
        ad = f"{ADLAR.get(v.stem, 'Match ' + v.stem[:2])} · {int(sn // 60)}:{int(sn % 60):02d}"
        if BOLME.get(v.stem) in ("test", "val"):
            gorulmemis[f"{ad} · unseen in training"] = v
        else:
            egitim[ad] = v
    return {**gorulmemis, **egitim}


VIDEOLAR = video_sureleri()


def video_calistir(hazir_ad, yuklenen, bas, sure, model_ad, oyuncu_ad, hedef_fps, gorunum, yumusat,
                   ilerleme=gr.Progress()):
    from sistem import video_isle
    yol = Path(yuklenen) if yuklenen else VIDEOLAR.get(hazir_ad)
    if yol is None:
        return None, log("video", [("error", "select a match or upload a video", "uyari")])
    komut = f'video {"upload" if yuklenen else chr(34) + hazir_ad.split(" · ")[0] + chr(34)} ' \
            f'--start {int(bas)} --duration {int(sure)}'
    ilerleme(0, desc="Loading models…")
    model = saha_modeli(model_ad, 0.5)
    cikti = VIDEO_CIKTI / f"{yol.stem[:40]}_{int(bas)}s_{time.strftime('%H%M%S')}.mp4"
    s = video_isle(model, yol, cikti, bas=float(bas), sure=float(sure), hedef_fps=float(hedef_fps),
                   yolo=OYUNCU_MODELLERI[oyuncu_ad], sadece_harita=gorunum.startswith("map"),
                   yumusat=bool(yumusat), top_goster=False, kirp=True,
                   ilerleme=lambda o, m: ilerleme(o, desc=m))
    if not s["yol"]:
        return None, log(komut, [("result", "no main-camera play in this interval — try another --start"
                                  if s["islenen"] else "video unreadable or interval out of range", "uyari")])
    return s["yol"], log(komut, [
        ("input", f"{s['video_sn']:.0f} s · {s['islenen']} frames · {s['kesme'] + 1} shots"),
        ("output", f"{s['cikti_sn']:.0f} s of live play", "vurgu"),
        ("time", f"{s['sure_sn']:.0f} s"),
    ])


def log(komut: str, satirlar: list[tuple]) -> str:
    """Terminal çıktısı gibi: '$ komut' + hizalı 'anahtar  değer' satırları + 'done'."""
    import html
    out = [f'<span class="pr">$</span> {html.escape(komut)}']
    for s in satirlar:
        k, v, sinif = (*s, "") if len(s) == 2 else s
        out.append(f'  <span class="k">{k:<10}</span><span class="{sinif}">{html.escape(v)}</span>')
    out.append('<span class="ok">done.</span>' if not any(len(s) > 2 and s[2] == "uyari" for s in satirlar)
               else "")
    return '<pre class="log">' + "\n".join(out) + "</pre>"


# --- Görünüm: siyah / beyaz / pembe ----------------------------------------------------------------
PEMBE, PEMBE_ACIK, PEMBE_KOYU = "#ff2e88", "#ff5ca3", "#d81b6a"
SIYAH, PANEL, CIZGI, METIN, SOLUK = "#000000", "#0d0d0d", "#2a2a2a", "#ffffff", "#a3a3a3"

_t = dict(
    body_background_fill=SIYAH, body_text_color=METIN, body_text_color_subdued=SOLUK,
    background_fill_primary=SIYAH, background_fill_secondary=PANEL,
    block_background_fill=SIYAH, block_border_color="#1c1c1c", block_label_background_fill=SIYAH,
    block_shadow="none", block_radius="6px", block_border_width="1px",
    block_label_text_color=SOLUK, block_title_text_color=METIN, border_color_primary=CIZGI,
    border_color_accent=PEMBE, color_accent=PEMBE, color_accent_soft="#2a0a18",
    input_background_fill="#141414", input_border_color=CIZGI, input_border_color_focus=PEMBE,
    button_primary_background_fill=PEMBE, button_primary_background_fill_hover=PEMBE_ACIK,
    button_primary_text_color=METIN, button_primary_border_color=PEMBE,
    button_secondary_background_fill="#1a1a1a", button_secondary_background_fill_hover="#262626",
    button_secondary_text_color=METIN, button_secondary_border_color=CIZGI,
    slider_color=PEMBE, checkbox_background_color_selected=PEMBE, checkbox_border_color_selected=PEMBE,
    link_text_color=PEMBE, link_text_color_hover=PEMBE_ACIK, link_text_color_visited=PEMBE_KOYU,
    loader_color=PEMBE, table_border_color=CIZGI, table_even_background_fill=PANEL,
    table_odd_background_fill="#121212",
)
_set_ok = set(__import__("inspect").signature(gr.themes.Base.set).parameters)
_MONO = [gr.themes.GoogleFont("JetBrains Mono"), "Cascadia Mono", "Consolas", "monospace"]
TEMA = gr.themes.Base(primary_hue=gr.themes.colors.pink, neutral_hue=gr.themes.colors.neutral,
                      font=_MONO, font_mono=_MONO).set(
    **_t, **{f"{k}_dark": v for k, v in _t.items() if f"{k}_dark" in _set_ok})   # açık/koyu modda aynı

# Terminal görünümü: her şey eş aralıklı yazı; kutu, gölge, dolgu yok; etiketler komut satırı bayrağı gibi
# (--match), düğmeler köşeli parantezli komut ([ run ]), çıktı '$ komut' + hizalı satırlar. Pembe = istem/vurgu.
CSS = f"""
* {{ font-family: 'JetBrains Mono', 'Cascadia Mono', Consolas, monospace !important; }}
.gradio-container {{ width: 100% !important; max-width: 1360px !important; margin: 0 auto; font-size: 14px; }}
.main {{ max-width: none !important; }}
footer {{ display: none !important; }}
.block, .form, .gr-group, .wrap, .container {{ background: transparent !important; box-shadow: none !important;
    border: none !important; }}
.block {{ padding: 2px 0 !important; }}
label span, .block-title, [data-testid="block-label"], .label-wrap span {{
    color: {SOLUK} !important; font-size: 13px !important; text-transform: none !important; }}
input, textarea, .secondary-wrap, .wrap-inner {{ background: transparent !important; border: none !important;
    border-bottom: 1px dashed #333 !important; border-radius: 0 !important; color: {METIN} !important;
    box-shadow: none !important; }}
input:focus {{ border-bottom-color: {PEMBE} !important; }}
button.lg, button.md, button.sm, button.primary, button.secondary {{
    background: transparent !important; border: none !important; box-shadow: none !important;
    color: {SOLUK} !important; justify-content: flex-start !important; padding: 4px 0 !important;
    font-weight: 400 !important; min-width: 0 !important; }}
button.primary {{ color: {PEMBE} !important; }}
button:focus, button:focus-visible {{ outline: none !important; box-shadow: none !important; }}
button.primary:hover, button.secondary:hover {{ color: {METIN} !important; }}
#baslik {{ margin: 6px 0 4px; line-height: 1.6; }}
#baslik .pr {{ color: {PEMBE}; }}
#baslik .yorum {{ color: #666; }}
.tab-nav, [role="tablist"] {{ border-bottom: 1px dashed #333 !important; }}
.tab-nav button, button[role="tab"] {{ color: #666 !important; font-size: 13px !important; padding: 4px 14px 4px 0 !important; }}
.tab-nav button.selected, button[role="tab"][aria-selected="true"] {{ color: {METIN} !important;
    border-bottom: 1px solid {PEMBE} !important; }}
.log {{ background: transparent !important; border: none !important; margin: 8px 0 0; padding: 0;
    font-size: 13px; line-height: 1.7; white-space: pre-wrap; color: {METIN}; }}
.log .pr {{ color: {PEMBE}; }}
.log .k {{ color: #666; }}
.log .vurgu {{ color: {PEMBE}; }}
.log .uyari {{ color: #ffb020; }}
.log .ok {{ color: #666; }}
.gr-accordion {{ overflow: hidden !important; }}
::-webkit-scrollbar {{ width: 6px; height: 6px; background: {SIYAH}; }}
::-webkit-scrollbar-thumb {{ background: #222; }}
* {{ scrollbar-color: #222 {SIYAH}; scrollbar-width: thin; }}
#cikti {{ border: 1px dashed #262626 !important; min-height: 420px; }}
"""

BASLIK = """
<div id="baslik"><span class="pr">PS&gt;</span> birds-eye-football
<span class="yorum"># broadcast footage → top-down player map</span></div>
"""

with gr.Blocks(title="Bird's-Eye Football") as demo:
    gr.HTML(BASLIK)
    with gr.Tab("video"):
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                v_hazir = gr.Dropdown(list(VIDEOLAR), value=next((k for k in VIDEOLAR if "Manchester" in k),
                                                                  next(iter(VIDEOLAR), None)),
                                      label="--match")
                with gr.Row():
                    v_bas = gr.Number(value=60, minimum=0, label="--start", min_width=80)
                    v_sure = gr.Number(value=30, minimum=5, maximum=300, label="--duration", min_width=80)
                v_calis = gr.Button("[ run ] ↵", variant="primary")
                with gr.Accordion("--options", open=False):
                    v_yuklenen = gr.Video(label="--input <file>", height=120)
                    v_gorunum = gr.Radio(["side-by-side", "map-only"], value="side-by-side", label="--layout")
                    v_model = gr.Dropdown(list(SAHA_MODELLERI), value=next(iter(SAHA_MODELLERI)),
                                          label="--landmarks")
                    v_oyuncu = gr.Radio(list(OYUNCU_MODELLERI), value=next(iter(OYUNCU_MODELLERI)),
                                        label="--detector")
                    v_fps = gr.Slider(4, 25, value=12, step=1, label="--fps")
                    v_yumusat = gr.Checkbox(value=True, label="--track (tracking + smoothing)")
                v_ozet = gr.HTML(log("video --help", [("ready", "pick a match, then run")]))
            with gr.Column(scale=3):
                v_sonuc = gr.Video(show_label=False, autoplay=True, height=560, elem_id="cikti")
        v_calis.click(video_calistir, [v_hazir, v_yuklenen, v_bas, v_sure, v_model, v_oyuncu, v_fps, v_gorunum,
                                       v_yumusat], [v_sonuc, v_ozet])

    with gr.Tab("frame"):
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                hazir = gr.Dropdown(list(HAZIR), value=next(iter(HAZIR), None), label="--frame  # unseen match")
                with gr.Row():
                    calis = gr.Button("[ run ] ↵", variant="primary", min_width=60)
                    rastgele = gr.Button("[ random ]", min_width=60)
                with gr.Accordion("--options", open=False):
                    yuklenen = gr.Image(label="--input <file>", type="numpy", height=120)
                    model_ad = gr.Dropdown(list(SAHA_MODELLERI), value=next(iter(SAHA_MODELLERI)),
                                           label="--landmarks")
                    oyuncu_ad = gr.Radio(list(OYUNCU_MODELLERI), value=next(iter(OYUNCU_MODELLERI)),
                                         label="--detector")
                    esik = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="--threshold")
                ozet = gr.HTML()
            with gr.Column(scale=3):
                sonuc = gr.Image(show_label=False)
                nokta_gorsel = gr.Image(label="landmarks")

        girdiler = [hazir, yuklenen, model_ad, oyuncu_ad, esik]
        ciktilar = [nokta_gorsel, sonuc, ozet]
        calis.click(calistir, girdiler, ciktilar)
        hazir.change(lambda: None, None, yuklenen).then(calistir, girdiler, ciktilar)
        rastgele.click(lambda: (str(np.random.choice(list(HAZIR))), None), None, [hazir, yuklenen])
        demo.load(calistir, girdiler, ciktilar)

if __name__ == "__main__":
    import sys
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser="--tarayici-acma" not in sys.argv,
                allowed_paths=[str(VIDEO_CIKTI)], theme=TEMA, css=CSS)
