# Usage

All commands are run from the repository root. On Windows, use `python -X utf8 …` so that Turkish console
output prints correctly. Scripts expect the folder layout below (created automatically where possible).

```
data/raw_videos/<id>.mp4               input highlight videos
data/frames/<video>/<frame>.jpg        selected frames            (kare_cikar.py)
data/labels/noktalar/<video>/*.json    keypoint labels            (etiketle.py)
data/labels/kutular/<video>/*.json     box labels                 (kutu_etiketle.py)
data/synthetic/                        synthetic dataset          (sentetik.py)
data/real_kp_<name>/                   training sets              (veri_hazirla.py)
models/*.pt                            trained weights
```

## 1. Build the dataset

```bash
# select frames from every video in data/raw_videos (max 25 per match by default)
python -X utf8 src/kare_cikar.py

# assign matches to train / val / test  → data/bolme.json
python -X utf8 src/bolme.py

# label pitch keypoints
python -X utf8 src/etiketle.py                      # continue from the first unlabelled frame
python -X utf8 src/etiketle.py --her-mactan 8       # 8 evenly spread frames per match
python -X utf8 src/etiketle.py --kareler 01:t00186 10:t00258   # only specific frames
python -X utf8 src/etiketle.py --ozet               # print progress only

# automatic label QA
python -X utf8 src/etiket_kontrol.py
```

Labelling tool: left click on empty space, type the keypoint ID (0–34, see `src/pitch.py`) and press
Enter to add a point (Esc cancels); drag an existing point to move it; wheel = zoom, right drag = pan,
`R` = reset view, `H` = geometric suggestion, `A` = accept suggestions, `D` = delete the point under the
cursor, `Z` = undo, `N` / `P` = next / previous (auto-saves), `X` = skip frame, `Q` = quit.
Red points marked `?` deviate > 1.5 m from the homography of all points (usually a wrong ID; on very wide
or lens-distorted shots correct points can also turn red).
Test matches are shown **without** model suggestions by default (add `--test-oneri` to override — not recommended).

## 2. Synthetic data

```bash
python -X utf8 src/sentetik.py --ornek 12             # a few samples + preview image
python -X utf8 src/sentetik.py --train 6000 --val 600 # full synthetic dataset
```

## 3. Training

```bash
# own U-Net: pre-train on synthetic data
python -X utf8 src/unet_keypoint.py --veri data/synthetic --ad sentetik_v1 --epoch 15

# build a real training set mixed with 1500 synthetic images, real frames oversampled 3×
python -X utf8 src/veri_hazirla.py --ad v2 --sentetik 1500 --cogalt 3

# fine-tune from the synthetic model
python -X utf8 src/unet_keypoint.py --veri data/real_kp_v2 --ad v2 --epoch 12 --lr 3e-4 \
    --baslangic models/unet_kp_sentetik_v1.pt --isci 2

# YOLO-pose baseline
python -X utf8 src/egit_keypoint.py --veri data/synthetic/saha_sentetik.yaml --ad sentetik_v2
```

Learning-curve subsets: `veri_hazirla.py --sinir K` keeps at most K real frames per training match.
On a 15 GB RAM machine keep data-loader workers ≤ 2 and never run two trainings in parallel.

## 4. Evaluation

```bash
python -X utf8 src/deney_degerlendir.py models/unet_kp_gercek_v1.pt models/unet_kp_v2.pt
python -X utf8 src/deney_degerlendir.py --tablo     # print the table from results/deneyler.json
python -X utf8 src/ogrenme_egrisi_ciz.py            # results/ogrenme_egrisi.png
python -X utf8 src/ayak_analizi.py                  # foot point vs. box centre
python -X utf8 src/keypoint_tahmin.py --model models/unet_kp_gercek_v1.pt --gorsel 15   # visual panel
```

`deney_degerlendir.py` evaluates each model on three groups: `val` (02, 10, 19), `test_04`
(labelled with suggestions) and `test_onerisiz` (11 + 12, suggestion-free — the headline numbers).

## 5. Running the system

```bash
# web interface → http://127.0.0.1:7860
python -X utf8 src/arayuz.py

# images
python -X utf8 src/sistem.py --model models/unet_kp_gercek_v1.pt --goruntu a.jpg b.jpg

# video: 120 s starting at 0 s, 12 processed frames per second, map only
python -X utf8 src/sistem.py --model models/unet_kp_gercek_v1.pt --video match.mp4 \
    --bas 0 --sure 120 --fps 12 --sadece-harita
```

Video options: `--alfa` temporal smoothing (1 = off), `--tolerans` frames to reuse the last homography,
`--kesme-esigi` scene-cut threshold (0–1), `--yolo yolov8m.pt` for the larger detector.

### Web interface

* **🎬 Match highlight → bird's-eye video**: choose a video from `data/raw_videos` or upload one, set start
  and duration (≤ 300 s), keypoint model, detector, processed fps and output layout (side by side or map only).
* **🖼️ Single frame**: pick a labelled val/test frame (the true error in metres is shown) or upload any
  image; compare keypoint models live; adjust the keypoint confidence threshold.

## Glossary of Turkish identifiers

| Turkish | English | | Turkish | English |
|---|---|---|---|---|
| saha | pitch | | oyuncu | player |
| nokta / noktalar | point(s) | | takım | team |
| etiket / etiketle | label / to label | | hakem / kaleci / top | referee / goalkeeper / ball |
| kare | frame | | kutu | bounding box |
| öneri | suggestion | | ayak noktası | foot point |
| homografi | homography | | kuş bakışı | bird's-eye |
| eğit / eğitim | train / training | | sentetik | synthetic |
| değerlendir | evaluate | | bölme / küme | split / subset |
| tahmin | prediction | | hata | error |
| güven / eşik | confidence / threshold | | sonuç | result |
| görüntü | image | | yumuşatma | smoothing |
| önerisiz | suggestion-free | | sahne kesmesi | scene cut |
