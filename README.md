# Bird's-Eye View Player Localization from Football Broadcast Images

Turn a single TV broadcast frame — or a whole match highlight video — into a **top-down tactical map**
showing where every player stands on the 105 × 68 m pitch.

```
broadcast frame ─┬─► pitch keypoint network (own ResNet-18 + U-Net) ─► RANSAC homography ─┐
                 └─► YOLOv8 person/ball detection ─► jersey colour K-means (teams) ───────┴─► foot point × H ─► bird's-eye map
```

Neural Network course project. The keypoint model, the synthetic data generator, the labelling tools and
the labelled dataset were all built from scratch for this project.

> **Language note:** source code identifiers and comments are in Turkish (course requirement);
> all documentation in this repository is in English. A glossary of the most common identifiers is in
> [docs/usage.md](docs/usage.md#glossary-of-turkish-identifiers).

---

## Highlights

| | |
|---|---|
| **Own dataset** | 19 match highlight videos → 475 automatically selected frames → **262 hand-labelled frames, 2,657 pitch keypoints** (35 keypoint types) |
| **Synthetic data** | Virtual broadcast camera + 3D pitch renderer: 6,600 images with error-free labels |
| **Own model** | ResNet-18 encoder + U-Net decoder predicting 35 heatmaps with sub-pixel peak refinement |
| **Main result** | On held-out matches labelled *without* model suggestions: **PCK@10 = 80.6 %**, homography error **< 1 m in 87 % of frames** (median 0.22 m) |
| **vs. YOLO-pose** | Same data, same test: YOLO-pose reaches only PCK@10 = 15.6 % and 26 % of frames < 1 m |
| **Learning curve** | Synthetic pre-training is worth ≈ 140 hand-labelled frames; with it, 39 real frames already reach the plateau |
| **Demo** | Gradio web app: upload a highlight video → bird's-eye video, or inspect single frames with ground-truth error |

![Learning curve](results/ogrenme_egrisi.png)

---

## How it works (short)

1. **Pitch keypoints.** A U-Net predicts one heatmap per pitch landmark (corners, box corners, arc tops,
   penalty spots, centre circle points, … 35 in total). Peaks above a confidence threshold become
   keypoint candidates.
2. **Homography.** Because the pitch is a plane, image ↔ pitch is a 3×3 homography `H`. It is fitted
   with RANSAC, which discards wrongly located keypoints, and is checked for degenerate (collinear) configurations.
3. **Players.** YOLOv8 (COCO) detects people and the ball. Players are split into two teams by K-means on
   the median jersey colour in Lab space.
4. **Projection.** Each player's **foot point** (bottom-centre of the box) is mapped through `H`.
   Using the box centre instead would shift players by a median of **4.2 m** (measured, see results).
5. **Video.** Frames are sampled at ~12 fps; scene cuts are detected with HSV-histogram distance;
   `H` is smoothed over time with an exponential moving average; frames without a visible pitch are
   marked instead of mapped.

Details: [docs/method.md](docs/method.md) · Full results: [docs/results.md](docs/results.md)

---

## Quick start

Requirements: Python 3.12, an NVIDIA GPU is recommended (developed on an RTX 4070 Laptop, 8 GB),
`ffmpeg` on `PATH` for browser-playable video output.

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available())"   # GPU visible?
python -X utf8 src/test_homografi.py   # geometry self-test with a known camera (no data needed)
```

Model weights and the dataset are **not** in this repository (see [Data availability](#data-availability)).
Put the weights into `models/` and run:

```bash
# Web demo (video tab + single-frame tab) → http://127.0.0.1:7860
python -X utf8 src/arayuz.py

# Command line: image or video → results/sistem/
python -X utf8 src/sistem.py --model models/unet_kp_gercek_v1.pt --goruntu my_frame.jpg
python -X utf8 src/sistem.py --model models/unet_kp_gercek_v1.pt --video highlights.mp4 --bas 0 --sure 120
```

Every step of the pipeline (frame extraction, labelling, synthetic data, training, evaluation) is
documented in [docs/usage.md](docs/usage.md).

---

## Repository layout

```
src/                     all code (see table below)
docs/                    English documentation (method, usage, results)
report/                  course report draft and labelling guide (Turkish)
results/                 experiment results (JSON) and self-generated figures
data/bolme.json          train / val / test split by match
data/raw_videos/kaynaklar.csv   source list of the 19 highlight videos (YouTube links, competition, stadium)
requirements.txt         exact package versions used
```

| Stage | Script | Purpose |
|---|---|---|
| Geometry | `pitch.py` | Pitch template (FIFA 105×68 m), 35 keypoints, minimap drawing |
| | `homografi.py` | Homography estimation (DLT + RANSAC), degeneracy checks, sign normalisation |
| | `test_homografi.py` | Self-test with a synthetic camera and known ground truth |
| Data | `kare_cikar.py` | Automatic frame selection from videos (pitch colour, sharpness, dHash novelty, wide-shot check) |
| | `bolme.py` | Match-level train/val/test split (prevents leakage between frames of the same match) |
| | `sentetik.py` | Synthetic broadcast images with exact keypoint labels |
| | `etiketle.py` | Keypoint labelling tool (model-assisted or suggestion-free mode) |
| | `etiket_kontrol.py` | Automatic label QA via homography consistency |
| | `kutu_oneri.py`, `kutu_etiketle.py` | Player / goalkeeper / referee / ball box proposals and labelling tool |
| | `veri_hazirla.py` | Converts labels to training sets (mixing synthetic data, subsetting, oversampling) |
| Models | `unet_keypoint.py` | **Own model**: ResNet-18 + U-Net heatmap regressor, training and inference |
| | `egit_keypoint.py` | YOLO-pose baseline training |
| | `keypoint_tahmin.py` | Inference, suggestions for labelling, PCK evaluation |
| | `oyuncu.py` | Player/ball detection and team assignment |
| Evaluation | `deney_degerlendir.py` | Evaluates all models on identical splits → `results/deneyler.json` |
| | `ogrenme_egrisi_ciz.py` | Learning-curve figure |
| | `ayak_analizi.py` | Foot point vs. box centre analysis, detection statistics |
| | `oneri_tutarlilik.py` | Label-free quality estimate (RANSAC inlier ratio) |
| System | `manuel_kalibrasyon.py` | First prototype: manual clicking → homography → minimap |
| | `sistem.py` | End-to-end automatic system for images and videos |
| | `arayuz.py` | Gradio web interface |

---

## Data availability

* **Videos and frames** come from publicly available highlight videos on YouTube (listed with links in
  `data/raw_videos/kaynaklar.csv`). They are copyrighted by the broadcasters and are therefore **not
  redistributed** here.
* **Labels** (262 frames, 2,657 keypoints) and **model weights** are available to the course staff on request.
* **Synthetic data** can be regenerated with `python src/sentetik.py --train 6000 --val 600`.

## Known limitations

* Close-up and low-angle shots near the goal expose too few pitch landmarks; the homography may fail or be wrong.
* Left/right ambiguity when only one penalty box is visible (mirror symmetry).
* The off-the-shelf COCO detector misses the small ball in ~40 % of frames and does not distinguish referees
  (a referee is assigned to one of the two colour clusters). A football-specific detector trained on the
  box-labelling tool's output is the planned fix.
* No player tracking yet: in video, player dots are re-detected every frame and may flicker.
* The ball is assumed to be on the ground (no height estimation).

## Acknowledgements

Ultralytics YOLOv8, PyTorch / torchvision (ImageNet ResNet-18), OpenCV, Gradio.
Related work that inspired the problem setup: SoccerNet camera calibration & game-state reconstruction,
the WorldCup 2014 homography dataset (Homayounfar et al., 2017), U-Net (Ronneberger et al., 2015).
