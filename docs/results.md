# Results

All numbers are produced by scripts in `src/` and stored in `results/deneyler.json` and
`results/ayak_analizi.json`. Unless stated otherwise, the test set is **matches 11 + 12, labelled without
model suggestions** (41 frames, 397 keypoints). Pixel errors are rescaled to a 1280 px image width.

## 1. Geometry sanity check

Known synthetic camera + ±1.5 px simulated click noise → mean player position error **0.32 m**
(worst 0.56 m). This is the floor that labelling noise alone imposes (`src/test_homografi.py`).

## 2. Keypoint models

| Model | Training data | Found | Median error (px) | PCK@5 | PCK@10 | PCK@20 |
|---|---|---|---|---|---|---|
| YOLO-pose | synthetic only | 38.3 % | 39.0 | 2.0 % | 4.0 % | 9.8 % |
| YOLO-pose | synthetic + real (143) | 50.1 % | 16.3 | 4.8 % | 15.6 % | 29.5 % |
| U-Net (own) | synthetic only | 82.6 % | 3.1 | 59.9 % | 72.8 % | 76.6 % |
| U-Net (own) | real only (143) | 76.8 % | 2.9 | 60.2 % | 72.5 % | 75.6 % |
| **U-Net (own)** | **synthetic + real (93), v1** | **88.9 %** | 2.7 | 68.3 % | **80.6 %** | **85.6 %** |
| U-Net (own) | synthetic + real (143), v2 | 88.2 % | **2.6** | **68.8 %** | 80.4 % | 84.6 % |

*Found* = fraction of labelled keypoints for which the model output a prediction above the confidence
threshold; median error is computed over found keypoints only; PCK@k = fraction of **all** labelled
keypoints predicted within k px (missed keypoints count as failures).

**Findings**
* The heatmap U-Net beats YOLO-pose by a wide margin. YOLO-pose regresses keypoints relative to an object
  box at 640 px; the U-Net produces a full-resolution heatmap per keypoint with a sub-pixel peak. For one
  image-filling "object" with 35 thin landmarks the heatmap formulation is clearly the better fit.
* On test match 04 — labelled *with* model suggestions — the same U-Net scores PCK@10 89.7 % and 100 % of
  frames < 1 m: the measured effect of confirmation bias. Headline numbers therefore use 11 + 12.

## 3. Learning curve (research question)

How much does synthetic data help, and how many real labels are needed? Real frames per training match:
3 / 5 / 8 / all (13 train matches → 39 / 65 / 89 / 143 frames; nested subsets). Each run gets the same
compute budget: ~1,930 images per epoch (real frames oversampled) and 12 epochs. Metric: PCK@10 on 11 + 12.

| Real frames | 0 | 39 | 65 | 89 | 143 |
|---|---|---|---|---|---|
| A: synthetic pre-training + synthetic/real mix | 72.8 | 79.3 | 78.6 | 79.3 | **80.4** |
| B: no synthetic (ImageNet encoder + real) | — | 59.4 | 67.5 | 66.0 | 72.5 |

![Learning curve](../results/ogrenme_egrisi.png)

1. **Synthetic data ≈ 143 hand-labelled frames**: the model that never saw a real frame (72.8) equals the
   model trained on all 143 real frames without synthetic data (72.5).
2. **Both together are best**: A beats B by 7–20 points at every size; the gap is largest with little data.
3. **A plateaus at 39 frames**: going from 39 to 143 real frames adds only ~1 point. With this recipe the
   bottleneck is no longer data quantity but pixel precision (PCK@5 ≈ 69 %) and model resolution.
4. Fluctuations such as 78.6 ↔ 79.3 are noise: ~400 test points, 1 point ≈ 4 keypoints.

## 4. End-to-end error in metres

The homography is fitted from the **model's** keypoints; the **hand-labelled** points are then projected
with it and compared with their true pitch positions. Only frames whose labels alone can define a
homography are counted. Success = homography found **and** mean error < 1 m.

| Model | Test | Frames | Homography found | Median error | Success (< 1 m) |
|---|---|---|---|---|---|
| YOLO-pose (synthetic + real) | 11+12 | 38 | 50 % | 0.97 m | 26 % |
| U-Net (synthetic only) | 11+12 | 38 | 92 % | 0.27 m | 79 % |
| U-Net (real only) | 11+12 | 38 | 82 % | 0.21 m | 79 % |
| **U-Net (synthetic + real, v1)** | **11+12** | 38 | **89 %** | **0.22 m** | **87 %** |
| U-Net (synthetic + real, v2) | 11+12 | 38 | 89 % | 0.22 m | 87 % |
| U-Net (synthetic + real, v1) | 04 (with suggestions) | 17 | 100 % | 0.29 m | 100 % |

When a homography is found, the error (~0.2 m) is close to the click-noise floor of section 1. Most remaining
failures are frames where no consistent homography exists at all.

## 5. Foot point vs. box centre

Homographies from **hand labels** (mean error < 0.5 m) on 251 real frames; 5,081 player boxes, foot point
and box centre projected separately.

| | Median | Mean | 90th pct. | Max |
|---|---|---|---|---|
| Box centre − foot point distance | **4.2 m** | 4.4 m | 6.6 m | 13.5 m |

The box centre would introduce a systematic error ~20× larger than the model's own error (~0.2 m), and it
grows for distant players (flatter viewing angle). The share of positions inside the pitch also drops
(77 % → 74 %).

## 6. Player detection (off-the-shelf YOLOv8m, COCO)

No box labels exist yet, so mAP is not reported. Label-free statistics on the same 251 frames:

| Statistic | Value |
|---|---|
| "Person" detections per frame (median) | 20 |
| Frames with the ball detected | 59 % |
| Foot points inside the pitch (±3 m) | 77 % |

~23 % of person detections fall outside the pitch (assistant referees, ball boys, staff, front-row fans);
the homography itself is used to filter them. The ball is small and motion-blurred → a football-specific
detector is the largest expected gain (box labelling tool: `src/kutu_etiketle.py`).

## 7. Error analysis (U-Net v1 on 11 + 12)

33 of 38 frames succeed. The 5 failures:

| Frame | Outcome | Cause |
|---|---|---|
| 11 / t00003 | 8.7 m error | Very wide, high pre-match shot of the whole pitch; thin distant lines, bottom covered by graphics — outside the synthetic camera range |
| 12 / t00139 | no homography (8 inconsistent points) | Only one penalty box visible, no half-way line → which goal is ambiguous; left/right IDs mixed up (mirror symmetry) |
| 12 / t00369 | no homography (6 points) | Low angle near the goal line; players and goal net occlude the lines |
| 12 / t00046, t00105 | no homography (1–3 points) | Close-ups with too few visible landmarks |

## 8. Video demo

Juventus – Inter highlight (test match 12), 15 s at 12 processed fps: 104 of 180 frames mapped (58 %; the
rest were close-ups, crowd shots and replays, correctly marked "pitch not visible"), 3 scene cuts detected,
26 s processing time on an RTX 4070 Laptop.
