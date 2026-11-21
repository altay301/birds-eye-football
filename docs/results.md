# Results

All numbers are produced by scripts in `src/` and stored in `results/deneyler.json`,
`results/guven_araligi.json` and `results/ayak_analizi.json`. Unless stated otherwise, the test set is
**matches 11 + 12, labelled without model suggestions** (41 frames, 397 keypoints). Pixel errors are
rescaled to a 1280 px image width.

**Statistical treatment.** Keypoints within a frame share camera, lighting and occlusions and are not
independent; the frame is therefore the sampling unit. 95 % confidence intervals (CI) are percentile
intervals of a frame-level bootstrap with B = 10,000 resamples (seed 0). Differences between models are
assessed with the **paired** bootstrap (both models evaluated on the same resampled frames); a difference
is called significant when its 95 % CI excludes zero. Marginal intervals are wide (≈ ±12 points) because
frame difficulty varies strongly, but this variation is shared by all models and largely cancels in the
paired comparison.

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

95 % CI of PCK@10 (frame bootstrap): YOLO-pose synthetic + real [9.2, 22.7]; U-Net synthetic only
[60.4, 83.5]; U-Net real only [58.7, 84.5]; U-Net v1 [68.1, 91.0]; U-Net v2 [67.4, 91.0].
U-Net v1 − YOLO-pose: +65.0 points, paired CI [+53.5, +76.0].

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

**Questions.** (Q1) Does synthetic pre-training improve landmark detection on real broadcast frames?
(Q2) How does accuracy scale with the number of real annotated frames, with and without synthetic data?

**Design.** Two training regimes are compared at four amounts of real data:

* **A** — initialise from the synthetic-only model, train on real frames mixed with 1,500 synthetic images;
* **B** — initialise the encoder from ImageNet, train on real frames only.

Real data: at most 3 / 5 / 8 / all frames per training match (13 matches → 39 / 65 / 89 / 143 frames;
some matches have fewer than 8 annotated frames). Subsets are nested (same seed). Every run receives the
same compute budget: 12 epochs of ≈ 1,930 images, obtained by oversampling the real frames, so that
differences reflect the data and not the number of optimisation steps.

**Reference.** The synthetic-only model (0 real frames) was trained in a separate run with a different
schedule; it is therefore reported as a reference level and **not** as the first point of curve A.

**PCK@10 (%) with 95 % CI**

| Real frames | 39 | 65 | 89 | 143 |
|---|---|---|---|---|
| A: synthetic pre-training + mix | 79.3 [66.2, 90.2] | 78.6 [65.4, 89.5] | 79.3 [66.1, 90.1] | 80.4 [67.4, 91.0] |
| B: real only | 59.4 [47.2, 71.2] | 67.5 [54.7, 79.2] | 66.0 [52.5, 78.3] | 72.5 [58.7, 84.5] |
| Reference: synthetic only (0 real frames) | 72.8 [60.4, 83.5] | | | |

**Paired differences in PCK@10 (percentage points, 95 % CI)**

| Comparison | Difference | 95 % CI | Significant |
|---|---|---|---|
| A − B at 39 real frames | +19.9 | [+14.3, +26.5] | yes |
| A − B at 143 real frames | +7.8 | [+3.5, +12.8] | yes |
| A: 143 − 39 real frames | +1.0 | [−1.0, +2.9] | no |
| B: 143 − 39 real frames | +13.1 | [+8.4, +18.0] | yes |
| A (39 real frames) − synthetic only | +6.5 | [+3.9, +9.7] | yes |
| synthetic only − B (143 real frames) | +0.3 | [−4.8, +5.4] | no |

![Learning curve and paired differences](../results/learning_curve.png)

*(a) PCK@10 versus the number of real training frames; dashed line: synthetic-only reference.
(b) Paired differences with 95 % bootstrap CI; filled markers: CI excludes zero.*

**Findings.**

1. **Q1 — synthetic pre-training helps at every data size**, most strongly in the low-data regime
   (+19.9 points at 39 frames, +7.8 points at 143 frames).
2. **Q2 — with synthetic pre-training, accuracy saturates early.** Adding 104 real frames beyond 39 yields
   no significant improvement (+1.0, CI [−1.0, +2.9]). Without synthetic data, accuracy still increases
   significantly over the same range (+13.1).
3. **Real data remains necessary.** Fine-tuning on only 39 real frames improves significantly over the
   synthetic-only model (+6.5), i.e. synthetic data narrows but does not close the domain gap.
4. **Synthetic data alone versus 143 real frames alone:** no significant difference (+0.3, CI [−4.8, +5.4]).
   This is a non-significant difference, not a demonstrated equivalence; the interval admits differences of
   about ±5 points. The statement "synthetic data is worth about 140 real frames" should be read in this sense.
5. Non-monotonic steps (A: 79.3 → 78.6; B: 67.5 → 66.0) lie well within sampling noise (1 point ≈ 4 keypoints).

Since curve A is flat beyond 39 frames, the remaining error is attributed to localisation precision rather
than data quantity (PCK@5 ≈ 69 % versus PCK@20 ≈ 85 %).

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

95 % CI of the success rate (frame bootstrap): YOLO-pose [13.2, 41.0]; U-Net synthetic only [65.7, 91.7];
U-Net real only [65.0, 91.7]; U-Net v1 and v2 [75.6, 97.3].

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

## 6. Player, referee and ball detection

### 6.1 Fine-tuned detector versus COCO baseline

Box annotations: 95 frames (5 per match), 1,680 boxes (1,422 players, 120 referees, 67 goalkeepers,
71 balls), split by match like the keypoints (train 65 / val 15 / test 15 frames). YOLOv8m initialised from
COCO was fine-tuned for 80 epochs at 1280 px (11 min on an RTX 4070 Laptop; checkpoint selected on
validation). Evaluation on the 15 test frames (matches 04, 11, 12; `src/tespit_degerlendir.py`,
`results/tespit.json`). Because the COCO model has no role classes, the comparison uses the two tasks both
models share: *person* (player, goalkeeper and referee merged) and *ball*. Operating point: confidence ≥ 0.25.
Differences: paired frame bootstrap, B = 5,000.

| | Person AP50 | Person AP50-95 | Person recall / precision | Ball AP50 | Ball recall / precision |
|---|---|---|---|---|---|
| COCO YOLOv8m | 0.800 | 0.777 | 0.90 / 0.49 | 0.631 | 0.80 / 0.33 |
| **Fine-tuned (ours)** | **0.934** | 0.793 | 0.94 / **0.80** | 0.545 | 0.50 / **0.83** |
| Difference [95 % CI] | **+0.134 [+0.062, +0.206]** | | | −0.086 [−0.363, +0.212] | recall −0.30 [−0.70, +0.11] |

* **Person detection improves significantly**, mainly through precision: the COCO model also detects
  spectators, staff and ball boys, which the annotation protocol treats as negatives.
* **Ball:** no significant difference. The fine-tuned model produces far fewer false balls (precision
  0.33 → 0.83) but misses more (recall 0.80 → 0.50). The test set contains only 10 balls, so the estimate
  is very uncertain.
* **Role classes** (fine-tuned model only), AP50: player 0.929, referee 0.591, goalkeeper 0.527, ball 0.545.

**Referee separation** (detected persons at confidence ≥ 0.25 matched to ground truth; rows: true role,
columns: predicted role):

| | player | goalkeeper | referee |
|---|---|---|---|
| player (205) | 198 | 0 | 7 |
| goalkeeper (10) | 4 | 5 | 1 |
| referee (18) | 2 | 1 | **15** |

15 of 18 referees (83 %) are recognised and therefore hidden on the bird's-eye map (with the COCO model all
of them appear as players); 7 of 205 players (3 %) are wrongly hidden. Goalkeepers (42 training examples)
are recognised only half of the time.

**Annotation bias.** 92 % of the final boxes kept the exact coordinates proposed by the COCO model. Metrics
that depend on precise box placement therefore favour the COCO model: its foot-point error with respect to the
labels is 0.00 m (median) versus 0.05 m for the fine-tuned model, which reflects shared coordinates, not
better localisation. AP50-95 is affected in the same direction. The reported improvements are conservative.

### 6.2 Label-free statistics of the COCO model (251 frames)

| Statistic | Value |
|---|---|
| "Person" detections per frame (median) | 20 |
| Frames with the ball detected | 59 % |
| Foot points inside the pitch (±3 m) | 77 % |

~23 % of person detections fall outside the pitch (assistant referees, ball boys, staff, front-row fans);
the homography itself is used to filter them.

## 7. Error analysis (U-Net v1 on 11 + 12)

33 of 38 frames succeed. The 5 failures:

| Frame | Outcome | Cause |
|---|---|---|
| 11 / t00003 | 8.7 m error | Very wide, high pre-match shot of the whole pitch; thin distant lines, bottom covered by graphics — outside the synthetic camera range |
| 12 / t00139 | no homography (8 inconsistent points) | Only one penalty box visible, no half-way line → which goal is ambiguous; left/right IDs mixed up (mirror symmetry) |
| 12 / t00369 | no homography (6 points) | Low angle near the goal line; players and goal net occlude the lines |
| 12 / t00046, t00105 | no homography (1–3 points) | Close-ups with too few visible landmarks |

## 8. Temporal stability in video

No ground-truth trajectories exist for the videos, but genuine motion (running, camera pans) is smooth while
jitter alternates from frame to frame. Smoothness is therefore measured by the norm of the **second
difference** of a position, `‖p(t+1) − 2p(t) + p(t−1)‖` (metres, zero for constant-velocity motion), over
consecutive sampled frames within a shot (12 fps):

* *homography jitter*: second difference of a fixed 6 × 4 image grid projected to the pitch;
* *player jitter*: second difference of the pitch position of the same tracked identity;
* *distance to raw*: distance between the smoothed and the unsmoothed position of a player in the same frame
  — a guard against over-smoothing (freezing positions would minimise jitter but maximise this distance).

Three methods are compared on the **same frames, keypoint predictions, detections and identities**:
*raw* (each frame independent), *previous* (element-wise EMA of `H`, α = 0.3, raw positions) and *new*
(homography filter with jump rejection and point-space smoothing + per-track position smoothing; §7 of the
method). The smoothing factor α ∈ {0.3, 0.5} was selected on two validation clips (matches 02 and 10): both
gave the same player jitter, α = 0.5 stayed closer to the raw positions. Results on 60 s clips from the three
test matches (`src/video_titreme.py`, `results/video_titreme.json`):

| Clip | Method | Coverage | Homography jitter, median / p90 (m) | Player jitter, median / p90 (m) | Distance to raw (m) |
|---|---|---|---|---|---|
| 12 (60–120 s) | raw | 47 % | 0.213 / 0.761 | 0.222 / 0.719 | — |
| | previous | 48 % | 0.068 / 0.198 | 0.134 / 0.437 | 0.60 |
| | **new** | 48 % | 0.099 / 0.309 | **0.042 / 0.193** | **0.49** |
| 11 (90–150 s) | raw | 79 % | 0.181 / 1.057 | 0.299 / 1.282 | — |
| | previous | 82 % | 0.050 / 0.283 | 0.148 / 0.517 | 0.51 |
| | **new** | 82 % | 0.075 / 0.463 | **0.046 / 0.237** | **0.40** |
| 04 (120–180 s) | raw | 47 % | 0.209 / 1.285 | 0.321 / 3.187 | — |
| | previous | 47 % | 0.047 / 0.326 | 0.135 / 0.867 | 0.27 |
| | **new** | 47 % | 0.078 / 0.508 | **0.046 / 0.413** | 0.30 |

* **Player jitter** is reduced by a factor of ≈ 3 relative to the previous method and 5–7 relative to raw
  output, while the smoothed positions stay as close to (clips 12, 11) or nearly as close to (clip 04) the raw
  positions as before; the improvement is not obtained by freezing players.
* **Homography jitter** of the new filter is somewhat higher than that of the previous matrix EMA at the
  selected α (less smoothing, less lag). However, the matrix EMA is not robust: on validation clip 10 its
  90th-percentile homography jitter reached 4.06 m — worse than raw output (2.79 m) — whereas the new filter
  stayed at 0.82 m. Jump rejection was triggered 15 times on the test clips (0 / 9 / 6).
* Coverage (frames with a homography) is unchanged; it is limited by the keypoint detector, not by the
  temporal processing (very wide shots with few landmarks remain unmapped).
* Limitations of this evaluation: the metric measures smoothness, not accuracy; parameters were selected on
  two short validation clips; one run per clip.

## 9. Video demo

Juventus – Inter highlight (test match 12), 15 s at 12 processed fps: 104 of 180 frames mapped (58 %; the
rest were close-ups, crowd shots and replays, correctly marked "pitch not visible"), 3 scene cuts detected,
26 s processing time on an RTX 4070 Laptop.
