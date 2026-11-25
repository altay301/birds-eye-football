# Annotation Protocol

Part A covers pitch keypoints, Part B bounding boxes.

# Part A — Pitch Keypoints

This protocol was fixed before annotation started and applied to every frame of the dataset, so that the
target the model learns is defined consistently. Keypoint IDs refer to the template in `src/pitch.py`
(35 points, IDs 0–13 on the left half, 14–20 on the half-way line and centre circle, 21–34 on the right half).

## 1. General rules

1. **Annotate every visible keypoint.** An unannotated but visible keypoint is an explicit negative for
   the heatmap model ("no keypoint here") and is the most harmful labelling error.
2. **Sparse frames are still informative.** Frames with only one or two visible keypoints are annotated;
   four non-collinear points are needed for a homography, not for keypoint training.
3. **Do not guess.** A keypoint whose position cannot be seen is left out.
4. **Line centre.** Points are placed at the centre of the painted line width, at 4–8× zoom.

## 2. Keypoint definitions

| Keypoint | Definition |
|---|---|
| Pitch corners; penalty-area and goal-area corners | Centre of the intersection of the two line centrelines |
| Where a line meets the goal line or touchline (IDs 1–4, 14, 18, 22–25) | Centre of the intersection |
| Penalty arc ends (10, 11, 31, 32) | Where the arc meets the penalty-area line |
| Penalty arc top (13, 34) | Point of the arc farthest from the goal, i.e. the intersection of the arc with the line through the goal centre and the penalty spot (the arc tangent is parallel to the penalty-area line there). 3–5 px ambiguity along the curve is expected |
| Penalty spots (12, 33), centre spot (16) | Centre of the painted spot |
| Centre circle top / bottom (15, 17) | Intersection of the circle with the half-way line |
| Centre circle left / right (19, 20) | End points of the circle diameter **perpendicular to the half-way line** — *not* the horizontal extremes of the projected ellipse. Equivalent construction: the line through the centre spot parallel to the touchlines. (Introducing this rule reduced the calibration error of a pilot frame from 0.45 m to 0.22 m.) |

## 3. Special cases

| Situation | Decision |
|---|---|
| Keypoint behind a player, lines clearly visible on both sides | Annotate |
| Keypoint behind a player, position ambiguous | Do not annotate |
| Keypoint at the image border | Annotate only if its centre lies inside the image |
| Goal net in front of the line | Annotate if the line is discernible through the net |
| Score bug / logo covers the keypoint | Do not annotate |
| Replay graphics, ceremonies, close-ups, split screen | Skip the frame |
| Camera facing the goal (touchlines not visible), spider-cam behind the goal | Skip the frame: the side cannot be determined, and because the pitch is symmetric a mirrored labelling also yields a low reprojection error, so a low error does not validate the IDs |
| Only an arc of the centre circle visible, half-way line outside the image | Do not annotate 19/20 (the perpendicular diameter is undefined); skip the frame if nothing else is visible |
| Only a straight line visible, no intersection | Skip the frame |
| Pitch not 105 × 68 m (two venues) | Annotate normally; IDs are dimension-independent |
| Slight motion blur / compression | Annotate if the line centre can be placed within ±2–3 px |
| Left/right convention | The goal on the left of the image is the left goal (IDs 0–13) |

Frame states: *labelled* (≥ 1 point, used for training/evaluation), *empty* (explicitly no visible
keypoint), *skipped* (unusable, never used). In case of doubt — in particular in test matches — the frame
is skipped.

## 4. Annotation modes

The annotation tool (`src/etiketle.py`) supports three modes, recorded per frame in the label file
(`"yontem"` field) and reported separately in the paper:

**Model-assisted.** Predictions of the current model are shown as suggestions; the annotator accepts,
moves, deletes and adds points. Typical suggestion errors that had to be corrected: points at the base of
the goal post (not a template point), goal-area corners given penalty-area IDs, IDs 19/20 placed at the
ellipse extremes.

**Suggestion-free (test matches 11 and 12).** In test match 04, annotated with suggestions, 66 % of the
final labels were identical to the suggestion, which biases evaluation towards the suggesting model. For
the headline test set the tool therefore hides all model output. The annotator places the 4–5 clearest,
well-spread keypoints; the tool then fits a homography and proposes further points **geometrically**
(independent of any learned model). Because extrapolation amplifies click noise, a point is proposed only
if its Monte-Carlo positional uncertainty is below 3 px (pilot: 25 % of visible points proposed after
5 clicks, 64 % after 8 clicks; median proposal error ≈ 1.5 px). Every accepted proposal is inspected at
zoom; occluded proposals are deleted.

**Automatic pre-label with human review** (validation match 19 and 8 frames each of training matches
08, 09, 13–18). Model predictions with confidence ≥ 0.5 are filtered by RANSAC (only inliers with
< 1 m reprojection error are kept), inspected visually, and then reviewed frame by frame by the human
annotator (`etiketle.py --kontrol`). During review, 33 missing keypoints were added in 19 of 91 frames.
This mode is never used for test matches.

## 5. Quality control

* Live feedback: the tool shows the reprojection error of the current homography; points deviating by more
  than 1.5 m are highlighted (usually a wrong ID; on very wide or lens-distorted shots correct points may
  also exceed the threshold).
* Post-hoc check: `src/etiket_kontrol.py` lists frames with high homography error for re-inspection.

# Part B — Bounding Boxes (players, goalkeepers, referees, ball)

Tool: `src/kutu_etiketle.py --her-mactan 5` — 5 frames per match (95 in total), drawn evenly from the
frames that already carry keypoint annotations, so that player positions can later be evaluated in metres
through the manually annotated homography. Boxes proposed by a COCO-pretrained YOLOv8m (confidence ≥ 0.3,
1280 px) are shown as suggestions and corrected by the annotator.

| Class | Who |
|---|---|
| `oyuncu` (player) | Outfield players of both teams on the pitch |
| `kaleci` (goalkeeper) | Goalkeepers |
| `hakem` (referee) | Referee and assistant referees, including assistants on the touchline |
| `top` (ball) | The match ball, on the ground or in the air, if it can be identified |

Rules:
1. **Delete** boxes of everyone else: spectators, staff, ball boys, photographers, substitutes warming up,
   people on advertising boards. They become negatives for the detector.
2. **Tight boxes** from the top of the head to the lowest point of the feet. The bottom edge defines the
   foot point used for the pitch projection and must touch the feet.
3. If the feet are hidden behind another player, extend the box to the estimated foot position; label only
   persons of whom roughly a third or more is visible.
4. **Ball:** tight box; when it is too blurred or hidden to be located within a few pixels, do not label it.
5. Every person on the pitch is labelled, also when partly outside the frame (box clipped at the border).

Known bias: as for keypoints, suggestion-based annotation favours the model that produced the suggestions
(here the COCO baseline). Any improvement of a fine-tuned detector measured on these labels is therefore a
conservative estimate.