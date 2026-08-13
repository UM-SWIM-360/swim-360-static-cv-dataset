# swim-360-static-cv-dataset

COCO-17 keypoint annotations for swimmer pose estimation, captured from a
fixed lane camera setup (full, non-segmented strokes), above and below
water.

## This is a two-part dataset:

| | Contains | Where |
|---|---|---|
| **This repo** | Annotation registry (`data/`), the pre-computed train/valid/test split (`dataset_config.yaml`), and the scripts you need to use the dataset | GitHub |
| **Images** | Face-blurred (anonymised) frames | [Hugging Face](https://huggingface.co/datasets/liamkazzopardi/swim360-static-cv-dataset) |

Images are hosted separately because they're large binary files; this repo
stays lightweight and diffable (Git-friendly JSON only). The two are matched
by file path — `data/<swimmer>/<camera_angle>/<stroke>/annotations/<video>/frame_NNNNNN.json`
pairs with `<video>/frame_NNNNNN.png` on Hugging Face.

## Quickstart

```bash
git clone https://github.com/UM-SWIM-360/swim360-static-cv-dataset.git
cd swim360-dive
pip install opencv-python numpy tqdm pyyaml huggingface_hub

# 1. Fetch the images into a folder of your choosing
python fetch_from_hf.py --repo_id UM-SWIM-360/swim360-static-cv-dataset --local_dir [name_of_hf_directory]

# 2. Compile the registry + images into a standard COCO dataset
python compile_coco.py --registry_root . --dataset_config dataset_config.yaml `
    --output_dir dataset --images_root "[name_of_hf_directory]\blurred_images"
```

That's it — `dataset_config.yaml` is already included in the repo (the
canonical published train/valid/test split), so step 2 is all you need.
This produces `dataset/{train,valid,test}/{images/, _annotations.coco.json}`,
ready to point standard COCO training pipelines at directly.

## Dataset conventions

- **Swimmer ID**: `P<number>` (e.g. `P007`)
- **Camera angle**: `Above Water` / `Under Water`
- **Stroke**: `freestyle`, `backstroke`, `breaststroke`, `butterfly`
- **Keypoints**: standard COCO-17 layout (nose, eyes, ears, shoulders,
  elbows, wrists, hips, knees, ankles)

## Repository structure

```
data/                     # Annotation registry (this repo's core content)
  P007/Under Water/freestyle/
    metadata.json         # categories, resolution, per-video stats
    annotations/<video_stem>/
      frame_000000.json   # one micro-JSON per annotated frame
dataset_config.yaml       # The published swimmer -> train/valid/test split

fetch_hf.py               # Download the image set from Hugging Face
compile_coco.py           # Merge registry + images into standard COCO JSON
generate_split_config.py  # compute your own custom split

```

Everything above is what you need to *use* the dataset. Scripts used to
*build* it in the first place (pairing raw CVAT exports with source videos,
assembling the registry, face-blurring, uploading to Hugging Face) aren't
part of this repo — they're maintainer-side tooling, not needed to train on
the dataset.


## if you want a custom train/valid/test ratio for your own experiments:

```bash
python generate_split_config.py --registry_root . --output dataset_config.yaml \
    --train_ratio 0.6 --valid_ratio 0.2 --test_ratio 0.2 --force
```

`--force` is required since `dataset_config.yaml` already exists — this
overwrites it with a freshly-computed (still participant-disjoint) split.
Save it under a different filename instead if you want to keep the original
alongside your custom one.


# Static Dataset — Metrics Summary

**Total frames:** 3,903 | **Total annotations:** 3,757 | **Image resolution:** 3840×2160 (100% of frames, all UHD) ([Hugging Face](https://huggingface.co/datasets/liamkazzopardi/swim360-static-cv-dataset))

---

## 1. Dataset Overview

| Metric | Value |
|---|---|
| Total frames | 3,903 |
| Total annotations | 3,757 |
| Total source videos | 44 |
| Participants (swimmers) | 15 |
| Strokes covered | 4 (backstroke, breaststroke, butterfly, freestyle) |
| Camera views | 2 (Above Water, Under Water) |
| Phase coverage | `full` only (no phase-segmented subsets) |


---

## 2. Coverage by Swimmer

| Swimmer | Frames | % of frames | Annotations | Videos |
|---|---|---|---|---|
| P007 | 692 | 17.7% | 566 | 6 |
| P014 | 588 | 15.1% | 582 | 8 |
| P008 | 523 | 13.4% | 521 | 4 |
| P015 | 355 | 9.1% | 355 | 5 |
| P012 | 284 | 7.3% | 280 | 3 |
| P024 | 248 | 6.4% | 248 | 3 |
| P026 | 192 | 4.9% | 182 | 2 |
| P030 | 175 | 4.5% | 175 | 3 |
| P028 | 161 | 4.1% | 162 | 2 |
| P027 | 158 | 4.0% | 158 | 2 |
| P022 | 148 | 3.8% | 146 | 2 |
| P023 | 121 | 3.1% | 120 | 1 |
| P016 | 98 | 2.5% | 100 | 1 |
| P021 | 85 | 2.2% | 85 | 1 |
| P011 | 75 | 1.9% | 77 | 1 |

**Notes:** P007, P014 and P008 together account for **46.2%** of all frames — the dataset is fairly top-heavy on three swimmers. P011, P021, and P016 are the thinnest slices (each backed by a single source video).

---

## 3. Coverage by Stroke

| Stroke | Frames | % of frames | Videos |
|---|---|---|---|
| Backstroke | 1,120 | 28.7% | 11 |
| Breaststroke | 1,096 | 28.1% | 12 |
| Freestyle | 970 | 24.9% | 12 |
| Butterfly | 717 | 18.4% | 9 |

Reasonably balanced across strokes; butterfly is the smallest category by both frames and videos.

---

## 4. Coverage by Camera View

| Camera | Frames | % of frames | Videos |
|---|---|---|---|
| Under Water | 2,557 | 65.5% | 32 |
| Above Water | 1,346 | 34.5% | 12 |

Underwater footage dominates ~2:1 over above-water, both in frame count and source video count.

---

## 5. Annotation Density (swimmers labeled per frame)

| Swimmers per frame | Frames | % of frames |
|---|---|---|
| 0 (no swimmer annotated) | 157 | 4.0% |
| 1 | 3,735 | 95.7% |
| 2 | 11 | 0.3% |

The vast majority of frames are single-swimmer. The 157 zero-annotation frames are worth auditing — likely empty/transition frames that should be confirmed as intentional negatives rather than missed annotations.

---

## 6. Keypoint Completeness per Annotation

Out of a 17-point skeleton (COCO-style: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles):

| Keypoints labeled | Annotations | % of annotations |
|---|---|---|
| 0 (bbox-only, all keypoints occluded) | 517 | 13.8% |
| 1 | 191 | 5.1% |
| 2 | 266 | 7.1% |
| 3 | 342 | 9.1% |
| 4 | 358 | 9.5% |
| 5 | 355 | 9.4% |
| 6 | 302 | 8.0% |
| 7 | 385 | 10.2% |
| 8 | 374 | 10.0% |
| 9 | 258 | 6.9% |
| 10 | 167 | 4.4% |
| 11 | 116 | 3.1% |
| 12 | 66 | 1.8% |
| 13 | 43 | 1.1% |
| 14 | 9 | 0.2% |
| 15 | 3 | 0.1% |
| 16 | 1 | 0.0% |
| 17 (fully labeled) | 4 | 0.1% |

**Notes:** Only **4 annotations (0.1%)** have the complete 17-point skeleton — expected for underwater swimming, where limbs are routinely self-occluded or out of frame. **13.8%** of annotations are bbox-only with no pose keypoints at all. The distribution peaks around 4–8 keypoints, consistent with partial-body visibility being the norm rather than the exception.

---

## 7. Keypoint Visibility Breakdown

Visibility codes: **1 = labeled, occluded/not visible** · **2 = labeled, visible** · **0 = not labeled**. Percentages are of all 3,757 annotations.

| Keypoint | Occluded (1) | Visible (2) | Not labeled (0) | Visible % |
|---|---|---|---|---|
| Nose | 2,893 (77.0%) | 864 (23.0%) | 0 | 23.0% |
| Left Eye | 3,443 (91.6%) | 313 (8.3%) | 1 (0.0%) | 8.3% |
| Right Eye | 3,068 (81.7%) | 689 (18.3%) | 0 | 18.3% |
| Left Ear | 3,625 (96.5%) | 132 (3.5%) | 0 | 3.5% |
| Right Ear | 3,472 (92.4%) | 285 (7.6%) | 0 | 7.6% |
| Left Shoulder | 2,807 (74.7%) | 950 (25.3%) | 0 | 25.3% |
| Right Shoulder | 2,406 (64.0%) | 1,351 (36.0%) | 0 | 36.0% |
| Left Elbow | 2,516 (67.0%) | 1,241 (33.0%) | 0 | 33.0% |
| Right Elbow | 2,287 (60.9%) | 1,470 (39.1%) | 0 | 39.1% |
| Left Wrist | 2,442 (65.0%) | 1,315 (35.0%) | 0 | 35.0% |
| Right Wrist | 2,247 (59.8%) | 1,510 (40.2%) | 0 | 40.2% |
| Left Hip | 2,696 (71.8%) | 1,061 (28.2%) | 0 | 28.2% |
| Right Hip | 2,194 (58.4%) | 1,563 (41.6%) | 0 | 41.6% |
| Left Knee | 2,224 (59.2%) | 1,533 (40.8%) | 0 | 40.8% |
| Right Knee | 2,023 (53.8%) | 1,734 (46.2%) | 0 | 46.2% |
| Left Ankle | 2,216 (59.0%) | 1,541 (41.0%) | 0 | 41.0% |
| Right Ankle | 1,980 (52.7%) | 1,777 (47.3%) | 0 | 47.3% |

**Notes:**
- **Facial keypoints (eyes, ears) are the least visible** — left ear is visible in only 3.5% of annotations, left eye in 8.3%. This tracks with swimmers facing away from camera or submerged/goggled.
- **Right-side joints are consistently more visible than their left-side counterparts** (e.g., right ankle 47.3% vs. left ankle 41.0%; right knee 46.2% vs. left knee 40.8%). This is a systematic asymmetry worth checking against camera placement/swimmer orientation conventions — it may reflect a real camera-side bias rather than annotation error.
- **Lower body (hips/knees/ankles) is more consistently visible than upper body facial features**, and right wrist/elbow are the most visible upper-limb points (~39–40%), likely reflecting stroke mechanics where the trailing/recovery arm faces the underwater camera more often.
- A breakdown of visibility **by stroke** and **by camera view** is also available in the source registry (`kp_visibility_by_stroke`, `kp_visibility_by_camera`) if you want that level of granularity added to the report — notably, above-water hip/knee/ankle visibility is near 100% (only 1–3 unlabeled out of 1,216 above-water annotations), while underwater visibility drops sharply for the same joints, confirming camera view is the dominant driver of occlusion.

---

## 8. Video-Level Coverage

| Dimension | Breakdown |
|---|---|
| By stroke | Breaststroke 12, Freestyle 12, Backstroke 11, Butterfly 9 |
| By camera | Under Water 32, Above Water 12 |
| By swimmer | P014 leads with 8 videos; P011, P016, P021, P023 have only 1 each |

---

## 9. Participant × Camera × Stroke Coverage Gaps

Of 120 possible (swimmer × camera × stroke × phase) combinations, **43 are covered and 77 are gaps (64.2%)**. Patterns in the gaps:

- **P007 and P008** have full above-water coverage across all 4 strokes but are missing (or nearly missing) underwater coverage for several strokes.
- **P011, P016, P021, P022, P023, P026, P027, P028, P030** have **no above-water footage at all** — every above-water combination for these 9 swimmers is a gap.
- Only **P007, P008, and P014** have any above-water representation.
- **P012 is missing above-water entirely and underwater-backstroke** specifically.

This means stroke/camera comparisons that hold swimmer identity constant are only reliably possible for P007, P008, and P014 (the three swimmers with both camera views). For everyone else, camera view and swimmer identity are confounded.

---

