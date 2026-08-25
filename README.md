# SWIM-360 Static Camera Setup Computer Vision Swimmer Pose Estimation Dataset

COCO-17 keypoint annotations for swimmer pose estimation, captured from a
fixed lane camera setup (full, non-segmented strokes), above and below
water.

![alt text](https://github.com/UM-SWIM-360/swim-360-static-cv-dataset/blob/main/assets/images/output_001.png?raw=true)


## This is a two-part dataset:

| | Contains | Where |
|---|---|---|
| **This repo** | Annotation registry (`data/`), the pre-computed train/valid/test split (`dataset_config.yaml`), and the scripts you need to use the dataset | GitHub |
| **Images** | Face-blurred (anonymised) frames | [Hugging Face](https://huggingface.co/datasets/UM-SWIM-360/swim360-static-cv-dataset) |

Images are hosted separately because they're large binary files; this repo
stays lightweight and diffable (Git-friendly JSON only). The two are matched
by file path — `data/<swimmer>/<camera_angle>/<stroke>/annotations/<video>/frame_NNNNNN.json`
pairs with `<video>/frame_NNNNNN.png` on Hugging Face.


## Quickstart
On **CMD**:
```bash
git clone https://github.com/UM-SWIM-360/swim-360-static-cv-dataset.git
cd swim-360-static-cv-dataset
python -m venv venv
venv\scripts\activate
pip install -r requirements.txt

# 1. Fetch the images into a folder of your choosing
python fetch_from_hf.py --repo_id UM-SWIM-360/swim360-static-cv-dataset --local_dir [name_of_hf_directory]

# 2. Compile the registry + images into a standard COCO dataset
python compile_coco.py --registry_root . --dataset_config dataset_config.yaml ^
    --output_dir dataset --images_root "[name_of_hf_directory]\blurred_images"
```

That's it — `dataset_config.yaml` is already included in the repo (the
canonical published train/valid/test split), so step 2 is all you need.
This produces `dataset/{train,valid,test}/{images/, _annotations.coco.json}`,
ready to point standard COCO training pipelines at directly.

## Using run_pipeleine.py
Instead of running scripts individually, 'run_pipeline.py' runs all the necessary scripts in order to generate the COCO dataset from the [Hugging Face](https://huggingface.co/datasets/UM-SWIM-360/swim360-static-cv-dataset) blurred images.
```bash
python run_pipeline.py
```

## Potential Issues with 'fetch_from_hf.py'
If an error 11001 arises at any point during the execution of 'fetch_from_hf.py', stop the execution and delete the local directory that stored the static images.
This error occurs due to not being logged in to your huggingface account, limiting the amount of images you can download. In the case that you are not logged in, use the following commands

In the virtual environment:
```bash
python -m pip install --upgrade huggingface_hub
```
Login:
```bash
hf auth login
```
After running the login command, follow the instructions on screen and run:
```bash
python fetch_from_hf.py --repo_id UM-SWIM-360/swim360-static-cv-dataset --local_dir [name_of_hf_directory]
```
**OR**
```bash
python run_pipeline.py
```

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

**Total frames:** 3,903 | **Total annotations:** 3,757 | **Image resolution:** 3840×2160 (100% of frames, all UHD) ([Hugging Face](https://huggingface.co/datasets/UM-SWIM-360/swim360-static-cv-dataset))

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

![alt text](https://github.com/UM-SWIM-360/swim-360-static-cv-dataset/blob/main/assets/images/distributions/frames_by_swimmer.png?raw=true)

![alt text](https://github.com/UM-SWIM-360/swim-360-static-cv-dataset/blob/main/assets/images/distributions/videos_by_swimmer.png?raw=true)

---

## 3. Coverage by Stroke

![alt text](https://github.com/UM-SWIM-360/swim-360-static-cv-dataset/blob/main/assets/images/distributions/frames_by_stroke.png?raw=true)

---

## 4. Coverage by Camera View

| Camera  | Videos |
|---|---|
| Under Water | 32 |
| Above Water | 12 |

![alt text](https://github.com/UM-SWIM-360/swim-360-static-cv-dataset/blob/main/assets/images/distributions/frames_by_camera.png?raw=true)

---

## 5. Annotation Density (swimmers labeled per frame)

![alt text](https://github.com/UM-SWIM-360/swim-360-static-cv-dataset/blob/main/assets/images/distributions/annotations_per_frame.png?raw=true)

---

## 6. Keypoint Completeness per Annotation

![alt text](https://github.com/UM-SWIM-360/swim-360-static-cv-dataset/blob/main/assets/images/distributions/keypoints_per_annotation.png?raw=true)



---

## 7. Keypoint Visibility Breakdown

Visibility codes: **1 = labeled, occluded/not visible** · **2 = labeled, visible** · **0 = not labeled**. Percentages are of all 3,757 annotations.

| Keypoint | Occluded (1) | Visible (2) | Not labeled (0) |
|---|---|---|---|
| Nose | 2,893 (77.0%) | 864 (23.0%) | 0 | 
| Left Eye | 3,443 (91.6%) | 313 (8.3%) | 1 (0.0%) | 
| Right Eye | 3,068 (81.7%) | 689 (18.3%) | 0 | 
| Left Ear | 3,625 (96.5%) | 132 (3.5%) | 0 | 
| Right Ear | 3,472 (92.4%) | 285 (7.6%) | 0 | 
| Left Shoulder | 2,807 (74.7%) | 950 (25.3%) | 0 | 
| Right Shoulder | 2,406 (64.0%) | 1,351 (36.0%) | 0 | 
| Left Elbow | 2,516 (67.0%) | 1,241 (33.0%) | 0 | 
| Right Elbow | 2,287 (60.9%) | 1,470 (39.1%) | 0 | 
| Left Wrist | 2,442 (65.0%) | 1,315 (35.0%) | 0 | 
| Right Wrist | 2,247 (59.8%) | 1,510 (40.2%) | 0 | 
| Left Hip | 2,696 (71.8%) | 1,061 (28.2%) | 0 | 
| Right Hip | 2,194 (58.4%) | 1,563 (41.6%) | 0 | 
| Left Knee | 2,224 (59.2%) | 1,533 (40.8%) | 0 | 
| Right Knee | 2,023 (53.8%) | 1,734 (46.2%) | 0 | 
| Left Ankle | 2,216 (59.0%) | 1,541 (41.0%) | 0 | 
| Right Ankle | 1,980 (52.7%) | 1,777 (47.3%) | 0 | 

![alt text](https://github.com/UM-SWIM-360/swim-360-static-cv-dataset/blob/main/assets/images/distributions/keypoint_visibility.png?raw=true)

**Notes:**
- **Facial keypoints (eyes, ears) are the least visible** — left ear is visible in only 3.5% of annotations, left eye in 8.3%. This tracks with swimmers facing away from camera or submerged/goggled.
- **Right-side joints are consistently more visible than their left-side counterparts** (e.g., right ankle 47.3% vs. left ankle 41.0%; right knee 46.2% vs. left knee 40.8%). This is a systematic asymmetry worth checking against camera placement/swimmer orientation conventions — it may reflect a real camera-side bias rather than annotation error.
- **Lower body (hips/knees/ankles) is more consistently visible than upper body facial features**, and right wrist/elbow are the most visible upper-limb points (~39–40%), likely reflecting stroke mechanics where the trailing/recovery arm faces the underwater camera more often.



---

## 8. Participant × Camera × Stroke Coverage Gaps

Of 120 possible (swimmer × camera × stroke × phase) combinations, **43 are covered and 77 are gaps (64.2%)**. Patterns in the gaps:

- **P007 and P008** have full above-water coverage across all 4 strokes but are missing (or nearly missing) underwater coverage for several strokes.
- **P011, P016, P021, P022, P023, P026, P027, P028, P030** have **no above-water footage at all** — every above-water combination for these 9 swimmers is a gap.
- Only **P007, P008, and P014** have any above-water representation.
- **P012 is missing above-water entirely and underwater-backstroke** specifically.

This means stroke/camera comparisons that hold swimmer identity constant are only reliably possible for P007, P008, and P014 (the three swimmers with both camera views). For everyone else, camera view and swimmer identity are confounded.

---

