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
git clone https://github.com/UM-SWIM-360/swim-360-static-cv-dataset.git
cd swim360-dive
pip install opencv-python numpy tqdm pyyaml huggingface_hub

# 1. Fetch the images into a folder of your choosing
python fetch_hf.py --repo_id UM-SWIM-360/swim360-static-cv-dataset --output_dir [name_of_hf_directory]

# 2. Compile the registry + images into a standard COCO dataset
python compile_coco.py --registry_root . --dataset_config dataset_config.yaml `
    --output_dir dataset --images_root "[name_of_hf_directory]\blurred-images"
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


# if you want a custom train/valid/test ratio for your own experiments:

```bash
python generate_split_config.py --registry_root . --output dataset_config.yaml \
    --train_ratio 0.6 --valid_ratio 0.2 --test_ratio 0.2 --force
```

`--force` is required since `dataset_config.yaml` already exists — this
overwrites it with a freshly-computed (still participant-disjoint) split.
Save it under a different filename instead if you want to keep the original
alongside your custom one.
