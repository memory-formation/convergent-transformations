"""
extract_object_boxes.py

This script performs **object detection** over NSD stimulus images using the pre-trained
[DETR](https://huggingface.co/facebook/detr-resnet-50) model (ResNet-50 backbone) to extract
bounding boxes and object labels.

---

### Purpose
To compute object-level annotations (bounding boxes, class labels, confidence scores, area coverage)
for all NSD images. These are later used to analyze the **presence, location, and category of visual objects**,
supporting representational analyses in fMRI and models.

The extracted object statistics (e.g., `percent_area`, `label`) are stored for each image in a dataframe.

---

### Model
- **Model**: `facebook/detr-resnet-50`
- **Framework**: HuggingFace Transformers
- **Post-processing**: Uses the built-in `post_process_object_detection` with adjustable confidence threshold

---

### Inputs
- NSD stimulus image IDs (optionally restricted to **shared** stimuli)
- Raw images are loaded from disk using `get_image(...)` from `convergence.nsd`
- Optional: `--threshold` argument for detection confidence

---

### Main Steps
1. Load NSD stimulus metadata and raw images.
2. Load pre-trained DETR model and processor (on GPU).
3. For each image:
   - Run object detection using DETR
   - Keep predictions above confidence threshold
   - Extract and normalize bounding box coordinates
   - Calculate object box area as % of full image
   - Collect metadata (e.g., label, confidence, box size, image size)

---

### Output
A single Parquet file with detected object information:

| Column         | Description                                 |
|----------------|---------------------------------------------|
| `nsd_id`       | NSD image identifier                        |
| `object_index` | Index of the object detected in the image   |
| `label_id`     | COCO category ID predicted by DETR          |
| `label`        | Human-readable label for the category       |
| `confidence`   | Detection confidence score (0–1)            |
| `box`          | Raw box coordinates `[x0, y0, x1, y1]`      |
| `percent_area` | Proportion of image area covered by object  |
| `box_width`    | Width of bounding box in pixels             |
| `box_height`   | Height of bounding box in pixels            |
| `image_width`  | Image width                                 |
| `image_height` | Image height                                |

File is saved as: `detected_objects.parquet` (or custom output if `--output` is used)

---

### CLI Usage

python extract_object_boxes.py --shared --threshold 0.8 --output boxes_shared.parquet

"""
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import argparse
from convergence.nsd import get_resource, get_image
from tqdm import tqdm
import pandas as pd
from dmf.alerts import alert
from pathlib import Path

def load_images(shared=False):
    df = get_resource("stimulus")
    # If shared
    if shared:
        df = df.query("shared and exists")
    nsd_ids = list(df.nsd_id.unique())

    df_images = get_resource("images")
    return nsd_ids, df_images


def load_model(device="cuda"):
    processor = DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-50", revision="no_timm"
    )
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50", revision="no_timm"
    )
    model.eval()
    model.to(device)
    return processor, model


def process_image(nsd_id, processor, model, df_images, threshold):

    image = get_image(nsd_id=nsd_id, df_images=df_images, output_type="pil")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=threshold
    )[0]

    image_size = image.size
    image_results = []
    for i, (score, label, box) in enumerate(
        zip(results["scores"], results["labels"], results["boxes"])
    ):
        box = box.tolist()
        percent_area = (box[2] - box[0]) * (box[3] - box[1]) / (image_size[0] * image_size[1])
        image_results.append(
            {
                "nsd_id": nsd_id,
                "object_index": i,
                "label_id": label.item(),
                "label": model.config.id2label[label.item()],
                "confidence": score.item(),
                "box": box,
                "image_width": image_size[0],
                "image_height": image_size[1],
                "box_width": box[2] - box[0],
                "box_height": box[3] - box[1],
                "box_x0": box[0],
                "box_y0": box[1],
                "box_x1": box[2],
                "box_y1": box[3],
                "percent_area": percent_area,
            }
        )

    return image_results

@alert(output=True)
@torch.no_grad()
def main():
    args = argparse.ArgumentParser()
    args.add_argument("--shared", action="store_true")
    args.add_argument("--output", default="detected_objects.parquet")
    args.add_argument("--threshold", default=0.80, type=float)
    args = args.parse_args()
    threshold = float(args.threshold)
    output = Path(args.output)

    nsd_ids, df_images = load_images(args.shared)
    processor, model = load_model()

    df_results = []
    for nsd_id in tqdm(nsd_ids, leave=False):
        res = process_image(nsd_id, processor, model, df_images, threshold)
        df_results.extend(res)

    df_results = pd.DataFrame(df_results)
    df_results.to_parquet(output, index=False)
    return output

if __name__ == "__main__":
    main()