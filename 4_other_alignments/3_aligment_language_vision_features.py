"""
alignment_language_vision_features.py

This script computes representational alignment between language-based and vision-based model features
using a variety of similarity metrics, such as RSA, CKA, and mutual k-NN.

It compares each language model (e.g., trained on captions or text descriptions) against
each vision model (e.g., CLIP, DINO, etc.) using batches of image features, and stores the resulting
alignment scores in a `.parquet` file.

### Use case

Helps quantify cross-modal representational similarity, and identify which vision-language pairs
share structure in their internal representations.

### Metrics Supported
- `rsa`: Pearson correlation on flattened RDMs
- `unbiased_cka`: CKA metric with bias correction
- `mutual_knn`: Mutual nearest neighbors in feature space (requires `topk`)

### Configuration

- Vision models are selected from files in `/mnt/tecla/Results/convergence/features/nsd/all/` 
  that do **not** end with `"pixtral"` or `"coco"`.
- Language models are selected from the same folder, filtering those ending in `"coco"`.

### Output
A `.parquet` file (`alignment_models_vision_coco.parquet`) containing one row per:
- model pair (vision, language)
- image batch
- metric
With fields: `model_x`, `model_y`, `batch`, `batch_size`, `metric`, `score`, etc.

### Example
To run:
```bash
python alignment_language_vision_features.py
```
"""

from tqdm import tqdm, trange
from pathlib import Path
import gc
import torch
import pandas as pd
from dmf.alerts import alert, send_message
from convergence.alignment import compute_alignment


@alert()
def compute_alignment_models(
    models_vision, models_language, metrics, batch_size, output_filename, simetric=False
):
    all_scores = []
    for i, model_x_path in (
        pbar := tqdm(list(enumerate(models_language)), leave=False, position=0)
    ):
        pbar.set_description(f"Model {model_x_path.stem}")
        features_x = torch.load(model_x_path, weights_only=True)

        for j, model_y_path in (
            pbar2 := tqdm(list(enumerate(models_vision)), leave=False, position=1)
        ):
            if simetric and (i > j):
                continue

            pbar2.set_description(f"Model {model_y_path.stem}")
            if simetric and (i == j):
                features_y = features_x
            else:
                features_y = torch.load(model_y_path, weights_only=True)

            for n in trange(
                0,
                len(features_x["feats"]),
                batch_size,
                leave=False,
                position=2,
                desc="Batch",
            ):
                features_x_batch = features_x["feats"][n : n + batch_size]
                features_y_batch = features_y["feats"][n : n + batch_size]

                for metric, kwargs in metrics:
                    info = {
                        "model_x": model_x_path.stem,
                        "model_y": model_y_path.stem,
                        "batch": n,
                        "batch_size": batch_size,
                    }

                    try:
                        scores = compute_alignment(
                            x_feats=features_x_batch,
                            y_feats=features_y_batch,
                            metric=metric,
                            **kwargs,
                        )
                        add_score_info(scores, info)
                        all_scores.extend(scores)
                    except KeyboardInterrupt as e:

                        # Save checkpoint of scores
                        df = pd.DataFrame(all_scores)
                        output_filename = Path("interrupted.parquet")
                        df.to_parquet(output_filename, index=False)

                        raise e
                    except Exception as e:

                        print(f"Error: {e}")
                        send_message(f"Error computing {info}")
                        df = pd.DataFrame(all_scores)
                        output_filename = Path("error.parquet")
                        df.to_parquet(output_filename, index=False)

            del features_y
            gc.collect()

        del features_x
        gc.collect()

    df = pd.DataFrame(all_scores)
    output_filename = Path(output_filename)
    df.to_parquet(output_filename, index=False)
    return output_filename


def add_score_info(scores, info):
    for score in scores:
        score.update(info)


def main():

    folder = Path("/mnt/tecla/Results/convergence/features/nsd/all/")
    model_paths = list(folder.glob("*pt"))

    # Thos that its stem does not end with pixtral or coco
    models_vision = [model for model in model_paths if not model.stem.endswith(("pixtral", "coco"))]
    models_language = [model for model in model_paths if model.stem.endswith("coco")]


    metrics = [
        ("rsa", {}),
        ("unbiased_cka", {}),
        ("mutual_knn", {"topk": 10}),
    ]
    compute_alignment_models(
        models_vision=models_vision,
        models_language=models_language,
        metrics=metrics,
        batch_size=750,
        output_filename="alignment_models_vision_coco.parquet",
    )


if __name__ == "__main__":
    main()
