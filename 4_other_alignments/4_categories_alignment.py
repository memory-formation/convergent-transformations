"""
categories_alignment.py

This script computes the representational alignment between brain activity and semantic
object annotations (from COCO categories and persons) in the NSD dataset using similarity
metrics such as unbiased CKA, RSA, or mutual k-NN.

It quantifies how well each brain ROI encodes the presence of object categories across
natural scene stimuli, session-wise and subject-wise.

### Input Data:
- NSD betas per subject and ROI (accessed via `get_subject_roi`)
- COCO object annotations (`category`, `supercategory`, and `person` info) per stimulus
- Stimulus-to-NSD image index (`get_index("stimulus")`)

### Semantic Vectors Tested:
- `category`, `supercategory`: Count vectors (multi-hot)
- One-hot versions of above (`*_onehot`)
- `person`: Presence of body part keypoints
- `is_person`: Binary flag for presence of any person
- Versions excluding the "person" category (e.g., `category_no_person`)

### Workflow:
1. For each subject and session, and for each ROI (joined or separated):
2. Extract corresponding betas and COCO-derived object vectors.
3. Compute similarity with a selected metric:
    - `unbiased_cka` (default)
    - `cka`
    - `rsa`
    - `mutual_knn`
4. Save alignment scores in a Parquet file.

### Command-Line Arguments:
- `--join_hemispheres`: Whether to average hemispheres (180 ROIs) or keep separate (360 ROIs)
- `--metric`: Similarity metric to compute (`unbiased_cka`, `cka`, `rsa`, `mutual_knn`)

### Output:
- File: `objects_cka_similarity_[joined|separated].parquet`
- Columns: `subject`, `session`, `roi`, `tokenizer`, `score`, `metric`

### Example:
```bash
python categories_alignment.py --join_hemispheres --metric mutual_knn
```
"""

import pandas as pd
from tqdm import trange, tqdm
import numpy as np
from convergence.nsd import get_index, get_subject_roi, get_resource
from convergence.metrics import measure
import torch
from dmf.alerts import alert, send_alert
import argparse


def load_objects() -> dict[str, pd.DataFrame]:
    # Could be cleaner...
    df = get_resource("coco-objects")

    df_g_super = df.groupby(["nsd_id", "supercategory"]).size().unstack(fill_value=0).reset_index()
    df_g_cat = df.groupby(["nsd_id", "category"]).size().unstack(fill_value=0).reset_index()
    df_g_cat_onehot = df_g_cat.set_index("nsd_id").astype(bool).astype(int).reset_index()
    df_g_super_onehot = df_g_super.set_index("nsd_id").astype(bool).astype(int).reset_index()

    df_person = get_resource("coco-persons")
    df_person = df_person.drop(columns=["coco_id", "coco_split", "person_image_id", "num_keypoints", "area", "bbox", "segmentation"])
    df_person = df_person.set_index("nsd_id")
    df_person = df_person.notna().astype(int).reset_index()
    df_person = df_person.groupby("nsd_id").sum().reset_index()
    

    df_person # Add vectors with all 0s for nsids not present in 0-72999
    not_present = set(range(73000)) - set(df_person.nsd_id)
    df_person_not_present = pd.DataFrame({"nsd_id": list(not_present)})
    df_person = pd.concat([df_person_not_present, df_person]).fillna(0).astype(int).sort_values("nsd_id").reset_index(drop=True).set_index("nsd_id")
    df_person

    df_person_onehot = df_person.astype(bool).astype(int).reset_index()

    df_is_person = df_person.sum(axis=1).astype(bool).astype(int).reset_index().rename(columns={0: "is_person"})

    df_person = df_person.reset_index()

    df_g_super_no_person = df_g_super.drop(columns=["person"])
    df_g_cat_no_person = df_g_cat.drop(columns=["person"])

    subsets = {
        "category": df_g_cat,
        "supercategory": df_g_super,
        "category_onehot": df_g_cat_onehot,
        "supercategory_onehot": df_g_super_onehot,
        "person": df_person,
        "person_onehot": df_person_onehot,
        "is_person": df_is_person,
        "category_no_person": df_g_cat_no_person,
        "supercategory_no_person": df_g_super_no_person,
    }


    return subsets


def prepare_betas(
    betas,
    subject_index: list[int],
    q: float = 0.003,
) -> torch.tensor:

    betas = betas[subject_index].astype("float32")
    q0, q1 = np.quantile(betas, [q, 1 - q])
    betas = np.clip(betas, q0, q1)
    betas = (betas - q0) / (q1 - q0)

    betas = torch.tensor(betas, device="cuda", dtype=torch.float32)
    return betas


def prepare_objects(subsets, vector, nsd_ids):
    df_objects = subsets[vector]
    assert len(df_objects) == 73000
    df_objects = df_objects.sort_values(by="nsd_id").reset_index(drop=True)
    df_objects.set_index("nsd_id", inplace=True)

    # Nsds are unique, select a numpy matrix and convert to tensor based on passed list of nsd_ids
    matrix = df_objects.loc[nsd_ids].values
    matrix = torch.tensor(matrix, device="cuda", dtype=torch.float32)
    return matrix


@alert(input=["join_hemispheres", "metric"])
def compute_alignment(subsets, output_file, join_hemispheres=True, metric="unbiased_cka"):
    df_nsd = get_index("stimulus")

    results = []
    subsets_keys = list(subsets.keys())
    for subject in trange(1, 9, position=0, leave=False):
        send_alert(f"Processing subject {subject} - computing objects alignment")
        sessions = df_nsd.query(f"subject == {subject} and exists").session.unique()
        total_rois = 180 if join_hemispheres else 360
        for roi in trange(1, total_rois + 1, position=1, leave=False):
            roi_n = roi
            if join_hemispheres:
                roi = [roi, roi + 180]
            betas = get_subject_roi(subject, roi)

            for session in tqdm(sessions, position=2, leave=False):
                df_session = df_nsd.query(
                    f"subject == {subject} and session == {session} and exists"
                )
                nsd_ids = df_session.nsd_id.tolist()
                subject_indexes = df_session.subject_index.tolist()
                betas_session = prepare_betas(betas, subject_indexes)

                for vector in subsets_keys:
                    token_matrix = prepare_objects(subsets, vector, nsd_ids)
                    r = measure(metric, betas_session, token_matrix)
                    results.append(
                        {
                            "subject": subject,
                            "session": session,
                            "roi": roi_n,
                            "tokenizer": vector,
                            "score": r,
                            "metric": metric,
                        }
                    )
        #         break
        # break

    df_results = pd.DataFrame(results)
    df_results.to_parquet(output_file)



def parse_args():
    # Parse arguments (join_hemispheres)
    parser = argparse.ArgumentParser()

    parser.add_argument("--join_hemispheres", action="store_true")
    parser.add_argument(
        "--metric",
        type=str,
        default="unbiased_cka",
        choices=["unbiased_cka", "cka", "rsa", "mutual_knn"],
        help="Metric to use for alignment. Default is unbiased_cka.",
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    
    args = parse_args()
    join_hemispheres = args.join_hemispheres
    metric = args.metric

    if join_hemispheres:
        output_file = "objects_cka_similarity_joined.parquet"
    else:
        output_file = "objects_cka_similarity_separated.parquet"
    subsets = load_objects()
    compute_alignment(subsets, output_file, join_hemispheres=join_hemispheres, metric=metric)
