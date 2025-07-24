"""
extract_perceptual_statistics.py

This script computes alignment between early visual fMRI responses and **low-level perceptual
image statistics**,
such as luminance, edge density, texture, and Gabor energy. It is designed to evaluate to what
extent early-stage
brain regions (e.g., V1-V4) encode basic visual features.

---

### Purpose
To quantify the representational alignment between **fMRI betas** from early visual areas and handcrafted
**perceptual image features**, using CKA or RSA.

This analysis serves as a control for model alignment analyses, by assessing whether early ROI responses
can be accounted for by standard low-level statistics.

---

### Main Steps
1. **Load perceptual features**
   - Extracted separately and saved to `nsd-low-level-features.parquet` (e.g., edge density, texture,
     spatial stats).
   - Columns like `contrast`, `entropy`, etc., are dropped or standardized.

2. **Iterate over all subjects, ROIs, and sessions**
   - For each ROI and fMRI session:
     - Extract subject's beta responses.
     - Normalize and prepare corresponding perceptual feature vectors for the same NSD stimuli.
     - Compute alignment between beta responses and perceptual vectors.

3. **Run multiple metric comparisons**
   - Currently uses `unbiased_cka` but is extensible to `rsa`, etc.

---

### Output
A single Parquet file with alignment scores:
| Column     | Description                                           |
|------------|-------------------------------------------------------|
| `subject`  | Subject ID (1-8)                                      |
| `session`  | fMRI session number                                   |
| `roi`      | ROI index (1-180, hemispheres joined by default)      |
| `metric`   | Metric used (e.g. `unbiased_cka`, `rsa`)              |
| `score`    | Alignment score between ROI responses and features    |
| `perceptual` | Name of the perceptual feature(s) used              |

File saved as: `perceptual_alignment_joined.parquet`

---

### Notes
- Perceptual features must be precomputed and stored in the specified location (`nsd-low-level-features.parquet`).
- ROIs are joined across hemispheres by default (`roi` indexes are 1-180).
- Supports grouping features (e.g., `["edge_density", "homogeneity"]`) for multivariate comparisons.
- Perceptual features are Z-scored across stimuli before alignment.

---

### Example Perceptual Features (columns)
- `luminance`
- `saturation`
- `gabor_energy`
- `edge_density`
- `homogeneity`
- `spatial_frequency`

"""

import pandas as pd
import os
from pathlib import Path
from convergence.nsd import get_subject_roi, get_index
import torch
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from convergence.metrics import measure
from tqdm import tqdm, trange
from dmf.alerts import alert, send_alert


@alert
def compare_perceptual_metrics(df_perceptual, columns, metrics=["unbiased_cka"]):

    comparisons = []

    df_index = get_index("stimulus")

    for subject in trange(1, 9, position=0, leave=False):
        send_alert(f"Subject {subject}")
        sessions = df_index.query("subject==@subject and exists")["session"].unique()
        for roi in trange(1, 181, position=1, leave=False):

            betas = get_subject_roi(subject=subject, roi=[roi, roi + 180])
            for session in tqdm(sessions, leave=False, position=2):

                nsd_id, subject_index = (
                    df_index.query("session==@session and subject==@subject and exists")[
                        ["nsd_id", "subject_index"]
                    ]
                    .astype(int)
                    .values.T
                )
                betas_session = betas[subject_index]

                a, b = np.quantile(betas_session, [0.005, 0.995])
                betas_session = np.clip(betas_session, a, b).astype(float)
                betas_session = betas_session - betas_session.min()
                betas_session = betas_session / betas_session.max()
                betas_session = torch.tensor(betas_session, device="cuda")

                for column_group in columns:
                    if not isinstance(column_group, list):
                        column_group = [column_group]

                    perceptual_session = df_perceptual[column_group].values[nsd_id]
                    perceptual_session = torch.tensor(perceptual_session, device="cuda")
                    for metric in metrics:

                        r = measure(
                            metric_name=metric,
                            feats_A=perceptual_session,
                            feats_B=betas_session,
                        )
                        result = {
                            "roi": roi,
                            "session": session,
                            "subject": subject,
                            "metric": metric,
                            "score": r,
                            "perceptual": "_".join(column_group),
                        }
                        comparisons.append(result)

    df_results = pd.DataFrame(comparisons)
    df_results.to_parquet("perceptual_alignment_joined.parquet")
    return df_results


if __name__ == "__main__":
    results_folder = Path(os.getenv("CONVERGENCE_RESULTS"))

    # Load low-level vision perceptual features
    perceptual_filename = results_folder / "perceptual" / "nsd-low-level-features.parquet"
    df_perceptual = pd.read_parquet(perceptual_filename).drop(
        columns=["contrast", "k", "laplacian_var", "v", "entropy", "correlation", "h"]
    )
    values = StandardScaler().fit_transform(df_perceptual)
    df_perceptual = pd.DataFrame(values, columns=df_perceptual.columns, index=df_perceptual.index)

    columns = list(df_perceptual.columns)
    columns.append(["edge_density", "homogeneity"])

    compare_perceptual_metrics(df_perceptual, columns)
