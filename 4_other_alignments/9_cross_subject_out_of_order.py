"""
cross_subject_out_of_order.py

This script computes **cross-subject alignment** using betas from two different NSD subjects
that are matched by stimulus index (i.e., same trial structure), but who were exposed to
**different stimuli** in those trials. The aim is to assess how much representational alignment
arises from shared **task structure** alone, without shared visual input.

This acts as a **control analysis** to test alignment when stimulus identity is disrupted but
trial timing and session structure are preserved.

---

### Purpose
- Quantify representational alignment between subjects performing the same task structure
  but on non-overlapping images
- Measure how well aligned brain activity is when **stimuli differ**, but **indexing is preserved**
- Compare against a baseline where one subject's responses are randomly shuffled

---

### Procedure
1. Randomly choose a session (between 10-28) for each subject
2. Load betas for each subject in those sessions
3. Use **the same subject-wise trial indices** (e.g. stimulus index 0, 1, ..., 749) to align beta patterns
   across subjects, even though they refer to **different images**
4. For each hemisphere atlas (`rh.HCP_MMP1`, `lh.HCP_MMP1`):
    - For each ROI (1 to 180):
        - Extract voxel patterns per subject
        - Clip outliers (0.001-0.999 quantile)
        - Compute RSA between subjects
        - Shuffle voxels in subject 1 and recompute RSA
5. Store both the aligned and shuffled similarity scores

---

### Output Columns (Parquet File)
| Column        | Description                                       |
|---------------|---------------------------------------------------|
| `roi`         | ROI index (1-180)                                 |
| `rsa`         | RSA between subjects on matched indices           |
| `rsa_shuffled`| RSA after voxel shuffle in subject 1              |
| `subject1`    | Subject ID for first subject                      |
| `subject2`    | Subject ID for second subject                     |
| `session1`    | Session number used for subject 1                 |
| `session2`    | Session number used for subject 2                 |
| `atlas`       | Atlas used (`lh.HCP_MMP1` or `rh.HCP_MMP1`)       |

---

### Output
- File: `alignments_differnet_stimuli.parquet`
- Format: Parquet table of RSA comparisons across all subject pairs and ROIs
- Notification: Completion message sent via `send_message(...)`

"""
from convergence.nsd import load_betas, load_mask, get_resource
from convergence.metrics import measure
import torch
from tqdm import trange
import numpy as np
import pandas as pd
from dmf.alerts import send_message



alignments = []
for i in trange(1, 9, position=0):
    for j in trange(1, 9, position=1, leave=False):
        # Random number from 20 to 30
        session_1 = np.random.randint(10, 28)
        session_2 = np.random.randint(10, 28)
        if session_1 == session_2:
            session_2 = session_1 + 1

        betas1 = load_betas(subject=i, session=session_1)
        betas2 = load_betas(subject=j, session=session_2)

        altases = ["rh.HCP_MMP1", "lh.HCP_MMP1"]

        # random vector of 750 indexes to shuffle the betas
        shuffled_indexes = np.random.choice(750, 750, replace=False)

        for atlas in altases:
            mask1 = load_mask(subject=i, roi=atlas)
            mask2 = load_mask(subject=j, roi=atlas)
            for roi_idx in trange(1, 181, leave=False, position=2):

                masked_betas_1 = betas1[mask1 == roi_idx, :]
                masked_betas_2 = betas2[mask2 == roi_idx, :]

                # Clip values to 0.001 and 0.999 quantiles
                a, b = np.quantile(masked_betas_1.ravel(), [0.001, 0.999])
                masked_betas_1 = np.clip(masked_betas_1, a, b)

                a, b = np.quantile(masked_betas_2.ravel(), [0.001, 0.999])
                masked_betas_2 = np.clip(masked_betas_2, a, b)

                # As tensors to gpu
                masked_betas_1 = torch.tensor(masked_betas_1.T, device="cuda")
                masked_betas_2 = torch.tensor(masked_betas_2.T, device="cuda")
                r = measure("rsa", masked_betas_1, masked_betas_2)
                r_shuffled = measure(
                    "rsa", masked_betas_1[shuffled_indexes], masked_betas_2
                )
                # # cka
                # c = measure("unbiased_cka", masked_betas_1, masked_betas_2)
                # c_shuffled = measure(
                #     "unbiased_cka", masked_betas_shuffled, masked_betas_2
                # )
                alignments.append(
                    {
                        "roi": roi_idx,
                        "rsa": r,
                        "subject1": i,
                        "subject2": j,
                        "session1": session_1,
                        "session2": session_2,
                        # "cka": c,
                        "atlas": atlas,
                        "rsa_shuffled": r_shuffled,
                        # "cka_shuffled": c_shuffled,
                    }
                )
                # To cpu to clear memory
                masked_betas_1 = masked_betas_1.cpu()
                masked_betas_2 = masked_betas_2.cpu()
            torch.cuda.empty_cache()


df_alignments = pd.DataFrame(alignments)
df_alignments.to_parquet("alignments_differnet_stimuli.parquet", index=False)
send_message("Alignments computed")