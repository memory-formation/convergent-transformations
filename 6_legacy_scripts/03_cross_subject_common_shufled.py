"""
Legacy script to compute cross-subject RSA alignment on shared NSD stimuli
after **randomizing trial matching** across subjects (shuffling indexes).

For each subject pair and ROI, computes RSA scores across multiple permutations.
Used to estimate a **null distribution** of cross-subject alignment under 
non-systematic trial correspondence.

Results are saved per subject as individual Parquet files.
"""

from convergence.nsd import load_mask
from convergence.metrics import measure
import torch
from tqdm import trange, tqdm
import numpy as np
import pandas as pd
from dmf.alerts import alert, send_message
import nibabel as nib
from pathlib import Path


def get_commom_betas(subject):
    path = f"/mnt/tecla/Datasets/nsd/common_betas/subject_{subject}_common_betas.nii.gz"
    return nib.load(path).get_fdata()


def clip_roi(mask, p=0.01):
    a, b = np.quantile(mask.ravel(), [p, 1 - p])
    return np.clip(mask, a, b)

def get_masked_betas(betas, subject, atlas):
    masked_betas = {}
    mask = load_mask(subject=subject, roi=atlas)
    for roi_idx in range(1, 181):
        masked_betas[roi_idx] = clip_roi(betas[mask == roi_idx, :].T)
    return masked_betas


@alert
def compute_shuffled_similarity(
    output_path,
    n_permutations=10,
    atlases=["rh.HCP_MMP1", "lh.HCP_MMP1"],
):
    output_path = Path(output_path)
    alignments = []
    for i in trange(1, 9, position=0):
        send_message(f"Starting subject {i}")
        betas_i = get_commom_betas(i)
        m_i = betas_i.shape[-1]

        data_i = {}
        for atlas in atlases:
            data_i[atlas] = get_masked_betas(betas_i, i, atlas)
        del betas_i

        for j in trange(1, 9, position=1, leave=False):
            if j < i:
                continue
            
            betas_j = get_commom_betas(j)    
            m_j = betas_j.shape[-1]

            data_j = {}

            for atlas in atlases:
                data_j[atlas] = get_masked_betas(betas_j, j, atlas)
            del betas_j

            m = min(m_i, m_j)
            
            for n in trange(n_permutations, position=2, leave=False):

                # random vector of 750 indexes to shuffle the betas
                shuffle_i = np.random.choice(m_i, m, replace=False)
                shuffle_j = np.random.choice(m_j, m, replace=False)

                for atlas in tqdm(atlases, leave=False, position=3):
                    masked_betas_i = data_i[atlas]
                    masked_betas_j = data_j[atlas]

                    for roi_idx in trange(1, 181, leave=False, position=4):

                        roi_betas_i = masked_betas_i[roi_idx][shuffle_i, :]
                        roi_betas_j = masked_betas_j[roi_idx][shuffle_j, :]


                        # As tensors to gpu
                        with torch.no_grad():
                            roi_betas_i = torch.tensor(roi_betas_i, device="cuda")
                            roi_betas_j = torch.tensor(roi_betas_j, device="cuda")
                            value = measure("rsa", roi_betas_i, roi_betas_j)

                        alignments.append(
                            {
                                "roi": roi_idx,
                                "subject_i": i,
                                "subject_j": j,
                                "permutation": n,
                                "rsa": value,
                                "atlas": atlas,
                            }
                        )

                        roi_betas_i.cpu()
                        roi_betas_j.cpu()
                        del roi_betas_i, roi_betas_j
            

                    torch.cuda.empty_cache()
            
            del data_j
        df_alignments = pd.DataFrame(alignments)
        df_alignments.to_parquet(f"cross_alignments_shuffled_{i}.parquet", index=False)
        alignments = []
        del data_i

    #df_alignments = pd.DataFrame(alignments)
    #df_alignments.to_parquet(output_path, index=False)

    return output_path


if __name__ == "__main__":

    output_path = "alignments_common_stimuli_shuffled.parquet"

    compute_shuffled_similarity(
        output_path=output_path,
        n_permutations=10,
        atlases=["rh.HCP_MMP1", "lh.HCP_MMP1"],
    )
