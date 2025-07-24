# Alignment Scripts

This folder contains the scripts used to compute representational alignment across brain subjects and between subjects and deep models using Representational Similarity Analysis (RSA). These scripts are core to the main analyses presented in the paper.

Subject betas and model features must be precomputed and stored in the expected locations prior to running these scripts (scripts 1\_dataset\_preparation and 2\_model\_extraction)

## Script List

| Script Filename                                                                                                | Description                                                                    |
| -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| [1\_rsa\_nsd\_subject\_subject\_alignment.py](./1_rsa_nsd_subject_subject_alignment.py)                        | Compute inter-subject RSA across all NSD subjects (ROI x ROI).                 |
| [2\_rsa\_nsd\_subject\_model\_alignment.py](./2_rsa_nsd_subject_model_alignment.py)                            | Compute RSA between each NSD subject and deep model layers (ROI x Layer).      |
| [3\_rsa\_nsd\_subject\_subject\_alignment\_partitions.py](./3_rsa_nsd_subject_subject_alignment_partitions.py) | Inter-subject RSA within subsets of NSD trials (e.g. scenes with social cues). |
| [4\_rsa\_nsd\_subject\_subject\_alignment\_controled.py](./4_rsa_nsd_subject_subject_alignment_controled.py)   | Inter-subject RSA controlling for low-level or model features (partial RSA).   |
| [5\_rsa\_nsd\_subject\_model\_alignment\_controled.py](./5_rsa_nsd_subject_model_alignment_controled.py)       | Subject-model RSA controlling for confounds (partial RSA).                     |
| [6\_rsa\_nsd\_subject\_group\_subject\_alignment.py](./6_rsa_nsd_subject_group_subject_alignment.py)           | RSA between individual NSD subjects and the group-average RDM.                 |
| [7\_rsa\_other\_subject\_subject\_alignment.py](./7_rsa_other_subject_subject_alignment.py)                    | Inter-subject RSA for THINGS or BOLD5000 datasets (ROI x ROI).                 |
| [8\_rsa\_other\_subject\_model\_alignment.py](./8_rsa_other_subject_model_alignment.py)                        | Subject-to-model RSA for THINGS or BOLD5000 datasets (ROI x Layer).            |
