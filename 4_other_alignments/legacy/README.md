# Legacy Alignment Scripts

This directory contains alignment scripts developed during the initial phase of the project. These scripts compute subject–subject and model–subject similarity using full-brain voxel-wise beta maps and apply ROI masks dynamically at runtime. While these scripts were later re-implemented for efficiency, they remain flexible and easy to modify.

In contrast to newer scripts in `3_alignment/` and `4_other_alignment/`, which rely on precomputed ROI-level betas for speed, these legacy versions load the full volumetric data. This design has several practical advantages:

* **Unified metric interface**: All scripts use a general `measure` function, allowing new similarity metrics (RSA, CKA, unbiased CKA, kNN, etc.) to be added with minimal changes.
* **Flexible masking**: Since masks are applied at runtime, testing a new ROI or parcellation only requires adding it to the mask list—no need to recompute or export betas.
* **Straightforward prototyping**: Because computations are done in Python loops over ROI or subject pairs, they are easier to debug, adapt, or extend for exploratory analyses.

The only reason these scripts were re-implemented was to support large-scale parallelization and high-throughput analyses using GPU acceleration. The newer scripts offer significant speed improvements (up to \~1000×) but follow a more rigid data pipeline.

---

## Summary of Scripts

| Script Name                                                                                                | Description                                                                              |
| ---------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| [`1_model_brain_alignment_legacy.py`](./1_model_brain_alignment_legacy.py)                                 | Computes model–brain similarity using voxel maps and runtime ROI masking.                |
| [`2_inter_subject_alignment_legacy.py`](./2_inter_subject_alignment_legacy.py)                             | Measures inter-subject alignment across brain ROIs using full beta volumes.              |
| [`3_cross_subject_common_shufled.py`](./3_cross_subject_common_shufled.py)                                 | Inter-subject alignment using same-task structure with shuffled stimuli identity.        |
| [`4_nsd_alignment_partitions_modalities.py`](./4_nsd_alignment_partitions_modalities.py)                   | Alignment within semantic partitions (e.g., person, motion, food) using runtime filters. |
| [`5_nsd_cka_cross_subject_permutations_edges.py`](./5_nsd_cka_cross_subject_permutations_edges.py)         | ROI-to-ROI inter-subject CKA with permutation testing on connectivity edges.             |
| [`6_nsd_cka_cross_subject_permutations.py`](./6_nsd_cka_cross_subject_permutations.py)                     | Permutation-based CKA alignment between subjects for each ROI.                           |
| [`7_nsd_cka_model_subject_permutations.py`](./7_nsd_cka_model_subject_permutations.py)                     | Model–subject CKA alignment with permutations across layers and ROIs.                    |
| [`8_nsd_extract_inter_subject_partitions.py`](./8_nsd_extract_inter_subject_partitions.py)                 | Extracts precomputed inter-subject similarities across defined image partitions.         |
| [`9_nsd_extract_subject_model_partitions.py`](./9_nsd_extract_subject_model_partitions.py)                 | Extracts subject–model similarities by semantic partition and layer.                     |
| [`10_partitions_cross_subject_similarities.py`](./10_partitions_cross_subject_similarities.py)             | Inter-subject alignment across semantic partitions with configurable metrics.            |
| [`11_partitions_subject_models_similarities.py`](./11_partitions_subject_models_similarities.py)           | Model–subject alignment by condition, layer, and region.                                 |
| [`12_rsa_other_dataset_intersubject_simplified.py`](./12_rsa_other_dataset_intersubject_simplified.py)     | Simplified inter-subject RSA implementation for BOLD5000 and THINGS datasets.            |
| [`13_subject_model_alignment_optimized_paritions.py`](./13_subject_model_alignment_optimized_paritions.py) | Optimized model–subject alignment over semantic partitions.                              |
| [`14_token_cka.py`](./14_token_cka.py)                                                                     | Computes alignment between brain activity and tokenized captions using CKA.              |
