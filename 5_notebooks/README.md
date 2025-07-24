# Analysis Notebooks

This folder contains all notebooks required to generate the figures and analyses presented in the manuscript, as well as additional control analyses that were performed but excluded from the paper due to space constraints. They are included here for completeness.

The notebooks assume you have the precomputed alignment data between subjects, models, and brain regions (ROIs) stored in the `derivatives` folder. Provided you have these files correctly named and located, and the required libraries installed, all notebooks should run smoothly (tested prior to upload). Notebooks have been cleaned and extensively commented to aid interpretation.

Manuscript figures refers the numeration of arxiv preprint [https://arxiv.org/abs/2507.13941](https://arxiv.org/abs/2507.13941).

## Notebook Overview

| Script | Script Name | Manuscript Figures |
|-------------|-------------|--------------------|
| [01_rsa_spatial_alignment.ipynb](./01_rsa_spatial_alignment.ipynb) | Spatial distribution of alignment in fsaverage | Figs: 2A-G, D1.A-C |
| [02_rsa_hemisphere_comparison.ipynb](./02_rsa_hemisphere_comparison.ipynb) | Alignment comparison across hemispheres | Figs A1.A-D |
| [03_rsa_model_hierarchy.ipynb](./03_rsa_model_hierarchy.ipynb) | Model–brain alignment across hierarchical depth | Figs. 3A-F, D3.A-B, D4 |
| [04_rsa_connectivity.ipynb](./04_rsa_connectivity.ipynb) | Representational connectivity analysis | Figs. 4A, D5.C |
| [05_rsa_networks.ipynb](./05_rsa_networks.ipynb) | RSA networks: graphs of connectivity | Figs. 4B, A3, D5.D, D6, D7.B-C  |
| [06_rsa_flat_surfaces.ipynb](./06_rsa_flat_surfaces.ipynb) | Flat cortical surface visualisation (pycortex) | D2.A-C |
| [07_kmcca_projections.ipynb](./07_kmcca_projections.ipynb) | Kernel multi-view CCA projections and partial rsa | Figs. 5A-D, A6.A-C |
| [08_split_biology_rsa.ipynb](./08_split_biology_rsa.ipynb) | RSA alignment removing biological stimuli | Figs 5E-F, D7.A-C |
| [09_alignment_cka_vs_rsa.ipynb](./09_alignment_cka_vs_rsa.ipynb) | Comparison between RSA pearson, spearman and CKA alignment | A5.A-F |
| [10_bold5000_alignment.ipynb](./10_bold5000_alignment.ipynb) | Validation analysis using BOLD5000 dataset | Figs. 6A-D |
| [11_things_alignment.ipynb](./11_things_alignment.ipynb) | Validation analysis using THINGS dataset | Figs. 6E-H |
| [12_power_law_attenuation.ipynb](./12_power_law_attenuation.ipynb) | Power-law attenuation between RSA metrics | A7.A-D, A8 |
| [13_model_model_alignment.ipynb](./13_model_model_alignment.ipynb) | Model–model alignment comparison | Not included |
| [14_inter_vs_within_rsa.ipynb](./14_inter_vs_within_rsa.ipynb) | Inter- vs. within-subject RSA alignment | D5.A-B |
| [15_shifted_vs_unshifted_rsa.ipynb](./15_shifted_vs_unshifted_rsa.ipynb) | Shifted vs. unshifted inter-subject RSA comparison | A2.B-C |
| [15b_miniature_shifted_strategy.ipynb](./15b_miniature_shifted_strategy.ipynb) | Illustration of shifted RSA strategy (miniature) | A2.A |
| [16_tokenizer_analysis_supplementary.ipynb](./16_tokenizer_analysis_supplementary.ipynb) | Tokenizer-level control for language–brain RSA | A4.A-E |
| [17_untrained_models_similarity.ipynb](./17_untrained_models_similarity.ipynb) | Untrained models control (CKA) | Not included |

