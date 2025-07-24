# Other Alignment Scripts

This folder contains **supplementary and control scripts** for computing various types of representational alignment. These analyses go beyond the main subject–subject and subject–model RSA comparisons, including:

* comparisons across **modalities** (language ↔ vision),
* **untrained** models,
* **semantic and perceptual** features,
* **object-level** alignment,
* and **control analyses** (e.g., mismatched stimulus identity).

All scripts are based on the [NSD dataset](http://naturalscenesdataset.org), and require that **subject beta responses** and **model features** have already been extracted and placed in their expected locations. Paths are assumed to be accessible via the `convergence` library interface.

---

## Scripts Summary

| Script Name                                                                                         | Description                                                                                    |
| --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| [1\_subject\_subject\_alignment\_other\_metrics.py](./1_subject_subject_alignment_other_metrics.py) | Compute subject–subject alignment using alternative metrics (e.g. CKA, mutual KNN).            |
| [2\_subject\_model\_alignment\_other\_metrics.py](./2_subject_model_alignment_other_metrics.py)     | Compute subject–model alignment with metrics beyond RSA.                                       |
| [3\_aligment\_language\_vision\_features.py](./3_aligment_language_vision_features.py)              | Measure alignment between language and vision model feature spaces.                            |
| [4\_categories\_alignment.py](./4_categories_alignment.py)                                          | Align brain activity to COCO object categories and presence of people.                         |
| [5\_extract\_tokenizer\_vocabulary.py](./5_extract_tokenizer_vocabulary.py)                         | Build token-level representations from captions using classic and LLM tokenizers.              |
| [6\_alignment\_perceptual\_statistics.py](./6_alignment_perceptual_statistics.py)                   | Align brain activity to low-level perceptual image statistics (e.g., edge density).            |
| [7\_untrained\_models\_alignment.py](./7_untrained_models_alignment.py)                             | Compute subject–model alignment using untrained models as a control.                           |
| [8\_extract\_object\_boxes.py](./8_extract_object_boxes.py)                                         | Use DETR to extract object bounding boxes and semantic categories from NSD images.             |
| [9\_cross\_subject\_out\_of\_order.py](./9_cross_subject_out_of_order.py)                           | Control analysis: cross-subject alignment with mismatched stimuli but matched trial structure. |

