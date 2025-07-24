# Model Extraction Scripts

Scripts for extracting model activations at each transformer block (output layer).

Dataset paths should be configure, to be able to load the image paths using the `load_dataset` function (Complete the 1\_dataset\_preparation instructions).

## Files summary

| Script Path                                                           | Script Name                 | Short Description                                                                                                                                     |
| --------------------------------------------------------------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| [models.yml](./models.yml)                                            | Model registry              | YAML list of all vision & language models to extract.                                                                                                 |
| [1\_extract\_vision\_features.py](./1_extract_vision_features.py)     | Vision feature extractor    | Loads images, runs ViT-like models layer-by-layer, saves activations (`n_imgs × n_layers × d`). Supports NSD, BOLD5000, THINGS; `--untrained` option. |
| [3\_extract\_language\_features.py](./3_extract_language_features.py) | Language feature extractor  | Feeds captions to LLMs, stores per-layer activations. Accepts different caption files/prompts; outputs tensors analogous to vision script.            |
| [2\_caption\_dataset\_pixtral.py](./2_caption_dataset_pixtral.py)     | Automatic caption generator | Uses Pixtral-12B to caption datasets or answer custom prompts (scene/object, motion, etc.); outputs CSVs for later feature extraction.                |

## Scripts

### `models.yml`

[`models.yml`](./models.yml): List of vision and language models used for feature extraction.
Includes the models referenced in the paper (selected from Huh et al., 2024, *Platonic Representation Hypothesis*, ICLR 2024) and those compatible with an A40 (48GB) GPU.

### `1_extract_vision_features.py`

[`1_extract_vision_features.py`](./1_extract_vision_features.py): Extracts layer-wise activations from each transformer block of the vision models listed in `models.yml`.
Supports the Natural Scenes Dataset, BOLD5000, and THINGS.

```bash
# Extract features from the Natural Scenes Dataset
python 1_extract_vision_features.py --dataset nsd

# Extract features from BOLD5000
python 1_extract_vision_features.py --dataset bold5000

# Extract features from THINGS
python 1_extract_vision_features.py --dataset things
```

To extract activations from untrained backbones, add the `--untrained` flag.

Each model will produce a `.pt` file containing metadata and a tensor of shape
`n_images × n_layers × n_dim`. All ViTs have the same latent dimension across blocks.

### `2_caption_dataset_pixtral.py`

[`2_caption_dataset_pixtral.py`](./2_caption_dataset_pixtral.py): Generates automatic captions for images using the Pixtral 12B model.
This can be used to describe images or generate labels from prompts (e.g., whether the image is scene-dominated, object-dominated, etc.).

Default prompts include: `caption`, `motion`, `indoor`, `scene`. Custom prompts can be passed with the `--prompt` argument.

```bash
# Generate default captions for NSD
python 2_caption_dataset_pixtral.py --dataset nsd --prompt caption --output nsd_captions.csv

# Generate captions for THINGS
python 2_caption_dataset_pixtral.py --dataset things --prompt caption --output things_captions.csv

# Generate captions for BOLD5000
python 2_caption_dataset_pixtral.py --dataset bold5000 --prompt caption --output bold5000_captions.csv

# Generate scene-to-object gradient
python 2_caption_dataset_pixtral.py --dataset nsd --prompt scene --output nsd_scene_gradient.csv

# Use a custom prompt
python 2_caption_dataset_pixtral.py --dataset nsd --prompt "Describe this image" --output test.csv
```

Note: Captions should be manually reviewed to fix malformed or non-English prompts.
Final versions are stored in the `derivatives/captions` folder.

### `3_extract_language_features.py`

[`3_extract_language_features.py`](./3_extract_language_features.py): Extracts activations from language models processing image captions.
Captions can be generated with `2_caption_dataset_pixtral.py` or, for THINGS, taken from MS-COCO.

Each caption file must be a CSV with a `caption` or `captions` column.

```bash
# Extract features using MS-COCO captions
python 3_extract_language_features.py --prompt_name coco --dataset_name nsd --captions ../derivatives/captions/nsd-captions-coco.csv 

# Extract features using Pixtral-generated captions
python 3_extract_language_features.py --prompt_name pixtral --dataset_name nsd --captions ../derivatives/captions/nsd-captions-pixtral.csv

python 3_extract_language_features.py --prompt_name pixtral --dataset_name bold5000 --captions ../derivatives/captions/bold5000-captions-pixtral.csv

python 3_extract_language_features.py --prompt_name pixtral --dataset_name things --captions ../derivatives/captions/things-captions-pixtral.csv
```

Each run generates a `.pt` file with model metadata and a tensor of shape
`n_image_captions × n_layers × n_dim`.

## Generated Outputs

### Captions

Available in `derivatives/captions/`:

* `nsd-captions-coco.csv`
* `bold5000-captions-pixtral.csv`
* `nsd-captions-pixtral.csv`
* `things-captions-pixtral.csv`

### Feature Files

Due to their large size (6–12GB each), feature files are not included in the repository but can be regenerated easily.
They are required for scripts that compute alignment with brain data or other models.

Directory structure generated after running all scripts:

```
features/
├── nsd/
│   ├── all/
│   └── all-untrained/
├── bold5000/
│   └── all/
└── things/
    └── all/
```

Each subfolder contains `.pt` files with activations for each model-dataset combination.
