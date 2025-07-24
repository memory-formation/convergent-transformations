# Dataset(s) Preparation

This folder contains scripts and instructions to organize datasets used throughout the project.
It includes downloading fMRI betas, stimulus images, metadata, and generating trial and region indexes. Ensure that you have at least 2TB of free space to download and 
store the processed datasets.

## Files summary

| Script Path                                                                                | Script Name                     | Short Description                                                                                              |
| ------------------------------------------------------------------------------------------ | ------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| [1\_nsd\_check\_dataset.py](./1_nsd_check_dataset.py)                                      | NSD integrity check             | Verifies required NSD files (stimuli, betas, masks, metadata) are present and correctly located.               |
| [2\_nsd\_create\_indexes.py](./2_nsd_create_indexes.py)                                    | NSD index builder               | Unrolls stimuli/trials and creates image, trial, and beta indices used by downstream RSA scripts.              |
| [3\_create\_coco\_indexes.py](./3_create_coco_indexes.py)                                  | MS‑COCO index generator         | Downloads MS‑COCO captions/annotations and builds trial/content indices aligned with NSD images.               |
| [4\_nsd\_organize\_betas.py](./4_nsd_organize_betas.py)                                    | NSD beta organizer              | Repackages voxelwise betas into ROI‐wise NumPy matrices for fast loading (`n_trials × n_voxels`).              |
| [5\_things\_process\_dataset.ipynb](./5_things_process_dataset.ipynb)                      | THINGS preprocessing notebook   | Organizes THINGS fMRI betas into ROI matrices and prepares auxiliary metadata (captions, categories, indices). |
| [6\_bold5000\_process\_dataset.ipynb](./6_bold5000_process_dataset.ipynb)                  | BOLD5000 preprocessing notebook | Extracts ROI betas using subject‑specific HCP atlases and builds stimulus/metadata indices for BOLD5000.       |
| [bold5000\_hcp\_atlas/mni\_to\_functional.sh](./bold5000_hcp_atlas/mni_to_functional.sh)   | MNI→functional warp             | Warps the HCP‑MMP1 atlas from MNI space into each BOLD5000 subject’s functional space using FSL.               |


---

## Natural Scenes Dataset (NSD)

To prepare NSD for analysis:

1. Set the environment variable pointing to the NSD data folder:

```bash
export NSD_DATASET="/your/path/to/Datasets/nsd"
```

2. After downloading the dataset (follow download instructions below), 
run the script to check file presence:

```bash
python 1_nsd_check_dataset.py
```

It will validate the presence of:

* `nsd_stimuli.hdf5`: The image dataset
* `nsd_stim_info_merged.csv`: Trial-level metadata
* Subject-specific betas and masks (e.g., `betas_session01.nii.gz`, `HCP_MMP1.nii.gz`, etc.)

3. Process the dataset using the following scripts:

* [`2_nsd_create_indexes.py`](2_nsd_create_indexes.py): Unroll images and create trial/image/beta indexes.
* [`3_create_coco_indexes.py`](3_create_coco_indexes.py): Download MS-COCO and generate caption and content trials.
* [`4_nsd_organize_betas.py`](4_nsd_organize_betas.py): Reorganize betas into ROI-specific matrices.

---

### Final Folder Structure

After completing the setup and processing steps, your `$NSD_DATASET` should look like:

```
$NSD_DATASET/
├── subj01/
│   └── func1mm/
│       ├── betas_fithrf_GLMdenoise_RR/
│       │   ├── betas_session01.nii.gz
│       │   ├── ...
│       │   └── betas_session40.nii.gz
│       ├── roi/
│       │   ├── both.HCP_MMP1.nii.gz
│       │   ├── lh.HCP_MMP1.nii.gz
│       │   └── rh.HCP_MMP1.nii.gz
│       └── brainmask.nii.gz
├── subj02/
├── ...
├── subj08/
├── betas/
│   ├── sub01/
│   │   ├── sub-01_roi-001.npy
│   │   ├── ...
│   │   └── sub-01_roi-360.npy
│   ├── sub02/
│   ├── ...
│   └── sub08/
├── images/
│   ├── nsd_00000_00999/
│   ├── ...
│   └── nsd_72000_72999/
├── annotations/
│   ├── coco_captions.csv
│   ├── coco_objects_annotations.csv
│   └── coco_persons_annotations.csv
├── info/
│   └── hcp.csv
├── nsd.csv
├── nsd_masks.csv
├── nsd_stimuli.csv
├── nsd_stimuli.hdf5
└── nsd_stim_info_merged.csv
```

> ✅ You can skip beta processing and use the prepared index files in `derivatives/datasets/nsd`
> by setting `NSD_DATASET=/your/path/to/derivatives/datasets/nsd`, so you can run part of the
> pipeline that do not need to access the images or the fmri data directly.


### Downloading the NSD Dataset

We use the [Natural Scenes Dataset (NSD)](https://naturalscenesdataset.org/) to compare model and brain representations.

### Step 1: Request Access

1. Visit [naturalscenesdataset.org](https://naturalscenesdataset.org/) and sign the **NSD Data Access Agreement**.
2. You’ll receive a link to the Data Manual and instructions to download via AWS.

### Step 2: Download with AWS CLI

Install the [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) and ensure at least 1 TB of disk space.

#### 2.1 Download Beta Files (≈737 GB)

```bash
export NSD_DATASET="/path/to/local/folder/"

aws s3 sync s3://natural-scenes-dataset/nsddata_betas/ppdata $NSD_DATASET \
  --exclude "*" \
  --include "subj*/func1mm/betas_fithrf_GLMdenoise_RR/betas_session*.nii.gz" \
  --no-sign-request
```

#### 2.2 Download ROI and Brain Masks

```bash
aws s3 sync s3://natural-scenes-dataset/nsddata/ppdata/ $NSD_DATASET \
  --exclude "*" \
  --include "subj*/func1mm/roi/*HCP_MMP1.nii.gz" \
  --include "subj*/func1mm/brainmask.nii.gz" \
  --no-sign-request
```

#### 2.3 Download Stimuli and Trial Info

```bash
# Stimuli (37GB)
aws s3 cp s3://natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5 $NSD_DATASET \
  --no-sign-request

# Trial metadata
aws s3 cp s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_stim_info_merged.csv $NSD_DATASET \
  --no-sign-request
```

## THINGS-fMRI

To use the THINGS dataset, first define the dataset location:

```bash
export THINGS_DATASET="/path/to/your/Dataset/things"
```

### 1. Download fMRI Preprocessed Data

Download the **`betas_csv.zip`** file containing single-trial response data from the THINGS team’s Figshare repository:

📦 [Direct link to file](https://plus.figshare.com/articles/dataset/THINGS-data_fMRI_Single_Trial_Responses_table_format_/20492835?file=43635873)
📚 DOI: [10.25452/figshare.plus.c.6161151](https://doi.org/10.25452/figshare.plus.c.6161151)

Unzip the file inside your `$THINGS_DATASET` folder.

### 2. Process fMRI Data

Run the notebook:

```bash
things_process_dataset.ipynb
```

This notebook will:

* Organize beta values per ROI
* Create one `.npy` file per ROI of shape `[n_trials × n_voxels]`

Resulting files will be created in:

```
$THINGS_DATASET/preprocessed/sub-0X/betas_roi/betas_glasser_roi_Y.npy
```

Where `X ∈ [1, 4]` (subjects) and `Y ∈ [1, 360]` (Glasser ROIs).

### 3. Organize Metadata Files

Move the following files into the `preprocessed/` directory:

* `captions.csv` – Captions for each image
* `categories.csv` – Image categories
* `hcp.csv` – Atlas information
* `images.csv` – Image filenames and paths
* `models-info.csv` – Info on model responses (if applicable)
* `stimulus_index.csv` / `.parquet` – Mapping from subject trials to matrix rows

> ✅ If you do not need raw betas, you can skip downloading and set:
>
> ```bash
> export THINGS_DATASET="derivatives/datasets/things"
> ```
>
> All required files are provided in that folder.

### 4. Download the Stimuli Images

Download the full THINGS image set from the OSF repository:

🔗 [THINGS stimuli (OSF)](https://osf.io/jum2f/files/osfstorage)

Unzip the folder, and move it to:

```
$THINGS_DATASET/stimuli/THINGS/Images/
```

The final image paths should follow this format:

```
$THINGS_DATASET/stimuli/THINGS/Images/object_images_X_Z/<category>/<category>_XX.jpg
```

Example:

```
/path/to/things/stimuli/THINGS/Images/object_images_A-C/aardvark/aardvark_01b.jpg
```

---

### Final Folder Structure

Once all steps are complete, your THINGS dataset should look like:

```
$THINGS_DATASET/
├── preprocessed/
│   ├── captions.csv
│   ├── categories.csv
│   ├── hcp.csv
│   ├── images.csv
│   ├── models-info.csv
│   ├── stimulus_index.csv
│   ├── stimulus_index.parquet
│   ├── sub-01/
│   │   ├── betas_roi/
│   │   │   ├── betas_glasser_roi_1.npy
│   │   │   └── betas_glasser_roi_360.npy
│   │   ├── mask-glasser.nii.gz
│   │   ├── mask-glasser.npy
│   │   └── roi_coordinates.parquet
│   ├── sub-02/
│   └── sub-03/
└── stimuli/
    └── THINGS/
        └── Images/
            ├── object_images_A-C/
            ├── object_images_D-K/
            ├── object_images_L-Q/
            ├── object_images_R-S/
            └── object_images_T-Z/
```


## BOLD5000

To use the BOLD5000 dataset, first define the dataset path in your environment:

```bash
export BOLD_DATASET="/path/to/datasets/bold5000"
```

If you do **not** require raw betas or image stimuli, you can use only the preprocessed metadata and alignment indexes by pointing to the derivatives folder:

```bash
export BOLD_DATASET="derivatives/datasets/bold5000"
```

---

### 1. Download Stimuli

Download the stimuli images:

📦 [BOLD5000\_Stimuli.zip](https://www.dropbox.com/s/5ie18t4rjjvsl47/BOLD5000_Stimuli.zip?dl=1)

Unzip the contents into your `$BOLD_DATASET` folder.

---

### 2. Download fMRI Betas

Get the preprocessed beta data from the official Figshare repository (Release 2):

🔗 [BOLD5000 Release 2.0 – Figshare](https://figshare.com/articles/dataset/BOLD5000_Release_2_0/14456124)

---

### 3. Download Structural (T1) Files

For preprocessing and alignment, download the T1-weighted anatomical scans from:

🔗 [BOLD5000 Release 1.0 – OpenNeuro](https://openneuro.org/datasets/ds001499/versions/1.3.0)

---

### 4. Generate HCP ROIs in Subject Space

You’ll need to project the HCP-MMP1 atlas into the native functional space of each subject.

#### Step 1: Download the volumetric MNI atlas

🔗 [HCP-MMP1 projected on MNI2009a (NIfTI)](https://figshare.com/articles/dataset/HCP-MMP1_0_projected_on_MNI2009a_GM_volumetric_in_NIfTI_format/3501911)

#### Step 2: Run the alignment script

Use FSL to warp the MNI atlas into each subject’s native space:

```bash
bash bold5000_hcp_atlas/mni_to_functional.sh
```

This will generate one `.nii.gz` file per subject:

```
$BOLD_DATASET/preprocessed/atlas/
├── subj-01_HCP_atlas_func_360.nii.gz
├── subj-02_HCP_atlas_func_360.nii.gz
├── subj-03_HCP_atlas_func_360.nii.gz
└── subj-04_HCP_atlas_func_360.nii.gz
```

---

### 5. Process Dataset

Run the notebook:

```bash
bold5000_process_dataset.ipynb
```

This will use the subject-specific HCP atlases to extract beta values per ROI and store them as numpy matrices of shape `[n_trials × n_voxels]`.

---

### Final Folder Structure

After completing all steps, your folder should look like this:

```
$BOLD_DATASET/
├── bold5000_stimuli/
│   └── Scene_Stimuli/
│       └── Presented_Stimuli/
│           ├── COCO/
│           │   ├── COCO_train2014_000000000036.jpg
│           │   └── ...
│           ├── ImageNet/
│           │   ├── n01440764_10110.JPEG
│           │   └── ...
│           └── Scene/
│               ├── airplanecabin1.jpg
│               └── ...
├── preprocessed/
│   ├── atlas/
│   │   ├── subj-01_HCP_atlas_func_360.nii.gz
│   │   ├── subj-02_HCP_atlas_func_360.nii.gz
│   │   ├── subj-03_HCP_atlas_func_360.nii.gz
│   │   └── subj-04_HCP_atlas_func_360.nii.gz
│   ├── betas/
│   │   ├── sub-1/
│   │   │   ├── sub-1_hcpmmp_roi1_A.npy
│   │   │   ├── ...
│   │   │   └── sub-1_hcpmmp_roi360_D.npy
│   │   ├── sub-2/
│   │   ├── sub-3/
│   │   └── sub-4/
│   ├── captions.csv
│   ├── images.csv
│   ├── images.parquet
│   ├── stimulus_index.csv
│   └── stimulus_index.parquet
```
