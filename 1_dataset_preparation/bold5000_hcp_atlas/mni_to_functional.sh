# Based on original script of E. Cámara

export LC_NUMERIC="en_US.UTF-8"

#NUM_CORES=$(nproc)
#export OMP_NUM_THREADS=$NUM_CORES

# Read the subject number from the command line argument
if [ $# -eq 0 ]; then
    echo "No subject number provided. Using default subject number: 01"
    exit 1
    #SUBJ="01"
else
    SUBJ=$1
fi

SUBJ_NAME="subj-${SUBJ}"

# Move to the directory where the script is located
SUBJECT_FOLDER="${SUBJ_NAME}"
cd "$(dirname "$0")/${SUBJECT_FOLDER}"
echo "Current working directory: $(pwd)"

REGISTRATION_FOLDER="Registration"
mkdir -p $REGISTRATION_FOLDER

TEMPLATE_FOLDER="../template"

# Files to provide
ref_T1_MNI_1mm_copy="${TEMPLATE_FOLDER}/MNI152_T1_1mm_brain.nii.gz"
HCP_ATLAS_ORIGINAL_copy="${TEMPLATE_FOLDER}/HCP_atlas_MNI152.nii.gz"
HCP_ATLAS_ORIGINAL_360_copy="${TEMPLATE_FOLDER}/HCP_atlas_MNI152_0_360.nii.gz"
T1="${SUBJ_NAME}_T1.nii.gz"
BOLD_MEAN="${SUBJ_NAME}_BOLD_mean.nii.gz"

# Check all exists in a loop
for file in "$ref_T1_MNI_1mm_copy" "$T1" "$BOLD_MEAN" "$HCP_ATLAS_ORIGINAL_copy" "$HCP_ATLAS_ORIGINAL_360_copy"; do
    if [ ! -f "$file" ]; then
        echo "Error: File not found: $file"
        exit 1
    fi
done

ref_T1_MNI_1mm="MNI152_T1_1mm_brain.nii.gz"
HCP_ATLAS_ORIGINAL="HCP_atlas_MNI152.nii.gz"
HCP_ATLAS_ORIGINAL_360="HCP_atlas_MNI152_0_360.nii.gz"

# Copy ref_T1_MNI_1mm_copy to ref_T1_MNI_1mm
if [ ! -f "$ref_T1_MNI_1mm" ]; then
    echo "Copying $ref_T1_MNI_1mm_copy to $ref_T1_MNI_1mm"
    cp "$ref_T1_MNI_1mm_copy" "$ref_T1_MNI_1mm"
else
    echo "File $ref_T1_MNI_1mm already exists, skipping copy."
fi

# Copy HCP_ATLAS_ORIGINAL_copy to HCP_ATLAS_ORIGINAL
if [ ! -f "$HCP_ATLAS_ORIGINAL" ]; then
    echo "Copying $HCP_ATLAS_ORIGINAL_copy to $HCP_ATLAS_ORIGINAL"
    cp "$HCP_ATLAS_ORIGINAL_copy" "$HCP_ATLAS_ORIGINAL"
else
    echo "File $HCP_ATLAS_ORIGINAL already exists, skipping copy."
fi

# Copy HCP_ATLAS_ORIGINAL_360_copy to HCP_ATLAS_ORIGINAL_360
if [ ! -f "$HCP_ATLAS_ORIGINAL_360" ]; then
    echo "Copying $HCP_ATLAS_ORIGINAL_360_copy to $HCP_ATLAS_ORIGINAL_360"
    cp "$HCP_ATLAS_ORIGINAL_360_copy" "$HCP_ATLAS_ORIGINAL_360"
else
    echo "File $HCP_ATLAS_ORIGINAL_360 already exists, skipping copy."
fi

# Derivative files - T1
T1_brain="${SUBJ_NAME}_T1_bet.nii.gz" # Output of T1 BET
T1_brain_reoriented="${SUBJ_NAME}_T1_bet_reoriented.nii.gz" # Output of T1 BET
T1_reoriented="${SUBJ_NAME}_T1_reoriented.nii.gz" # Output of T1
BOLD_MEAN_brain="${SUBJ_NAME}_BOLD_mean_bet.nii.gz"
ALIGNED_HCP_ATLAS_NAME="HCP_atlas_MNI152_aligned_to_template.nii.gz"
ATLAS_ALIGNMENT_MAT="${REGISTRATION_FOLDER}/hcp_atlas_to_mni_template.mat"
REGISTERED_HCP_ATLAS="${SUBJ_NAME}_HCP_atlas_func.nii.gz"

ALIGNED_HCP_ATLAS_360_NAME="HCP_atlas_MNI152_0_360_aligned_to_template.nii.gz"
ATLAS_ALIGNMENT_360_MAT="${REGISTRATION_FOLDER}/hcp_atlas_360_to_mni_template.mat"
REGISTERED_HCP_ATLAS_360_NAME="${SUBJ_NAME}_HCP_atlas_func_360.nii.gz"

# Brain extraction
EXTRACCION=0.25
EXTRACTION_BOLD=0.40

echo "1 Running BET on T1"
bet $T1 $T1_brain -R -f $EXTRACCION -g 0 -c 87 127 158

echo "2 Running BET on BOLD MEAN"
bet $BOLD_MEAN $BOLD_MEAN_brain -R -f $EXTRACTION_BOLD -g 0 -c 36 52 37

echo "3 Reorient T1 to standard"
fslreorient2std $T1_brain $T1_brain_reoriented
fslreorient2std $T1 $T1_reoriented

echo "4 FLIRTing diff to str"
flirt -in $BOLD_MEAN_brain -ref $T1_brain_reoriented -out "${REGISTRATION_FOLDER}/diff2str" -omat "${REGISTRATION_FOLDER}/func2str.mat"

echo "5 Inverting the matrix"
convert_xfm -omat "${REGISTRATION_FOLDER}/str2func.mat" -inverse "${REGISTRATION_FOLDER}/func2str.mat"

echo "6 FLIRTing str to std"
flirt -in $T1_brain -ref $ref_T1_MNI_1mm -out "${REGISTRATION_FOLDER}/str2std" -omat "${REGISTRATION_FOLDER}/str2std.mat"

echo "7 Inverting the matrix"
convert_xfm -omat "${REGISTRATION_FOLDER}/std2str.mat" -inverse "${REGISTRATION_FOLDER}/str2std.mat"

echo "8 fnirting str to std" #For FNIRT it´s better to use an image together with the skull 
fnirt --in=$T1_reoriented --ref=${ref_T1_MNI_1mm} --aff="${REGISTRATION_FOLDER}/str2std.mat" --iout="${REGISTRATION_FOLDER}/iout_str2std" --cout="${REGISTRATION_FOLDER}/cout_str2std"

echo "9 fnirting std to str"
fnirt --in=${ref_T1_MNI_1mm} --ref=$T1_reoriented --aff="${REGISTRATION_FOLDER}/std2str.mat" --iout="${REGISTRATION_FOLDER}/iout_std2str" --cout="${REGISTRATION_FOLDER}/cout_std2str"

echo "10 fnirting str to func" 
fnirt --in=$T1_reoriented --ref=$BOLD_MEAN_brain --aff="${REGISTRATION_FOLDER}/str2func.mat" --iout="${REGISTRATION_FOLDER}/iout_str2func" --cout="${REGISTRATION_FOLDER}/cout_str2func"

echo "11 concatenate transforms"
convertwarp --ref=$BOLD_MEAN_brain -w "${REGISTRATION_FOLDER}/cout_std2str.nii.gz" --postmat="${REGISTRATION_FOLDER}/str2func.mat" --out="${REGISTRATION_FOLDER}/standard2func"

echo "12 invert the transform"
invwarp -w "${REGISTRATION_FOLDER}/standard2func" -o "${REGISTRATION_FOLDER}/func2standard" -r ${ref_T1_MNI_1mm}

echo "13 applying warp from std to diff space"
applywarp -i ${ref_T1_MNI_1mm} -o standard2func_ima -r $BOLD_MEAN_brain -w "${REGISTRATION_FOLDER}/standard2func"

echo "14 Aligning HCP atlas to the MNI template used in the pipeline"
flirt -in "${HCP_ATLAS_ORIGINAL}" \
      -ref "${TEMPLATE_FOLDER}/MNI152_T1_1mm_brain.nii.gz" \
      -out "${ALIGNED_HCP_ATLAS_NAME}" \
      -omat "${ATLAS_ALIGNMENT_MAT}" \
      -interp nearestneighbour \
      -nosearch

echo "15 Transforming the aligned full HCP atlas to functional space"
applywarp -i "${ALIGNED_HCP_ATLAS_NAME}" \
          -o "${REGISTERED_HCP_ATLAS}" \
          -r "$BOLD_MEAN_brain" \
          -w "${REGISTRATION_FOLDER}/standard2func" \
          --interp="nn"

echo "16 Aligning HCP atlas 360 to the MNI template used in the pipeline"
flirt -in "${HCP_ATLAS_ORIGINAL_360}" \
      -ref "${TEMPLATE_FOLDER}/MNI152_T1_1mm_brain.nii.gz" \
      -out "${ALIGNED_HCP_ATLAS_360_NAME}" \
      -omat "${ATLAS_ALIGNMENT_360_MAT}" \
      -interp nearestneighbour \
      -nosearch

echo "17 Transforming the aligned full HCP atlas 360 to functional space"
applywarp -i "${ALIGNED_HCP_ATLAS_360_NAME}" \
          -o "${REGISTERED_HCP_ATLAS_360_NAME}" \
          -r "$BOLD_MEAN_brain" \
          -w "${REGISTRATION_FOLDER}/standard2func" \
          --interp="nn"

echo "OUTPUT FILES: ${REGISTERED_HCP_ATLAS} ${REGISTERED_HCP_ATLAS_360_NAME}"