
# Move to the directory where the script is located
cd "$(dirname "$0")"

ATLAS_FILE='template/HCP_atlas_MNI152.nii.gz'
ROIS_FOLDER='rois'

mkdir -p $ROIS_FOLDER

# Loop from 1 to 180 (both included)
for i in $(seq 1 180); do
    # Create the ROI file name. Save as HCP_atlas_MNI152_ROI_001.nii.gz
    # The number is padded with leading zeros to 3 digits
    ROI_FILE="${ROIS_FOLDER}/HCP_atlas_MNI152_ROI_$(printf "%03d" $i).nii.gz"
    

    # Low threshold is i-0.1, upper threshold is i+0.1
    LOW_THRESHOLD=$(echo "$i - 0.1" | bc)
    HIGH_THRESHOLD=$(echo "$i + 0.1" | bc)

    echo "Extracting ROI $i ($LOW_THRESHOLD - $HIGH_THRESHOLD) to $ROI_FILE"
    # Extract the ROI using fslmaths
    # fslmaths 'atlas_1mm.nii.gz' -uthr 179 -thr 178 -bin ROI_178.nii.gz
    fslmaths $ATLAS_FILE -uthr $HIGH_THRESHOLD -thr $LOW_THRESHOLD -bin $ROI_FILE
done
