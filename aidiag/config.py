"""Configuration file."""

# Files

IMAGES_FOLDER = "subset"

MASKS_FOLDER = "subset_masks"

MASKS_SUFFIX = "_effusion_first_reviewer.nii.gz"

# Image settings

# Depth of image and mask will be extended or truncated to this value during preprocessing:
IMAGE_Z_DEPTH = 128

# Values range of images for normalization:
MIN_VALUE = - 1024

MAX_VALUE = 3071
