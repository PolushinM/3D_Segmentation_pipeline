import os
from pathlib import Path

import SimpleITK as sitk
import nibabel as nib
import numpy as np

from .config import *


def load_dicom(directory: str) -> np.array:
    """Load dicom image from disk.
    (https://lightningseas.notion.site/Junior-CV-Engineer-5995a1401f3d4316a13051f7aa6a6d53#987dc626d5914b298e1e20fde140ce99)
            Parameters
            ----------
            directory : {str} directory, containing dicom image.

            Returns
            -------
            image : {np.ndarray} Array of shape D*W*H (dept, width, height).
            """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory)
    reader.SetFileNames(dicom_names)
    image_itk = reader.Execute()

    image_zyx = sitk.GetArrayFromImage(image_itk).astype(np.int16)
    return image_zyx


def load_3d_mask(directory: str) -> np.array:
    """Load 3d segmentation mask from disk.
    (https://lightningseas.notion.site/Junior-CV-Engineer-5995a1401f3d4316a13051f7aa6a6d53#987dc626d5914b298e1e20fde140ce99)
            Parameters
            ----------
            directory : {str} directory, containing mask.

            Returns
            -------
            mask : {np.ndarray} Array of shape D*W*H (dept, width, height).
            """
    mask = nib.load(directory)
    mask = mask.get_fdata().transpose(2, 0, 1)
    return mask


def normalize(image: np.ndarray, min_v: float, max_v: float) -> np.ndarray:
    """Normalize image to a range of values [0, 1].
            Parameters
            ----------
            image : {np.ndarray} original image.
            min_v : {float} minimum possible value in original image.
            max_v : {float} maximum possible value in original image.

            Returns
            -------
            image : {np.ndarray} Normalized image.
            """
    return (image - min_v) / (max_v - min_v)


def cut_and_pad_depth(image: np.ndarray, target_size: int, fill_value: float = 0.) -> np.ndarray:
    """Crop (cut) or add padding to image depth to bring images to the specified size.
            Parameters
            ----------
            image : {np.ndarray} original image - array of shape (D, W, H).
            target_size : {int} target size of cutted/padded image.
            fill_value : {float} fill value, default 0.0.

            Returns
            -------
            image : {np.ndarray} Cutted/padded image - array of shape (target_size, W, H).
            """
    if image.shape[0] < target_size:
        top_pad = (target_size - image.shape[0]) // 2
        bottom_pad = target_size - image.shape[0] - top_pad
        result = np.vstack(
            (np.ones((top_pad, image.shape[1], image.shape[2])) * fill_value,
             image,
             np.ones((bottom_pad, image.shape[1], image.shape[2])) * fill_value
             )
        )
    else:
        top_cut = (image.shape[0] - target_size) // 2
        result = image[top_cut + 1:target_size + top_cut + 1, :, :]
    return result


def load_sample(directory: str, name: str) -> tuple:
    """Load sample (image and mask) from the disk.
            Parameters
            ----------
            directory : {str} directory, containing dicom image and masks.
            name: {str} name of image to load.

            Returns
            -------
            sample : {tuple[np.ndarray, np.ndarray]} (image, mask)
                Returns chosen sample. Shape of both image and mask: D*W*H (dept, width, height).
            """
    dicom_path = str(next(Path(directory, IMAGES_FOLDER, name).rglob('*.json')).with_suffix(''))
    mask_path = os.path.join(directory, MASKS_FOLDER, name, ''.join([name, MASKS_SUFFIX]))
    image_zyx = normalize(load_dicom(dicom_path), min_v=MIN_VALUE, max_v=MAX_VALUE)
    image_zyx = cut_and_pad_depth(image_zyx, IMAGE_Z_DEPTH)
    image_zyx = np.moveaxis(image_zyx, 1, 2).copy()
    mask = load_3d_mask(mask_path)
    mask = cut_and_pad_depth(mask, IMAGE_Z_DEPTH)
    mask = np.flip(mask, 2).copy()
    return image_zyx, mask
