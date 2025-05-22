import nibabel as nib
import numpy as np
import os
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import binary_erosion
from skimage.morphology import ball
def expand_mask(original_mask, voxel_spacing, affine,header,output_mask_path,expansion_mm):
    distance = distance_transform_edt(original_mask == 0, sampling=voxel_spacing)
    expanded_mask = (distance <= expansion_mm)
    mask_nii = nib.Nifti1Image(expanded_mask.astype(np.uint8), affine, header)
    nib.save(mask_nii, output_mask_path)
    print(f"Mask saved to: {output_mask_path}")
    return expanded_mask.astype(np.uint8)

def randomized_expand_mask(original_mask, spacing,affine,header, output_mask_path,mm=2.0, random_fraction=0.5, seed=5):
    if seed is not None:
        np.random.seed(seed)
    dist_map = distance_transform_edt(original_mask == 0, sampling=spacing)
    band_region = (dist_map > 0) & (dist_map <= mm)
    random_selection = (np.random.rand(*original_mask.shape) < random_fraction)
    band_mask = band_region & random_selection
    randomized_mask = original_mask | band_mask
    mask_nii = nib.Nifti1Image(randomized_mask.astype(np.uint8), affine, header)
    nib.save(mask_nii, output_mask_path)
    print(f"Mask saved to: {output_mask_path}")
    return randomized_mask.astype(np.uint8)
