import nibabel as nib
import numpy as np
import os
from scipy.ndimage import generate_binary_structure,binary_opening, binary_propagation, label, sum as nd_sum
import matplotlib.pyplot as plt
import cv2


def segment_bones_with_multistage_filter(ct_volume,
affine,
header,
output_mask_path="output/original_mask.nii.gz",
                                      diameter=2,
                                      sigma_col=5,
                                      sigma_sp=2,
                                      gaussian_sigma=3,  # Added parameter for Gaussian filter
                                      low_threshold=80,
                                      high_threshold=250,
                                      min_voxel_size=5000,
                                      ):
    #ct_volume, spacing,affine,header=load_image(orginal_path)
    clipped = np.clip(ct_volume, -1000, 1500)  
    volume_min, volume_max = np.min(clipped), np.max(clipped)
    norm_volume = ((clipped - volume_min) / (volume_max - volume_min)).astype(np.float32)
    volume_min, volume_max = np.min(ct_volume), np.max(ct_volume)
    norm_volume = ((ct_volume - volume_min) / (volume_max - volume_min)).astype(np.float32)
    scaled_volume = (norm_volume * 255).astype(np.uint8)
    gaussian_filtered = np.zeros_like(norm_volume, dtype=np.float32)
    for i in range(ct_volume.shape[2]):
        gaussian_filtered[:, :, i] = cv2.GaussianBlur(scaled_volume[:, :, i], 
                                                     (445, 445),
                                                     gaussian_sigma)
    filtered_volume = np.zeros_like(norm_volume, dtype=np.float32)
    for i in range(ct_volume.shape[2]):
        filtered_slice = cv2.bilateralFilter(gaussian_filtered[:, :, i], diameter, sigma_col, sigma_sp)
        filtered_volume[:, :, i] = filtered_slice.astype(np.float32) / 255.0
    filtered_hu = filtered_volume * (volume_max - volume_min) + volume_min
    strong_mask = filtered_hu > high_threshold
    weak_mask = (filtered_hu > low_threshold) & (filtered_hu <= high_threshold)
    structure = generate_binary_structure(3, 3)
    propagated_mask = binary_propagation(weak_mask, mask=strong_mask, structure=structure)
    full_mask = np.logical_or(strong_mask, propagated_mask)
    cleaned_mask = binary_opening(full_mask, structure=np.ones((3, 3, 3)))
    labeled_mask, num_labels = label(cleaned_mask)
    sizes = nd_sum(cleaned_mask, labeled_mask, index=np.arange(num_labels + 1))
    mask_cleaned = np.isin(labeled_mask, np.where(sizes > min_voxel_size)[0])
    mask_nii = nib.Nifti1Image(mask_cleaned.astype(np.uint8), affine, header)
    nib.save(mask_nii, output_mask_path)
    print(f"Mask saved to: {output_mask_path}")
    #breakpoint()
    return mask_cleaned.astype(np.uint8)
