import nibabel as nib
import numpy as np
import os
from scipy.ndimage import binary_closing,generate_binary_structure,binary_opening, binary_propagation, label, sum as nd_sum,gaussian_filter
import matplotlib.pyplot as plt
import cv2

def load_image(path):
    image=nib.load(path)
    volume=image.get_fdata()
    spacing = image.header.get_zooms()
    return volume, spacing
def gaussian_segment_bones(volume,sigma,threshold):
    smoothed = gaussian_filter(volume, sigma=sigma)
    mask = smoothed > threshold  # volume>275 threshold for cortical bone
    mask = binary_opening(mask, structure=np.ones((3,3,3))).astype(np.uint8)
    return mask
def segment_bones_with_bilateral_filter(ct_volume,
                                        diameter=9,
                                        sigma_col=35,
                                        sigma_sp=15,
                                        low_threshold=125,
                                        high_threshold=250,
                                        min_voxel_size=500):
    clipped = np.clip(ct_volume, -1000, 1500)  # Example HU range for bone CT
    volume_min, volume_max = np.min(clipped), np.max(clipped)
    norm_volume = ((clipped - volume_min) / (volume_max - volume_min)).astype(np.float32)
    volume_min, volume_max = np.min(ct_volume), np.max(ct_volume)
    norm_volume = ((ct_volume - volume_min) / (volume_max - volume_min)).astype(np.float32)
    scaled_volume = (norm_volume * 255).astype(np.uint8)
    filtered_volume = np.zeros_like(norm_volume, dtype=np.float32)
    for i in range(ct_volume.shape[2]):
        filtered_slice = cv2.bilateralFilter(scaled_volume[:, :, i], diameter, sigma_col, sigma_sp)
        filtered_volume[:, :, i] = filtered_slice.astype(np.float32) / 255.0
    filtered_hu = filtered_volume * (volume_max - volume_min) + volume_min
    strong_mask = filtered_hu > high_threshold
    weak_mask = (filtered_hu > low_threshold) & (filtered_hu <= high_threshold)
    structure = generate_binary_structure(3, 2)
    propagated_mask = binary_propagation(weak_mask, mask=strong_mask, structure=structure)
    full_mask = np.logical_or(strong_mask, propagated_mask)
    cleaned_mask = binary_opening(full_mask, structure=np.ones((3, 3, 3)))
    labeled_mask, num_labels = label(cleaned_mask)
    sizes = nd_sum(cleaned_mask, labeled_mask, index=np.arange(num_labels + 1))
    mask_cleaned = np.isin(labeled_mask, np.where(sizes > min_voxel_size)[0])

    return mask_cleaned.astype(np.uint8)

def segment_bones_with_multistage_filter(ct_volume, diameter=5, sigma_col=75, sigma_sp=75, hu_threshold=275):
    volume_min, volume_max = np.min(ct_volume), np.max(ct_volume)
    norm_volume = ((ct_volume - volume_min) / (volume_max - volume_min) * 255).astype(np.uint8)
    filtered_volume = np.zeros_like(norm_volume, dtype=np.uint8)
    for slice_index in range(norm_volume.shape[2]):
        norm_slice = norm_volume[:, :, slice_index]
        blurred_slice = cv2.GaussianBlur(norm_slice, (3, 3),1.2)
        filtered_slice = cv2.bilateralFilter(blurred_slice, diameter, sigma_col, sigma_sp)
        filtered_volume[:, :, slice_index] = filtered_slice
    filtered_hu = filtered_volume.astype(np.float32) / 255.0 * (volume_max - volume_min) + volume_min
    bone_mask = filtered_hu > hu_threshold
    cleaned_mask = binary_opening(bone_mask, structure=np.ones((3, 3, 3))).astype(np.uint8)
    return cleaned_mask
  
def visualize_scan_with_mask(ct_scan, bone_mask, output_filename):
    os.makedirs("output", exist_ok=True)
    save_path = f"output/{output_filename}"

    axial_idx = ct_scan.shape[2] // 2
    coronal_idx = ct_scan.shape[1] // 2
    sagittal_idx = ct_scan.shape[0] // 2

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    cmap_overlay = plt.cm.Reds
    alpha = 0.3

    # Axial view
    orig_axial = ct_scan[:, :, axial_idx]
    mask_axial = bone_mask[:, :, axial_idx]

    axes[0, 0].imshow(orig_axial, cmap='gray')
    axes[0, 0].set_title("Original - Axial View")

    axes[0, 1].imshow(orig_axial, cmap='gray')
    axes[0, 1].imshow(mask_axial, cmap=cmap_overlay, alpha=alpha)
    axes[0, 1].set_title("Overlay - Axial")

    axes[0, 2].imshow(mask_axial, cmap='Reds')
    axes[0, 2].set_title("Mask Only - Axial")

    # Coronal view
    orig_coronal = ct_scan[:, coronal_idx, :]
    mask_coronal = bone_mask[:, coronal_idx, :]

    axes[1, 0].imshow(orig_coronal, cmap='gray')
    axes[1, 0].set_title("Original - Coronal View")

    axes[1, 1].imshow(orig_coronal, cmap='gray')
    axes[1, 1].imshow(mask_coronal, cmap=cmap_overlay, alpha=alpha)
    axes[1, 1].set_title("Overlay - Coronal")

    axes[1, 2].imshow(mask_coronal, cmap='Reds')
    axes[1, 2].set_title("Mask Only - Coronal")

    # Sagittal view
    orig_sagittal = ct_scan[sagittal_idx, :, :]
    mask_sagittal = bone_mask[sagittal_idx, :, :]

    axes[2, 0].imshow(orig_sagittal, cmap='gray')
    axes[2, 0].set_title("Original - Sagittal View")

    axes[2, 1].imshow(orig_sagittal, cmap='gray')
    axes[2, 1].imshow(mask_sagittal, cmap=cmap_overlay, alpha=alpha)
    axes[2, 1].set_title("Overlay - Sagittal")

    axes[2, 2].imshow(mask_sagittal, cmap='Reds')
    axes[2, 2].set_title("Mask Only - Sagittal")

    # Turn off axes
    for row in axes:
        for ax in row:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    