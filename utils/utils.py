import nibabel as nib
import numpy as np
import os
from scipy.ndimage import binary_closing,generate_binary_structure,distance_transform_edt,binary_opening, binary_propagation, label, sum as nd_sum,gaussian_filter
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import binary_erosion
from skimage.morphology import ball
def load_image(path):
    image=nib.load(path)
    print(image.affine)
    volume=image.get_fdata()
    spacing = image.header.get_zooms()
    affine = image.affine
    header = image.header
    return volume, spacing,affine,header

def visualize_scan_with_mask(ct_scan, bone_mask, output_filename):
    os.makedirs("output", exist_ok=True)
    save_path = f"output/{output_filename}"
    axial_idx = ct_scan.shape[2] // 2
    coronal_idx = ct_scan.shape[1] // 2
    sagittal_idx = ct_scan.shape[0] // 2
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    cmap_overlay = plt.cm.Reds
    alpha = 0.3
    orig_axial = ct_scan[:, :, axial_idx]
    mask_axial = bone_mask[:, :, axial_idx]
    axes[0, 0].imshow(orig_axial, cmap='gray')
    axes[0, 0].set_title("Original - Axial View")
    axes[0, 1].imshow(orig_axial, cmap='gray')
    axes[0, 1].imshow(mask_axial, cmap=cmap_overlay, alpha=alpha)
    axes[0, 1].set_title("Overlay - Axial")
    axes[0, 2].imshow(mask_axial, cmap='Reds')
    axes[0, 2].set_title("Mask Only - Axial")
    orig_coronal = ct_scan[:, coronal_idx, :]
    mask_coronal = bone_mask[:, coronal_idx, :]
    axes[1, 0].imshow(orig_coronal, cmap='gray')
    axes[1, 0].set_title("Original - Coronal View")
    axes[1, 1].imshow(orig_coronal, cmap='gray')
    axes[1, 1].imshow(mask_coronal, cmap=cmap_overlay, alpha=alpha)
    axes[1, 1].set_title("Overlay - Coronal")
    axes[1, 2].imshow(mask_coronal, cmap='Reds')
    axes[1, 2].set_title("Mask Only - Coronal")
    orig_sagittal = ct_scan[sagittal_idx, :, :]
    mask_sagittal = bone_mask[sagittal_idx, :, :]
    axes[2, 0].imshow(orig_sagittal, cmap='gray')
    axes[2, 0].set_title("Original - Sagittal View")
    axes[2, 1].imshow(orig_sagittal, cmap='gray')
    axes[2, 1].imshow(mask_sagittal, cmap=cmap_overlay, alpha=alpha)
    axes[2, 1].set_title("Overlay - Sagittal")
    axes[2, 2].imshow(mask_sagittal, cmap='Reds')
    axes[2, 2].set_title("Mask Only - Sagittal")
    for row in axes:
        for ax in row:
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
def quick_mask_preview(mask,file):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    slices = [
        mask.shape[0]//2,  # Sagittal
        mask.shape[1]//2,  # Coronal
        mask.shape[2]//2   # Axial
    ]
    
    titles = ['Sagittal', 'Coronal', 'Axial']
    for ax, sl, title in zip(axes, slices, titles):
        if "Sagittal" in title:
            ax.imshow(mask[sl, :, :], cmap='gray')
        elif "Coronal" in title:
            ax.imshow(mask[:, sl, :], cmap='gray')
        else:
            ax.imshow(mask[:, :, sl], cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(file, bbox_inches='tight', dpi=300)
def quick_mask_preview_with_points(mask, file="Points.png", points=None, spacing=(1, 1, 1)):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    slices = [
        mask.shape[0] // 2,  # Sagittal
        mask.shape[1] // 2,  # Coronal
        mask.shape[2] // 2   # Axial
    ]

    titles = ['Sagittal', 'Coronal', 'Axial']

    # Convert physical coordinates to voxel indices if points are provided
    voxel_points = []
    if points is not None:
        for pt in points:
            voxel_pt = [int(round(pt[i] / spacing[i])) for i in range(3)]
            voxel_points.append(voxel_pt)

    for ax, sl, title in zip(axes, slices, titles):
        if title == "Sagittal":
            img = mask[sl, :, :]
            ax.imshow(img, cmap='gray')
            for pt in voxel_points:
                if pt[0] == sl:
                    ax.plot(pt[2], pt[1], 'ro')  # z (col), y (row)
        elif title == "Coronal":
            img = mask[:, sl, :]
            ax.imshow(img, cmap='gray')
            for pt in voxel_points:
                if pt[1] == sl:
                    ax.plot(pt[2], pt[0], 'bo')  # z (col), x (row)
        else:  # Axial
            img = mask[:, :, sl]
            ax.imshow(img, cmap='gray')
            for pt in voxel_points:
                if pt[2] == sl:
                    ax.plot(pt[1], pt[0], 'go')  # y (col), x (row)

        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(file, bbox_inches='tight', dpi=300)
    plt.close()