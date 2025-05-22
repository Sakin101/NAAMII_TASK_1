from .utils import quick_mask_preview_with_points
import numpy as np
from scipy.ndimage import binary_erosion,label
from skimage.morphology import ball
def get_tibia(mask):
    z_mid = mask.shape[2] // 2
    lower_half = mask[:, :, z_mid:]
    labeled, num = label(lower_half)
    if num == 0:
        return mask  
    sizes = [(labeled == i).sum() for i in range(1, num + 1)]
    tibia_label = np.argmax(sizes) + 1
    tibia_mask = (labeled == tibia_label).astype(np.uint8)
    result = np.zeros_like(mask, dtype=np.uint8)
    result[:, :, z_mid:] = tibia_mask
    return result
def get_tibial_plateau(tibia_mask, plateau_height_ratio=0.999):
    tibia_coords = np.where(tibia_mask > 0)
    if len(tibia_coords[0]) == 0:
        return tibia_mask
    z_coords = tibia_coords[2]
    z_min, z_max = np.min(z_coords), np.max(z_coords)
    tibia_height = z_max - z_min
    plateau_thickness = int(tibia_height * plateau_height_ratio)
    plateau_z_min = z_max - plateau_thickness
    plateau_mask = tibia_mask.copy()
    plateau_mask[:, :, plateau_z_min:] = 0
    structure = ball(1)
    eroded = binary_erosion(plateau_mask, structure=structure)
    surface_mask = plateau_mask.astype(np.uint8) - eroded.astype(np.uint8)
    return surface_mask

def find_medial_lateral_low_points_physical(mask, spacing=(1.0, 1.0, 1.0), leg_side='right'):
    tibia=get_tibia(mask)
    surface_mask=get_tibial_plateau(tibia)
    coords_voxel = np.array(np.where(surface_mask > 0)).T  
    if coords_voxel.shape[0] == 0:
        print("No surface points found.")
        return None, None
    coords_physical = coords_voxel * np.array(spacing)
    x_coords = coords_physical[:, 0]
    x_mid = np.median(x_coords)
    medial_coords = coords_physical[x_coords < x_mid]
    lateral_coords = coords_physical[x_coords > x_mid]
    if leg_side.lower() == 'left':
        medial_coords, lateral_coords = lateral_coords, medial_coords
    medial_lowest = medial_coords[np.argmax(medial_coords[:, 1])]
    lateral_lowest = lateral_coords[np.argmax(lateral_coords[:, 1])]
    quick_mask_preview_with_points(surface_mask,"points.png",[medial_lowest,lateral_lowest],spacing)
    return tuple(medial_lowest), tuple(lateral_lowest)
