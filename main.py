from utils.landmark import find_medial_lateral_low_points_physical
from utils.segmenation import segment_bones_with_multistage_filter
from utils.expand_mask import expand_mask, randomized_expand_mask
from utils.utils import load_image
import matplotlib.pyplot as plt
import os
import json

os.makedirs("output", exist_ok=True)

results = []
landmark_points = {}

volume, spacing, affine, header = load_image("3702_left_knee.nii.gz")

orginal_mask = segment_bones_with_multistage_filter(volume, affine, header)
results.append(("original_mask", orginal_mask))

expanded_mask_2mm = expand_mask(
    original_mask=orginal_mask,
    voxel_spacing=spacing,
    affine=affine,
    header=header,
    output_mask_path="output/2mm_expanded.nii.gz",
    expansion_mm=2
)
results.append(("expanded_mask_2mm", expanded_mask_2mm))

expanded_mask_4mm = expand_mask(
    original_mask=orginal_mask,
    voxel_spacing=spacing,
    affine=affine,
    header=header,
    output_mask_path="output/4mm_expanded.nii.gz",
    expansion_mm=4
)
results.append(("expanded_mask_4mm", expanded_mask_4mm))

for i in range(3, 5):
    randomized_mask = randomized_expand_mask(
        original_mask=orginal_mask,
        spacing=spacing,
        affine=affine,
        header=header,
        output_mask_path=f"output/{i}_expanded.nii.gz",
        mm=2,
        random_fraction=0.8 * i,
        seed=5
    )
    results.append((f"expanded_mask_{i}", randomized_mask))

# Extract and save medial/lateral points
for name, mask in results:
    points = find_medial_lateral_low_points_physical(mask)
    landmark_points[name] = {
        "medial": list(points[0]),
        "lateral": list(points[1])
    }

# Save to JSON
with open("output/landmark_points.json", "w") as f:
    json.dump(landmark_points, f, indent=4)