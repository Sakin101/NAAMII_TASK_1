from utils.preprocess import load_image,segment_bones_with_bilateral_filter,visualize_scan_with_mask
import matplotlib.pyplot as plt
sigma=1.5
threshold=300
output_file=f"bilateral_{threshold}_sigma_{sigma}.png"
volume,spacing=load_image("/home/sachin/Desktop/Naamii_Task_1/3702_left_knee.nii.gz")
print(volume.shape)
#mask=segment_bones(volume=volume,sigma=sigma,threshold=threshold)
mask=segment_bones_with_bilateral_filter( volume, 
    diameter=5, 
    sigma_col=75, 
    sigma_sp=75, 
    hu_threshold=175)
print("Mask")
visualize_scan_with_mask(volume,mask,output_file)
#print(spacing.size)