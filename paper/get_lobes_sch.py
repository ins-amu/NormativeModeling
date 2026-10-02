#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 16:48:33 2026

@author: tng
"""

import numpy as np
from nilearn import datasets, image

# 1. Fetch the 3D maps of the Schaefer-100 and Talairach Lobe atlases
print("Downloading/Loading atlases...")
schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7)

# CORRECTED: Load the image from the file path string
schaefer_img = image.load_img(schaefer.maps) 
schaefer_data = schaefer_img.get_fdata()

# The Talairach atlas has a specific 'lobe' level we can use
talairach = datasets.fetch_atlas_talairach(level_name='lobe')

# CORRECTED: Load the Talairach image from the file path string
tal_img = image.load_img(talairach.maps)
tal_labels = talairach.labels

# 2. Resample the Talairach atlas to match the Schaefer resolution and space
tal_img_res = image.resample_to_img(tal_img, schaefer_img, interpolation='nearest')
tal_data_res = tal_img_res.get_fdata()

# 3. Initialize our dictionary to hold the matrix indices (0-99) for each lobe
lobe_indices = {
    "Frontal Lobe": [],
    "Parietal Lobe": [],
    "Temporal Lobe": [],
    "Occipital Lobe": [],
    "Limbic Lobe": [] # Captures Insula/Cingulate
}

# 4. Map each of the 100 Schaefer parcels to its dominant Lobe
# Schaefer regions in the Nifti image are numbered 1 to 100
for roi_val in range(1, 101):
    # Isolate the current Schaefer region
    roi_mask = (schaefer_data == roi_val)
    
    # Extract the Talairach lobe values that fall inside this Schaefer region
    overlapping_lobes = tal_data_res[roi_mask]
    
    # Filter out 0 (background space)
    valid_lobes = overlapping_lobes[overlapping_lobes > 0]
    
    if len(valid_lobes) > 0:
        # Find the most frequent lobe value (the mode) for this parcel
        most_frequent_val = int(np.bincount(valid_lobes.astype(int)).argmax())
        lobe_name = tal_labels[most_frequent_val]
        
        # Matrix indices are 0-indexed, so we subtract 1 from the roi_val
        matrix_index = roi_val - 1
        
        if lobe_name in lobe_indices:
            lobe_indices[lobe_name].append(matrix_index)

# 5. Print the results to verify how many regions fell into each lobe
print("\n--- Schaefer-100 Lobe Mapping ---")
for lobe, indices in lobe_indices.items():
    print(f"{lobe} ({len(indices)} regions): {indices}")
    
new_lobe_indices = {}
for l in lobe_indices.keys() :
    new_lobe_indices[l[:-5]] = lobe_indices[l].copy()
    
    
np.save("lobe_indices_sch.npy", new_lobe_indices)

np.load("lobe_indices_sch.npy", allow_pickle=True).item()