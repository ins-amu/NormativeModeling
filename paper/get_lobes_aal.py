#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 17:02:10 2026

@author: tng
"""

import numpy as np
import pandas as pd 
from nilearn import datasets

labels = datasets.fetch_atlas_aal().labels

lobe_keywords = {
    "Frontal": ["Frontal", "Precentral", "Rolandic", "Rectus", "Olfactory", "Supp_Motor"],
    "Parietal": ["Parietal", "Postcentral", "Precuneus", "Angular", "SupraMarginal", "Paracentral"],
    "Occipital": ["Occipital", "Calcarine", "Cuneus", "Lingual"],
    "Temporal": ["Temporal", "Heschl", "ParaHippocampal", "Fusiform"],
    "Limbic": ["Cingulate", "Hippocampus", "Amygdala", "Insula"],
    
}

#we excluded Caudate, Putamen, Pallidum, Thalamus from Limbic because Schaeffer doesn't have them
# we exclude also "Cerebellum": ["Cerebellum", "Vermis"] 

lobe_indices = {lobe: [] for lobe in lobe_keywords}

#assign each index 0 to 119 to a lobe
for index, region_name in enumerate(labels):
    for lobe, keywords in lobe_keywords.items():
        if any(keyword.lower() in region_name.lower() for keyword in keywords):
            lobe_indices[lobe].append(index)
            break
        
        
np.save("lobe_indices_aal.npy", lobe_indices)