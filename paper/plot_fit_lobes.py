#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 13:01:20 2026

@author: tng
"""

import models
from utils import *
from sklearn.model_selection import train_test_split
import os
import warnings
import logging

import pandas as pd
import matplotlib.pyplot as plt

from pcntoolkit import (
    HBR,
    NormData,
    load_fcon1000,
    plot_centiles,
    plot_qq,
    plot_ridge
)

import numpy as np
import pcntoolkit.util.output
import arviz as az
import seaborn as sns
sns.set_style("ticks")

import utils

# Suppress some annoying warnings and logs
pymc_logger = logging.getLogger("pymc")
pymc_logger.setLevel(logging.WARNING)
pymc_logger.propagate = False
warnings.simplefilter(action="ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'
pcntoolkit.util.output.Output.set_show_messages(False)


data = pd.read_json("Df_FC_lobes.json")
covariates = ["Age"]
batch_effects = ["Dataset", "Atlas"]
lobe_features = [col for col in data.columns if "Mean Intra" in col or "Mean Inter" in col]


fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(10, 15))

model_name = "shash_spline_rand_intcp_noise"
control_data = data[data['Dx'] == 'CONTROL']
case_data = data[data['Dx'] != 'CONTROL']
train_control, test_control = train_test_split(control_data,
                                               test_size=0.3,
                                               random_state=0,
                                               stratify=control_data["Name"])
be = {"Dataset": ["HCP", "HCA", "1000BRAINS", "CAMCAN", "ADNI", "ABIDE", "ADHD"],
      "Atlas": data.Atlas.unique()}
#fig, ax = plt.subplots(ncols=3, figsize=(15, 2.5))

for i in range(len(lobe_features)) :
    responses = [lobe_features[i]]
    test_ndata = NormData.from_dataframe(
     name="test_control", dataframe=test_control,
     covariates=covariates, batch_effects=batch_effects,
     response_vars=responses)
    save_dir = 'out/lobes/' + lobe_features[i] +'/'
    _ = utils.plot_fit(model_name, save_dir, test_ndata, be, ax=axes[i//3, i%3],
                    ylabel=responses[0])  
 
for ax in axes[:-1, :].flatten() :
    ax.set_xlabel("")

fig.subplots_adjust(left=0, right=1, wspace=0.3, hspace=0.3)

fig.savefig("out/plots/lobes_fit.png", bbox_inches="tight", dpi=300)
fig.savefig("out/plots/lobes_fit.svg", bbox_inches="tight", dpi=300)