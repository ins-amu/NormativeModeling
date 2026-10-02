#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 14:16:03 2026

@author: tng
"""

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

# Suppress some annoying warnings and logs
pymc_logger = logging.getLogger("pymc")
pymc_logger.setLevel(logging.WARNING)
pymc_logger.propagate = False
warnings.simplefilter(action="ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'
pcntoolkit.util.output.Output.set_show_messages(False)

from sklearn.model_selection import train_test_split
from utils import *
import models


data_df = pd.read_json("Df_FC_lobes.json")
lobe_features = [col for col in data_df.columns if "Mean Intra" in col or "Mean Inter" in col]
covariates = ["Age"]
batch_effects = ["Dataset", "Atlas"]

for feature in lobe_features :
    
    save_dir = "out/lobes/" + feature + "/"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        os.mkdir(save_dir + "/plots/")
        
    responses = [feature]
    data = data_df.copy().dropna(subset=[covariates[0], responses[0], batch_effects[0]])
    n_sites = len(data["Name"].unique())
    be = {"Dataset": data.Dataset.unique(),
          "Atlas": data.Atlas.unique()}
    
    control_data = data[data['Dx'] == 'CONTROL']
    case_data = data[data['Dx'] != 'CONTROL']
    train_control, test_control = train_test_split(control_data, 
                                               test_size=0.3, 
                                               random_state=0, 
                                               stratify=control_data["Name"])
    train_ndata = NormData.from_dataframe(
        name="train_control", dataframe=train_control, 
        covariates=covariates, batch_effects=batch_effects, 
        response_vars=responses)
    test_ndata = NormData.from_dataframe(
        name="test_control", dataframe=test_control, 
        covariates=covariates, batch_effects=batch_effects, 
        response_vars=responses)

    sns.scatterplot(train_ndata.to_dataframe(), x=("X", "Age"), y=('Y', responses[0]))
    plt.show()
    
    
    model_name = "shash_spline_rand_intcp_noise"
    likelihood = models.lik_shash_spline_rand_intcp_noise()
    
    if not os.path.exists(save_dir + "out_" + model_name) :
        model = make_model(likelihood, model_name, save_dir)
        test_data = model.fit_predict(train_ndata, test_ndata)
        test_data = model.predict(test_ndata)
        plot_fit_save(model_name, save_dir, test_data, be, 
                      ylabel=responses[0])
    else:
        continue
