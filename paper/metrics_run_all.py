#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 13:07:15 2025

@author: tng
"""
# This script fits a bunch of models to each metrics and 
# performs bayesian model comparison in each metric
# Important: uses pcntoolkit version 1.01 


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

# Suppress some annoying warnings and logs
pymc_logger = logging.getLogger("pymc")
pymc_logger.setLevel(logging.WARNING)
pymc_logger.propagate = False
warnings.simplefilter(action="ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'
pcntoolkit.util.output.Output.set_show_messages(False)

metrics = ["pearsonr", "spearmanr", "mutualinfo", "coherence", 
           "precision", "euclidean"]

for metric in metrics:
    
    save_dir = "out/out_" + metric + "_interh_fc/"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        os.mkdir(save_dir + "plots/")

    data_file = "Df_metricsFC_" + metric + ".json"
    data = pd.read_json(data_file)
    covariates = ["Age"]
    responses = ["Mean interh FC"]
    batch_effects = ["Dataset", "Atlas"]
    be = {"Dataset": data.Dataset.unique(),
          "Atlas": data.Atlas.unique()}
    data = data.dropna(subset=[covariates[0], responses[0], batch_effects[0]])
    n_sites = len(data["Dataset"].unique())
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

    plt.figure()
    sns.scatterplot(test_ndata.to_dataframe(), x=("X", "Age"), 
                    y=("Y", "Mean interh FC"), 
                    hue=("batch_effects", "Dataset"), 
                    style=("batch_effects", "Atlas"))
    plt.tight_layout()
    
    model_names = ["lin_rand_intcp_noise",
                   "lin_rand_intcp_noise_intcp_lin",
                   "lin_rand_intcp_noise_intcp_slope_lin",
                   "lin_rand_intcp_slope_noise",
                   "spline_rand_intcp_noise",
                   "spline_rand_intcp_slope_noise",
                   "shash_lin_rand_intcp_noise",
                   "shash_lin_rand_intcp_noise_delta",
                   "shash_lin_rand_intcp_noise_eps",
                   "shash_lin_rand_intcp_noise_intcp_lin",
                   "shash_lin_rand_intcp_slope_noise",
                   "shash_spline_rand_intcp_noise",
                   "shash_spline_rand_intcp_slope_noise",
                   "shash_spline_rand_intcp_slope_noise_eps"
                   ]
    

    likelihoods = [models.lik_lin_rand_intcp_noise(),
                   models.lik_lin_rand_intcp_noise_intcp_lin(),
                   models.lik_lin_rand_intcp_noise_intcp_slope_lin(),
                   models.lik_lin_rand_intcp_slope_noise(),
                   models.lik_spline_rand_intcp_noise(),
                   models.lik_spline_rand_intcp_slope_noise(),
                   models.lik_shash_lin_rand_intcp_noise(),
                   models.lik_shash_lin_rand_intcp_noise_delta(),
                   models.lik_shash_lin_rand_intcp_noise_eps(),
                   models.lik_shash_lin_rand_intcp_noise_intcp_lin(),
                   models.lik_shash_lin_rand_intcp_slope_noise(),
                   models.lik_shash_spline_rand_intcp_noise(),
                   models.lik_shash_spline_rand_intcp_slope_noise(),
                   models.lik_shash_spline_rand_intcp_slope_noise_eps()]


    

    for model_name, lik in zip(model_names, likelihoods):

        if not os.path.exists(save_dir + "out_" + model_name):
            model = make_model(lik, model_name, save_dir)
            test_data = model.fit_predict(train_ndata, test_ndata)
            test_data = model.predict(test_ndata)
            plot_fit_save(model_name, save_dir, test_data, be,
                          ylabel=responses[0] + " " + metric)
        else:
            continue
        
    
    # Model comparison, the higher elpd the better
    model_dirs = [save_dir + "out_" + mn for mn in model_names]
    models_dict = {n: d for n, d in zip(model_names, model_dirs)}
    comparisons = compare_hbr_models(models_dict)
    comparisons[responses[0]].to_csv(
        save_dir + metric + "_compared_interhFC.csv")

    comparisons_nowarning = comparisons[responses[0]][comparisons[responses[0]]
                                                          ["warning"] == False]
    az.plot_compare(comparisons_nowarning)
    plt.savefig(save_dir + "plots/" + metric +
                "_compare_all.png", bbox_inches="tight")
    cutoff = comp_nowarning["elpd_diff"] > 2*comp_nowarning["dse"]


