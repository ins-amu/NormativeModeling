import os 
import numpy as np
import pickle
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import pcntoolkit as ptk  

import logging
logger = logging.getLogger("pymc")
logger.propagate = False

def hbr_fit(df, 
           x_str, 
           y_str,
           save_fit = True,
           fit_suffix_str='',
           n_samples = 1000,
           n_warmup = 1000,
           n_chains = 2,
           n_cores = 8,
           target_accept = 0.99,
           likelihood = 'Normal', 
           model_type = 'linear',
           linear_mu = 'True',
           linear_sigma = 'False',
           linear_epsilon = 'False',
           linear_delta = 'False',
           random_intercept_mu = 'False',
           random_slope_mu = 'False',
           random_intercept_sigma = 'False',
           random_slope_sigma = 'False',
           random_intercept_epsilon = 'False',
           random_slope_epsilon = 'False',
           random_intercept_delta = 'False',
           random_slope_delta = 'False') :
    
    #save x and y in files 
    data_path = os.path.join(os.getcwd(), 'reg_data/')
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
        
    xfile = os.path.join(data_path, x_str + '.pkl')
    yfile = os.path.join(data_path, y_str + '.pkl')
    
    with open(xfile, 'wb') as file:
        pickle.dump(pd.DataFrame(df[x_str]), file)
    with open(yfile, 'wb') as file:
        pickle.dump(pd.DataFrame(df[y_str]), file)
    
    output_path = os.path.join(os.getcwd(), 'reg_fits/') # output path, where the models and inference will be written
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    
    
    #for faster computation:
    inscaler_type='standardize'
    outscaler_type='standardize'
    inscaler = ptk.util.utils.scaler(inscaler_type)
    outscaler = ptk.util.utils.scaler(outscaler_type)

    alg='hbr'
    binary='True'

    out = ptk.normative.fit(covfile=xfile,
                      respfile=yfile,
                      saveoutput=False, 
                      savemodel=False,
                      binary=binary,
                      alg=alg,
                      n_samples=n_samples,
                      n_tuning=n_warmup,
                      n_chains=n_chains,
                      cores=n_cores,
                      target_accept=target_accept,
                      inscaler=inscaler_type,
                      outscaler=outscaler_type,

                      likelihood=likelihood,
                      model_type=model_type,

                      linear_mu = linear_mu,
                      linear_sigma = linear_sigma,
                      linear_epsilon = linear_epsilon,
                      linear_delta = linear_delta,

                      random_intercept_mu = random_intercept_mu,
                      random_slope_mu = random_slope_mu,
                      random_intercept_sigma = random_intercept_sigma,
                      random_slope_sigma = random_slope_sigma,
                      random_intercept_epsilon = random_intercept_epsilon,
                      random_slope_epsilon = random_slope_epsilon,
                      random_intercept_delta = random_intercept_delta,
                      random_slope_delta = random_slope_delta
                     )
    if save_fit :
        output_name = 'x=' + x_str + '_y=' + y_str + '_fit_' + fit_suffix_str
        with open(output_path + output_name + '.pkl', 'wb') as file:
            pickle.dump(out, file)
        
    return out, inscaler, outscaler  


def get_quantiles(nm, df, x_str, y_str, inscaler, outscaler, n_qs=2, n_samples=500) :
    
    x_standardized = inscaler.fit_transform(df[x_str])
    y_standardized = outscaler.fit_transform(df[y_str])
    minx = np.min(x_standardized)
    maxx = np.max(x_standardized)
    zscores = np.arange(-n_qs, n_qs+1)[:,np.newaxis]
    synthetic_x = np.linspace(minx, maxx, n_samples)[:,np.newaxis]
    q = nm.get_mcmc_quantiles(synthetic_x, z_scores=zscores)
    
    return q


def plot_quantiles(nm, df, x_str, y_str, inscaler, outscaler, n_qs=2, n_samples=500, ax=None) :
    
    x_standardized = inscaler.fit_transform(df[x_str])
    y_standardized = outscaler.fit_transform(df[y_str])
    minx = np.min(x_standardized)
    maxx = np.max(x_standardized)
    zscores = np.arange(-n_qs, n_qs+1)[:,np.newaxis]
    synthetic_x = np.linspace(minx, maxx, n_samples)[:,np.newaxis]
    q = nm.get_mcmc_quantiles(synthetic_x, z_scores=zscores)
    
    if ax is None :
        fig, ax = plt.subplots()
        
    ax.scatter(df[x_str], df[y_str], s=5, alpha=0.8, color='royalblue')
    for i, v in enumerate(zscores):
        thickness = 1.5
        linestyle = "-"
        if v == 0:
            thickness = 4
        if abs(v) > 1:
            linestyle = "--"
        ax.plot(inscaler.inverse_transform(synthetic_x), 
                 outscaler.inverse_transform(q[i]), 
                 linewidth = thickness, 
                 linestyle = linestyle, 
                 color = 'black', 
                 alpha = 0.7)
        ax.set(xlabel=x_str, ylabel=y_str)
    return ax


def gaussian_pdf(y, mu, sigma) :
    return norm(mu, sigma).pdf(y)


def shasho2_pdf(y, mu, sigma, epsilon, delta) :
    z = (y - mu) / (sigma * delta)
    r = np.sinh(delta * np.arcsinh(z) - epsilon)
    c = np.cosh(delta * np.arcsinh(z) - epsilon)
    prob = (c / sigma) * (delta / np.sqrt(2 * np.pi)) * (1 / np.sqrt(1 + z**2)) - np.exp(-r**2 / 2)
    return prob


def shash_pdf(y, mu, sigma, epsilon, delta) :
    z = (y - mu) /  sigma
    r = 0.5 * (np.exp(delta * np.arcsinh(z)) - np.exp(-epsilon * np.arcsinh(z)))
    c = 0.5 * (delta * np.exp(delta * np.arcsinh(z)) + epsilon * np.exp(-epsilon * np.arcsinh(z)))
    prob = np.exp(-r**2 / 2) * c / (np.sqrt(2 * np.pi) * sigma * np.sqrt((1 + z**2)))
    return prob


def shasho_pdf(y, mu, sigma, epsilon, delta) :
    z = (y - mu) /  sigma
    r = np.sinh(delta * np.arcsinh(z) - epsilon)
    c = np.cosh(delta * np.arcsinh(z) - epsilon)
    prob = np.exp(-r**2 / 2) * delta * c / (np.sqrt(2 * np.pi) * sigma * np.sqrt((1 + z**2)))
    return prob


def compute_loglikelihood(nm, samples, data) :
    fun = None
    if nm.configs['likelihood'] == 'Normal' :
        lik = gaussian_pdf(data.to_numpy(),
                           samples['mu_samples'], 
                           samples['sigma_plus_samples'])
        return np.log(lik)
    #for now, all return shasho_pdf it's the only stable one
    elif nm.configs['likelihood'] == 'SHASH' :
        fun = shasho_pdf
    elif nm.configs['likelihood'] == 'SHASHo' :
        fun = shasho_pdf
    elif nm.configs['likelihood'] == 'SHASHo2' :
        fun = shasho_pdf
    else :
        print('likelihood not defined')
        return 0
    lik = fun(data.to_numpy(), 
              samples['mu_samples'], 
              samples['sigma_plus_samples'], 
              samples['epsilon_samples'], 
              samples['delta_plus_samples'])
    return np.log(lik)


def get_transformed_param(intercept, slope, df, x_str, nm, inscaler, outscaler) :
    
    x_standardized = inscaler.fit_transform(df[x_str]).to_numpy()
    if nm.hbr.bsp is not None :
        x_standardized_spline = ptk.model.hbr.bspline_transform(x_standardized[:, np.newaxis], nm.hbr.bsp)
        phi = np.concatenate((x_standardized[:, np.newaxis], x_standardized_spline), axis=1)
    else :
        phi = x_standardized[:, np.newaxis]
    transformed_slope = np.sum(phi * np.array(slope), axis=1)
    
    return outscaler.inverse_transform(intercept + transformed_slope)


def transform_positive(transformed_param, param_name, nm) :
    if nm.configs['likelihood'] == 'Normal' :
        return log(1 + np.exp(transformed_param / 3)) * 3
    elif nm.configs['likelihood'][:5] == 'SHASH' :
        if param_name == 'sigma' :
            return log(1 + np.exp(transformed_param))
        elif param_name == 'delta' :
            return log(1 + np.exp(transformed_param * 10)) / 10 + 0.3
