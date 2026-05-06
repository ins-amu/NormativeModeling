#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 13:14:56 2025

@author: tng
"""

# Defines regression models with different random effect specifications
# In pcntoolkit v 1.01

from pcntoolkit import (
    LinearBasisFunction,
    BsplineBasisFunction,
    NormalLikelihood,
    SHASHbLikelihood,
    make_prior,
)

def lik_lin_rand_intcp_noise() :
    mu = make_prior('mu', linear=True, 
                    basis_function=LinearBasisFunction(), 
                    intercept = make_prior('intercept_mu', random=True),
                    mapping='softplus')
    sigma = make_prior('sigma', random=True, mapping='softplus')
    likelihood = NormalLikelihood(mu, sigma)
    return likelihood

def lik_lin_rand_intcp_slope_noise() :
    mu = make_prior('mu', linear=True, 
                    basis_function=LinearBasisFunction(),
                    intercept = make_prior('intercept_mu', random=True), 
                    slope = make_prior('slope_mu', random=True),
                    mapping="softplus")
    sigma = make_prior('sigma', random=True, mapping="softplus")
    likelihood = NormalLikelihood(mu, sigma)
    return likelihood
    
def lik_spline_rand_intcp_noise() :
    mu = make_prior('mu', linear=True, 
                    basis_function=BsplineBasisFunction(),
                    intercept = make_prior('intercept_mu', random=True), 
                    mapping="softplus")
    sigma = make_prior('sigma', random=True, mapping="softplus")
    likelihood = NormalLikelihood(mu, sigma)
    return likelihood

def lik_spline_rand_intcp_slope_noise():
    mu = make_prior('mu', linear=True, 
                    basis_function=BsplineBasisFunction(),
                    intercept = make_prior('intercept_mu', random=True),
                    slope = make_prior('slope_mu', random=True), 
                    mapping='softplus')
    sigma = make_prior('sigma', random=True, mapping='softplus')
    likelihood = NormalLikelihood(mu, sigma)
    return likelihood
    
def lik_lin_rand_intcp_noise_intcp_lin():
    mu = make_prior('mu', linear=True, 
                    basis_function=LinearBasisFunction(),
                    intercept = make_prior('intercept_mu', random=True), 
                    mapping='softplus')
    sigma = make_prior('sigma', linear=True, 
                       basis_function=LinearBasisFunction(),
                       intercept = make_prior('intercept_sigma', random=True),
                       mapping='softplus')
    likelihood = NormalLikelihood(mu, sigma)
    return likelihood
    
def lik_lin_rand_intcp_slope_noise_intcp_lin() :
    mu = make_prior('mu', linear=True, 
                    basis_function=LinearBasisFunction(), 
                    intercept = make_prior('intercept_mu', random=True),
                    slope = make_prior('slope_mu', random=True), 
                    mapping="softplus")
    sigma = make_prior('sigma', linear=True, 
                       basis_function=LinearBasisFunction(),
                       intercept = make_prior('intercept_sigma', random=True),
                       mapping='softplus')
    likelihood = NormalLikelihood(mu, sigma)
    return likelihood

def lik_spline_rand_intcp_noise_intcp_lin():
    mu = make_prior('mu', linear=True, 
                    basis_function=BsplineBasisFunction(),
                    intercept = make_prior('intercept_mu', random=True), 
                    mapping="softplus")

    sigma = make_prior('sigma', linear=True,
                       basis_function=LinearBasisFunction(),
                       intercept = make_prior('intercept_sigma', random=True),
                       mapping='softplus')

    likelihood = NormalLikelihood(mu, sigma)
    return likelihood  
  
def lik_spline_rand_intcp_slope_noise_intcp_lin():
    mu = make_prior('mu', linear=True, 
                    basis_function=BsplineBasisFunction(),
                    intercept = make_prior('intercept_mu', random=True),
                    slope = make_prior('slope_mu', random=True),
                    mapping="softplus")
    sigma = make_prior('sigma', linear=True, 
                       basis_function=LinearBasisFunction(),
                       intercept = make_prior('intercept_sigma', random=True),
                       mapping='softplus')
    likelihood = NormalLikelihood(mu, sigma)
    return likelihood

def lik_lin_rand_intcp_noise_intcp_slope_lin():
    mu = make_prior('mu', linear=True, 
                    basis_function=LinearBasisFunction(),
                    intercept = make_prior('intercept_mu', random=True),
                    mapping='softplus')
    sigma = make_prior('sigma', linear=True, 
                       basis_function=LinearBasisFunction(),
                       intercept = make_prior('intercept_sigma', random=True),
                       slope = make_prior('slope_sigma', random=True),
                       mapping='softplus')
    likelihood = NormalLikelihood(mu, sigma)
    return likelihood

def lik_lin_rand_intcp_slope_noise_intcp_slope_lin() :
    mu = make_prior('mu', linear=True, 
                    basis_function=LinearBasisFunction(),
                    intercept = make_prior('intercept_mu', random=True),
                    slope = make_prior('slope_mu', random=True), 
                    mapping="softplus")
    sigma = make_prior('sigma', linear=True, 
                       basis_function=LinearBasisFunction(),
                       intercept = make_prior('intercept_sigma', random=True),
                       slope = make_prior('slope_sigma', random=True),
                       mapping='softplus')
    likelihood = NormalLikelihood(mu, sigma)
    return likelihood

def lik_spline_rand_intcp_noise_intcp_slope_lin():
    mu = make_prior('mu', linear=True, 
                    basis_function=BsplineBasisFunction(),
                    intercept = make_prior('intercept_mu', random=True), 
                    mapping='softplus')
    sigma = make_prior('sigma', linear=True, 
                   basis_function=LinearBasisFunction(), 
                   intercept = make_prior('intercept_sigma', random=True),
                   slope = make_prior('slope_sigma', random=True),
                   mapping='softplus')
    likelihood = NormalLikelihood(mu, sigma)
    return likelihood

def lik_spline_rand_intcp_slope_noise_intcp_slope_lin():
    mu = make_prior('mu', linear=True, 
                    basis_function=BsplineBasisFunction(),
                    intercept = make_prior('intercept_mu', random=True),
                    slope = make_prior('slope_mu', random=True),
                    mapping='softplus')
    sigma = make_prior('sigma', linear=True,
                       basis_function=LinearBasisFunction(),
                       intercept = make_prior('intercept_sigma', random=True),
                       slope = make_prior('slope_sigma', random=True),
                       mapping='softplus')
    likelihood = NormalLikelihood(mu, sigma)
    return likelihood

def lik_shash_lin_rand_intcp_noise():
    mu = make_prior('mu', linear=True,
                    basis_function=LinearBasisFunction(),
                    intercept = make_prior('intercept_mu', random=True),
                    mapping='softplus')
    sigma = make_prior('sigma', random=True, mapping='softplus')
    epsilon = make_prior(dist_name="Normal",dist_params=(0.0, 1.0),)
    delta = make_prior(dist_name="Normal", dist_params=(1.0, 1.0), 
                       mapping="softplus", mapping_params=(0.0, 3.0, 0.6))
    likelihood = SHASHbLikelihood(mu, sigma, epsilon, delta)
    return likelihood
    
def lik_shash_spline_rand_intcp_noise():
    mu = make_prior('mu', linear=True,
                    basis_function=BsplineBasisFunction(),  
                    intercept = make_prior('intercept_mu', random=True),
                    mapping='softplus')
    sigma = make_prior('sigma', random=True, mapping='softplus')
    epsilon = make_prior(dist_name="Normal",dist_params=(0.0, 1.0),)
    delta = make_prior(dist_name="Normal", dist_params=(1.0, 1.0), 
                       mapping="softplus", mapping_params=(0.0, 3.0, 0.6))
    likelihood = SHASHbLikelihood(mu, sigma, epsilon, delta)
    return likelihood

def lik_shash_lin_rand_intcp_noise_intcp_lin():
    mu = make_prior('mu', linear=True,
                    basis_function=LinearBasisFunction(),  
                    intercept = make_prior('intercept_mu', random=True),
                    mapping='softplus')
    sigma = make_prior('sigma', mapping='softplus', linear=True,
                    basis_function=LinearBasisFunction(),
					intercept = make_prior('intercept_mu', random=True))
    epsilon = make_prior(dist_name="Normal",dist_params=(0.0, 1.0),)
    delta = make_prior(dist_name="Normal", dist_params=(1.0, 1.0), 
                       mapping="softplus", mapping_params=(0.0, 3.0, 0.6))
    likelihood = SHASHbLikelihood(mu, sigma, epsilon, delta)
    return likelihood

def lik_shash_spline_rand_intcp_noise_intcp_spline():
    mu = make_prior('mu', linear=True,
                    basis_function=LinearBasisFunction(),  
                    intercept = make_prior('intercept_mu', random=True),
                    mapping='softplus')
    sigma = make_prior('sigma', mapping='softplus', linear=True,
                    basis_function=BsplineBasisFunction(),
					intercept = make_prior('intercept_mu', random=True))
    epsilon = make_prior(dist_name="Normal",dist_params=(0.0, 1.0),)
    delta = make_prior(dist_name="Normal", dist_params=(1.0, 1.0), 
                       mapping="softplus", mapping_params=(0.0, 3.0, 0.6))
    likelihood = SHASHbLikelihood(mu, sigma, epsilon, delta)
    return likelihood

def lik_shash_lin_rand_intcp_noise_eps():
    mu = make_prior('mu', linear=True,
                    basis_function=LinearBasisFunction(),  
                    intercept = make_prior('intercept_mu', random=True),
                    mapping='softplus')
    sigma = make_prior('sigma', random=True, mapping='softplus')
    epsilon = make_prior(dist_name="Normal", random=True)
    delta = make_prior(dist_name="Normal", dist_params=(1.0, 1.0), 
                       mapping="softplus", mapping_params=(0.0, 3.0, 0.6))
    likelihood = SHASHbLikelihood(mu, sigma, epsilon, delta)
    return likelihood

def lik_shash_spline_rand_intcp_noise_eps():
    mu = make_prior('mu', linear=True,
                    basis_function=BsplineBasisFunction(),  
                    intercept = make_prior('intercept_mu', random=True),
                    mapping='softplus')
    sigma = make_prior('sigma', random=True, mapping='softplus')
    epsilon = make_prior(dist_name="Normal", random=True)
    delta = make_prior(dist_name="Normal", dist_params=(1.0, 1.0), 
                       mapping="softplus", mapping_params=(0.0, 3.0, 0.6))
    likelihood = SHASHbLikelihood(mu, sigma, epsilon, delta)
    return likelihood

def lik_shash_lin_rand_intcp_noise_delta():
    mu = make_prior('mu', linear=True, 
                    basis_function=LinearBasisFunction(),  
                    intercept = make_prior('intercept_mu', random=True),
                    mapping='softplus')
    sigma = make_prior('sigma', random=True, mapping='softplus')
    epsilon = make_prior(dist_name="Normal",dist_params=(0.0, 1.0),)
    delta = make_prior(random=True, 
                       mapping="softplus", mapping_params=(0.0, 3.0, 0.6))
    likelihood = SHASHbLikelihood(mu, sigma, epsilon, delta)
    return likelihood

def lik_shash_spline_rand_intcp_noise_delta():
    mu = make_prior('mu', linear=True,
                    basis_function=BsplineBasisFunction(),  
                    intercept = make_prior('intercept_mu', random=True),
                    mapping='softplus')
    sigma = make_prior('sigma', random=True, mapping='softplus')
    epsilon = make_prior(dist_name="Normal",dist_params=(0.0, 1.0),)
    delta = make_prior(random=True, 
                       mapping="softplus", mapping_params=(0.0, 3.0, 0.6))
    likelihood = SHASHbLikelihood(mu, sigma, epsilon, delta)
    return likelihood

def lik_shash_lin_rand_intcp_noise_delta_eps():
    mu = make_prior('mu', linear=True,
                    basis_function=LinearBasisFunction(),  
                    intercept = make_prior('intercept_mu', random=True),
                    mapping='softplus')
    sigma = make_prior('sigma', random=True, mapping='softplus')
    epsilon = make_prior(dist_name="Normal", random=True)
    delta = make_prior(random=True, 
                       mapping="softplus", mapping_params=(0.0, 3.0, 0.6))
    likelihood = SHASHbLikelihood(mu, sigma, epsilon, delta)
    return likelihood

def lik_shash_spline_rand_intcp_noise_delta_eps():
    mu = make_prior('mu', linear=True,
                    basis_function=BsplineBasisFunction(),  
                    intercept = make_prior('intercept_mu', random=True),
                    mapping='softplus')
    sigma = make_prior('sigma', random=True, mapping='softplus')
    epsilon = make_prior(dist_name="Normal", random=True)
    delta = make_prior(random=True, 
                       mapping="softplus", mapping_params=(0.0, 3.0, 0.6))
    likelihood = SHASHbLikelihood(mu, sigma, epsilon, delta)
    return likelihood

def lik_shash_lin_rand_intcp_lin3():
	mu = make_prior('mu', linear=True,
		                basis_function=LinearBasisFunction(),  
		                intercept = make_prior('intercept_mu', random=True),
		                mapping='softplus')
	sigma = make_prior('sigma', random=True, mapping='softplus', linear=True, basis_function=LinearBasisFunction())
	epsilon = make_prior(dist_name="Normal",dist_params=(0.0, 1.0), linear=True, basis_function=LinearBasisFunction())
	delta = make_prior(dist_name="Normal", dist_params=(1.0, 1.0), linear=True, basis_function=LinearBasisFunction(), 
	                   mapping="softplus", mapping_params=(0.0, 3.0, 0.6))
	likelihood = SHASHbLikelihood(mu, sigma, epsilon, delta)
	return likelihood

def lik_shash_lin_rand_intcp_lin2():
	mu = make_prior('mu', linear=True,
		                basis_function=LinearBasisFunction(),  
		                intercept = make_prior('intercept_mu', random=True),
		                mapping='softplus')
	sigma = make_prior('sigma', random=True, mapping='softplus', linear=True, basis_function=LinearBasisFunction())
	epsilon = make_prior(dist_name="Normal",dist_params=(0.0, 1.0), linear=True, basis_function=LinearBasisFunction())
	delta = make_prior(dist_name="Normal", dist_params=(1.0, 1.0), 
	                   mapping="softplus", mapping_params=(0.0, 3.0, 0.6))
	likelihood = SHASHbLikelihood(mu, sigma, epsilon, delta)
	return likelihood

def lik_shash_spline_rand_intcp_spline3():
	mu = make_prior('mu', linear=True,
		                basis_function=BsplineBasisFunction(),  
		                intercept = make_prior('intercept_mu', random=True),
		                mapping='softplus')
	sigma = make_prior('sigma', random=True, mapping='softplus', linear=True, basis_function=BsplineBasisFunction())
	epsilon = make_prior(dist_name="Normal",dist_params=(0.0, 1.0), linear=True, basis_function=BsplineBasisFunction())
	delta = make_prior(dist_name="Normal", dist_params=(1.0, 1.0), linear=True, basis_function=BsplineBasisFunction(),
	                   mapping="softplus", mapping_params=(0.0, 3.0, 0.6))
	likelihood = SHASHbLikelihood(mu, sigma, epsilon, delta)
	return likelihood

def lik_shash_spline_rand_intcp_spline2():
	mu = make_prior('mu', linear=True,
		                basis_function=BsplineBasisFunction(),  
		                intercept = make_prior('intercept_mu', random=True),
		                mapping='softplus')
	sigma = make_prior('sigma', random=True, mapping='softplus', linear=True, basis_function=BsplineBasisFunction())
	epsilon = make_prior(dist_name="Normal",dist_params=(0.0, 1.0), linear=True, basis_function=BsplineBasisFunction())
	delta = make_prior(dist_name="Normal", dist_params=(1.0, 1.0),
	                   mapping="softplus", mapping_params=(0.0, 3.0, 0.6))
	likelihood = SHASHbLikelihood(mu, sigma, epsilon, delta)
	return likelihood

def lik_shash_spline_rand_intcp_slope_noise():
	mu = make_prior('mu', linear=True,
		                basis_function=BsplineBasisFunction(),  
		                intercept = make_prior('intercept_mu', random=True),
		                mapping='softplus')
	sigma = make_prior('sigma', random=True, mapping='softplus')
	epsilon = make_prior(dist_name="Normal",dist_params=(0.0, 1.0))
	delta = make_prior(dist_name="Normal", dist_params=(1.0, 1.0),
	                   mapping="softplus", mapping_params=(0.0, 3.0, 0.6))
	likelihood = SHASHbLikelihood(mu, sigma, epsilon, delta)
	return likelihood

def lik_shash_lin_rand_intcp_slope_noise():
	mu = make_prior('mu', linear=True,
		                basis_function=LinearBasisFunction(),  
		                intercept = make_prior('intercept_mu', random=True),
		                mapping='softplus')
	sigma = make_prior('sigma', random=True, mapping='softplus')
	epsilon = make_prior(dist_name="Normal",dist_params=(0.0, 1.0))
	delta = make_prior(dist_name="Normal", dist_params=(1.0, 1.0),
	                   mapping="softplus", mapping_params=(0.0, 3.0, 0.6))
	likelihood = SHASHbLikelihood(mu, sigma, epsilon, delta)
	return likelihood

def lik_shash_spline_rand_intcp_slope_noise_eps():
	mu = make_prior('mu', linear=True,
		                basis_function=BsplineBasisFunction(),  
		                intercept = make_prior('intercept_mu', random=True),
		                mapping='softplus')
	sigma = make_prior('sigma', random=True, mapping='softplus')
	epsilon = make_prior(dist_name="Normal",dist_params=(0.0, 1.0), random=True)
	delta = make_prior(dist_name="Normal", dist_params=(1.0, 1.0),
	                   mapping="softplus", mapping_params=(0.0, 3.0, 0.6))
	likelihood = SHASHbLikelihood(mu, sigma, epsilon, delta)
	return likelihood
