"""Utils, some modified from pcntoolkit"""

from typing import TYPE_CHECKING, Any, Dict, List, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
from matplotlib.font_manager import FontProperties
from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model import NormativeModel
from pcntoolkit import (
    HBR,
    NormativeModel,
    plot_qq,
    plot_ridge
)
import arviz as az
import pymc as pm
import os
import copy

sns.set_theme(style="ticks")


def plot_centiles(
    model: "NormativeModel",
    centiles: List[float] | np.ndarray | None = None,
    conditionals: List[float] | np.ndarray | None = None,
    covariate: str | None = None,
    covariate_range: tuple[float, float] = (None, None),  # type: ignore
    batch_effects: Dict[str, List[str]] | None | Literal["all"] = None,
    scatter_data: NormData | None = None,
    harmonize_data: bool = True,
    hue_data: str = "site",
    markers_data: str = "sex",
    show_other_data: bool = False,
    show_thrivelines: bool = False,
    z_thrive: float = 0.0,
    save_dir: str | None = None,
    show_centile_labels: bool = True,
    show_legend: bool = True,
    show_yhat: bool = False,
    plt_kwargs: dict | None = None,
    scatter_kwargs : dict | None = None,
    ax = None,
    **kwargs: Any,
) :
    """Generate centile plots for response variables with optional data overlay.

    This function creates visualization of centile curves for all response variables
    in the dataset. It can optionally show the actual data points overlaid on the
    centile curves, with customizable styling based on categorical variables.

    Parameters
    ----------
    model: NormativeModel
        The model to plot the centiles for.
    centiles: List[float] | np.ndarray | None, optional
        The centiles to plot. If None, the default centiles will be used.
    conditionals: List[float] | np.ndarray | None, optional
        A list of x-coordinates for which to plot the conditionals
    covariate: str | None, optional
        The covariate to plot on the x-axis. If None, the first covariate in the model will be used.
    covariate_range: tuple[float, float], optional
        The range of the covariate to plot on the x-axis. If None, the range of the covariate that was in the train data will be used.
    batch_effects: Dict[str, List[str]] | None | Literal["all"], optional
        The batch effects to plot the centiles for. If None, the batch effect that appears first in alphabetical order will be used.
    scatter_data: NormData | None, optional
        Data to scatter on top of the centiles.
    harmonize_data: bool, optional
        Whether to harmonize the scatter data before plotting. Data will be harmonized to the batch effect for which the centiles were computed.
    hue_data: str, optional
        The column to use for color coding the data. If None, the data will not be color coded.
    markers_data: str, optional
        The column to use for marker styling the data. If None, the data will not be marker styled.
    show_other_data: bool, optional
        Whether to scatter data belonging to groups not in batch_effects.
    save_dir: str | None, optional
        The directory to save the plot to. If None, the plot will not be saved.
    show_centile_labels: bool, optional
        Whether to show the centile labels on the plot.
    show_legend: bool, optional
        Whether to show the legend on the plot.
    plt_kwargs: dict, optional
        Additional keyword arguments for the plot.
    **kwargs: Any, optional
        Additional keyword arguments for the model.compute_centiles method.

    Returns
    -------
    None
        Displays the plot using matplotlib.
    """
    if covariate is None:
        covariate = model.covariates[0]
        assert isinstance(covariate, str)

    cov_min = covariate_range[0] or model.covariate_ranges[covariate]["min"]
    cov_max = covariate_range[1] or model.covariate_ranges[covariate]["max"]
    covariate_range = (cov_min, cov_max)

    if batch_effects == "all":
        if scatter_data:
            batch_effects = scatter_data.unique_batch_effects
        else:
            batch_effects = model.unique_batch_effects
    elif batch_effects is None:
        if scatter_data:
            batch_effects = {k: [v[0]] for k, v in scatter_data.unique_batch_effects.items()}
        else:
            batch_effects = {k: [v[0]] for k, v in model.unique_batch_effects.items()}

    if plt_kwargs is None:
        plt_kwargs = {}
    if scatter_kwargs is None:
        scatter_kwargs = dict(s=50,
                              alpha=0.8,
                              zorder=1,
                              linewidth=0)

    if ax is None: 
        ax = plt.gca()

    # Create some synthetic data with a single batch effect
    # The plotted covariate is just a linspace
    centile_covariates = np.linspace(covariate_range[0], covariate_range[1], 150)
    centile_df = pd.DataFrame({covariate: centile_covariates})

    # TODO: use the mean here
    # Any other covariates are taken to be the midpoint between the observed min and max
    for cov in model.covariates:
        if cov != covariate:
            minc = model.covariate_ranges[cov]["min"]
            maxc = model.covariate_ranges[cov]["max"]
            centile_df[cov] = (minc + maxc) / 2

    # Batch effects are the first ones in the highlighted batch effects
    for be, v in batch_effects.items():
        centile_df[be] = v[0]
    # Response vars are all 0, we don't need them
    for rv in model.response_vars:
        centile_df[rv] = 0
    centile_data = NormData.from_dataframe(
        "centile",
        dataframe=centile_df,
        covariates=model.covariates,
        response_vars=model.response_vars,
        batch_effects=list(batch_effects.keys()),
    )  # type:ignore

    conditionals_data = []
    if conditionals is not None:
        for c in conditionals:
            # Compute the endpoints of the conditional curve (0.01th and 0.99th centile)
            centile = copy.deepcopy(centile_data).isel(observations=[0, 1])
            centile.X.loc[{"covariates": covariate}] = c
            model.compute_centiles(centile, centiles=[0.01, 0.99])

            # Compute the curve in between the endpoints
            conditional_d = copy.deepcopy(centile_data)
            conditional_d.X.loc[{"covariates": covariate}] = c
            for rv in model.response_vars:
                conditional_d.Y.loc[{"response_vars": rv}] = np.linspace(
                    *(centile.centiles.sel(observations=0, response_vars=rv).values.tolist()), 150
                )
            if not hasattr(conditional_d, "logp"):
                model.compute_logp(conditional_d)
            conditionals_data.append(conditional_d)

    if not hasattr(centile_data, "centiles"):
        model.compute_centiles(centile_data, centiles=centiles, recompute=False,**kwargs)
    if scatter_data and show_thrivelines:
        model.compute_thrivelines(scatter_data, z_thrive=z_thrive)
    if show_yhat and not hasattr(centile_data, "yhat"):
        model.compute_yhat(centile_data)

    if not model.has_batch_effect:
        batch_effects = {}

    if harmonize_data and scatter_data:
        if model.has_batch_effect:
            reference_batch_effect = {k: v[0] for k, v in batch_effects.items()}
            model.harmonize(scatter_data, reference_batch_effect=reference_batch_effect)
        else:
            model.harmonize(scatter_data)

    for response_var in model.response_vars:
             _plot_centiles(
            centile_data=centile_data,
            response_var=response_var,
            covariate=covariate,
            conditionals_data=conditionals_data,
            batch_effects=batch_effects,
            scatter_data=scatter_data,
            harmonize_data=harmonize_data,
            hue_data=hue_data,
            markers_data=markers_data,
            show_other_data=show_other_data,
            show_thrivelines=show_thrivelines,
            save_dir=save_dir,
            show_centile_labels=show_centile_labels,
            show_legend=show_legend,
            show_yhat=show_yhat,
            plt_kwargs=plt_kwargs,
            scatter_kwargs=scatter_kwargs,
            ax=ax,
        )
    return ax


def _plot_centiles(
    centile_data: NormData,
    response_var: str,
    covariate: str = None,  # type: ignore
    conditionals_data: List[NormData] | None = None,
    batch_effects: Dict[str, List[str]] = None,  # type: ignore
    scatter_data: NormData | None = None,
    harmonize_data: bool = True,
    hue_data: str = "site",
    markers_data: str = "sex",
    show_other_data: bool = False,
    show_thrivelines: bool = False,
    save_dir: str | None = None,
    show_centile_labels: bool = True,
    show_legend: bool = True,
    show_yhat: bool = False,
    plt_kwargs: dict = None,
    scatter_kwargs: dict = None, # type: ignore,
    ax = None,
) :

    if ax is None: ax = plt.gca()

    filter_dict = {
        "covariates": covariate,
        "response_vars": response_var,
    }

    filtered = centile_data.sel(filter_dict)

    for centile in centile_data.coords["centile"][::-1]:
        d_mean = abs(centile - 0.5)
        if d_mean == 0:
            thickness = 3
        else:
            thickness = 2
        if d_mean <= 0.25:
            style = "-"

        elif d_mean <= 0.475:
            style = "--"
        else:
            style = ":"

        sns.lineplot(
            x=filtered.X,
            y=filtered.centiles.sel(centile=centile),
            color="black",
            linestyle=style,
            linewidth=thickness,
            zorder=2,
            legend="brief",
            ax=ax
        )

        font = FontProperties()
        font.set_weight("bold")
        if show_centile_labels:
            ax.text(
                s=centile.item(),
                x=filtered.X[0] - 1,
                y=filtered.centiles.sel(centile=centile)[0],
                color="black",
                horizontalalignment="right",
                verticalalignment="center",
                fontproperties=font,
            )
            ax.text(
                s=centile.item(),
                x=filtered.X[-1] + 1,
                y=filtered.centiles.sel(centile=centile)[-1],
                color="black",
                horizontalalignment="left",
                verticalalignment="center",
                fontproperties=font,
            )
    if show_yhat:
        ax.plot(filtered.X, filtered.Yhat, color="red", linestyle="--", 
                linewidth=thickness, zorder=2, label="$\\hat{Y}$")

    minx, maxx = ax.get_xlim()
    ax.set_xlim(minx - 0.1 * (maxx - minx), maxx + 0.1 * (maxx - minx))

    if scatter_data:
        scatter_filter = scatter_data.sel(filter_dict)
        df = scatter_filter.to_dataframe()
        scatter_data_name = "Y_harmonized" if harmonize_data else "Y"
        thriveline_data_name = "thrive_Y_harmonized" if harmonize_data else "thrive_Y"
        columns = [("X", covariate), (scatter_data_name, response_var)]
        columns.extend([("batch_effects", be.item()) for be in scatter_data.batch_effect_dims])
        df = df[columns]
        df.columns = [c[1] for c in df.columns]
        if batch_effects == {}:
            sns.scatterplot(
                df,
                x=covariate,
                y=response_var,
                label=scatter_data.name,
                color="black",
                s=20,
                alpha=0.6,
                zorder=1,
                linewidth=0,
                ax=ax
            )
            if show_thrivelines:
                ax.plot(scatter_filter.thrive_X.to_numpy().T, 
                        scatter_filter[thriveline_data_name].to_numpy().T)
        else:
            idx = np.full(len(df), True)
            for j in batch_effects:
                idx = np.logical_and(
                    idx,
                    df[j].isin(batch_effects[j]),
                )
            be_df = df[idx]
            scatter = sns.scatterplot(
                data=be_df,
                x=covariate,
                y=response_var,
                hue=hue_data if hue_data in df else None,
                style=markers_data if markers_data in df else None,
                ax=ax,
                **scatter_kwargs
            )
            if show_thrivelines:
                ax.plot(scatter_filter.thrive_X.to_numpy().T, scatter_filter[thriveline_data_name].to_numpy().T)

            if show_other_data:
                non_be_df = df[~idx]
                markers = ["Other data"] * len(non_be_df)
                sns.scatterplot(
                    data=non_be_df,
                    x=covariate,
                    y=response_var,
                    color="lime",
                    style=markers,
                    linewidth=0,
                    s=20,
                    alpha=0.4,
                    zorder=0,
                    ax=ax
                )

            if show_legend:
                legend = scatter.get_legend()
                if legend:
                    handles = legend.legend_handles
                    labels = [t.get_text() for t in legend.get_texts()]
                    ax.legend(
                        handles,
                        labels,
                        title_fontsize=10,
                    )
            else:
                ax.legend().remove()

    title = f"Centiles of {response_var}"
    if scatter_data:
        if harmonize_data:
            plotname = f"centiles_{response_var}_{scatter_data.name}_harmonized"
            title = f"{title}\n With harmonized {scatter_data.name} data"
        else:
            plotname = f"centiles_{response_var}_{scatter_data.name}"
            title = f"{title}\n With raw {scatter_data.name} data"

    if conditionals_data:
        for conditional_d in conditionals_data:
            filter_cond = conditional_d.sel(filter_dict)
            ax.plot(
                np.exp(filter_cond.logp.values) * 10 + filter_cond.X,
                filter_cond.Y,
                color="blue",
                linestyle="--",
                linewidth=1,
                zorder=2,
                label="Conditional",
            )
            # Put a text annotation on top of the plot, rotate the text 90 degrees
            ax.text(
                filter_cond.X[-1],
                filter_cond.Y[-1],
                f"{filter_cond.X[-1].values.item():.2f}",
                color="black",
                fontsize=10,
                ha="right",
                va="bottom",
                rotation=-90,
            )

    ax.set_title(title)
    ax.set_xlabel(covariate)
    ax.set_ylabel(response_var)

    return 0


def make_model(likelihood, model_name, save_dir) :
    
    template_hbr = HBR(
        name="template",
        # The number of cores to use for sampling.
        cores=8,
        # Whether to show a progress bar during the model fitting.
        progressbar=True,
        # The number of draws to sample from the posterior per chain.
        draws=1500,
        # The number of tuning steps to run.
        tune=500,
        # The number of MCMC chains to run.
        chains=1,
        # The sampler to use for the model.
        nuts_sampler="nutpie",
        # The likelihood function to use for the model.
        likelihood=likelihood,)

    model = NormativeModel(
        # The regression model to use for the normative model.
        template_regression_model=template_hbr,
        # Whether to save the model after fitting.
        savemodel=True,
        # Whether to evaluate the model after fitting.
        evaluate_model=True,
        # Whether to save the results after evaluation.
        saveresults=True,
        # Whether to save the plots after fitting.
        saveplots=False,
        # The directory to save the model, results, and plots.
        save_dir=save_dir + "out_" + model_name,
        # The scaler to use for the input data. Can be either one of "standardize", "minmax", "robminmax", "none"
        inscaler="none",
        # The scaler to use for the output data. Can be either one of "standardize", "minmax", "robminmax", "none"
        outscaler="none",)

    return model


def get_statistics(model_name, test_data) :

    model = NormativeModel.load(save_dir + "out_" + model_name)
    test_data = model.predict(test_data)
    test_data = model.compute_centiles(test_data, recompute=True)
    return test_data.get_statistics_df()


from matplotlib.colors import to_hex
cmap = plt.get_cmap("viridis", 12)
hex_c = [to_hex(cmap(i)) for i in range(12)]
c_1000 = "#6172ed"
c_abide = hex_c[-3]
c_camcan = hex_c[6]
c_adni = hex_c[0]
c_adhd = hex_c[-1]
c_hcp = "#4dff3d"
c_hcpaal = "#588540"
c_hca = "#00ff95"
c_hcaaal = "#3d8567"

global palette_match
palette_match = {"1000BRAINS": c_1000, "ABIDE": c_abide, 
                 "CAMCAN": c_camcan, "ADNI": c_adni, "ADHD": c_adhd,
                 "HCP": c_hcp, "HCPaal": c_hcpaal, 
				  "HCA": c_hca,
                 "HCA-AABC": c_hca, "HCA-AABCaal": c_hcaaal}


def get_zout(model, data) :
    data = model.predict(data)
    data = model.compute_centiles(data, recompute=True)
    data_ = data.to_dataframe()
    n_z196 = data_[np.array(np.abs(data["Z"]) > 1.96)].groupby(("batch_effects", "Dataset")).count()[("X", "Age")]
    return n_z196
    
def get_zout_case(model, data) :
    data = model.predict(data)
    data = model.compute_centiles(data, recompute=True)
    data_ = data.to_dataframe()
    data_["Dx"] = data["Dx"].values
    n_z196 = data_[np.array(np.abs(data["Z"]) > 1.96)].groupby([("batch_effects", "Dataset"), "Dx"]).count()[("X", "Age")]
    return n_z196


def compare_hbr_models(models:dict[str, str]):
    """Compares HBR models

    Args:
        models (dict[str, str]): dictionary of [model name, path]
    
    Returns:
        dictionary of (responsevar, comparison): [str, dataframe]
    """

    loaded_models: dict[str, NormativeModel] = {}
    for k, v in models.items():
        try:
            m = NormativeModel.load(v)
            m.predict(m.synthesize())
            loaded_models[k] = m
        except Exception as e:
            print("Cannot load model at location:", v, e)
    
    comparisons = {}
    for respvar in loaded_models[k].response_vars:
        traces = {}
        for name, model in loaded_models.items():
            with model[respvar].pymc_model:
                pm.compute_log_likelihood(model[respvar].idata)
            traces[name] = model[respvar].idata
        comparisons[respvar] = az.compare(traces)

    return comparisons


def plot_fit_save(model_name, save_dir, test_data, batch_effects, ylabel=None) :
    
    model = NormativeModel.load(save_dir + "out_" + model_name)

    stats = test_data.get_statistics_df()
    fig, ax =plt.subplots(1, 1)
    ax = plot_centiles(
        model,
        centiles=[0.05, 0.5, 0.95],
        scatter_data=test_data,   
        batch_effects=batch_effects,
        hue_data="Dataset",
        show_other_data=True,  
        harmonize=True,
        scatter_kwargs={"palette": palette_match, "alpha":.7, "s": 20}
    )
    ax.set_title("R2=" + str(stats["R2"].values[0]) + " rho=" + str(stats["Rho"].values[0]))
    if ylabel is None :
        ylabel_str = model.response_vars[-1]
    else :
        ylabel_str = ylabel
    ax.set_ylabel(ylabel_str)
    ax.legend(framealpha=0.5, bbox_to_anchor=(1, 1))
    plt.savefig(save_dir + "plots/" + model_name + ".svg", dpi=300)
    plt.savefig(save_dir + "plots/" + model_name + ".png", dpi=300)


def plot_fit(model_name, save_dir, test_data, batch_effects, ylabel=None, ax=None,
kwargs={"palette": palette_match, "alpha":.7, "s": 20}):
    
    model = NormativeModel.load(save_dir + "out_" + model_name)
    test_data = model.predict(test_data)
    stats = test_data.get_statistics_df()
    ax0 = plot_centiles(
        model,
        centiles=[0.05, 0.5, 0.95],
        scatter_data=test_data,   
        batch_effects=batch_effects,
        hue_data="Dataset",
        show_other_data=True,  
        harmonize=True,
        show_centile_labels=False,
        ax=ax,
        scatter_kwargs=kwargs
    )
    ax.set_title("")
    ylim = ax.get_ylim()
    ax.text(40, ylim[1]-0.1*ylim[1], s=r"$R^2=$" + str(stats["R2"].values[0]) + r" $ \rho = $" + str(stats["Rho"].values[0]))
    if ylabel is None :
        ylabel_str = model.response_vars[-1]
    else :
        ylabel_str = ylabel
    ax.set_ylabel(ylabel_str)
    ax.legend("", frameon=False)
    return test_data


def plot_metrics(model_name, save_dir, test_data, batch_effects, ylabel=None, ax=None, kwargs={"palette": palette_match, "alpha":.7, "s": 20}):
    
    model = NormativeModel.load(save_dir + "out_" + model_name)
    test_data = model.predict(test_data)
    stats = test_data.get_statistics_df()
    ax0 = plot_centiles(
        model,
        centiles=[0.05, 0.5, 0.95],
        scatter_data=test_data,   
        batch_effects=batch_effects,
        hue_data="Dataset",
        show_other_data=True,  
        harmonize=True,
        show_centile_labels=False,
        ax=ax,
        scatter_kwargs=kwargs,
    )
    ax.set_title(ylabel + "\n" + r"$R^2=$" + str(stats["R2"].values[0]) + r" $ \rho = $" + str(stats["Rho"].values[0]))
    if ylabel is None :
        ylabel_str = model.response_vars[-1]
    else :
        ylabel_str = ylabel
    ax.set_ylabel(ylabel_str)
    ax.legend("", frameon=False)
    return test_data
