"""
Unit operations of the full analysis pipeline.

Every unit operation requires a working directory argument ``wd``.
Some unit operations need additional kwargs.
"""

import logging
import pathlib

import arviz
import calibr8
import json
import murefi
import numpy
import pandas
import scipy
import sys
import time
from matplotlib import pyplot

try:
    import pymc as pm
except ModuleNotFoundError:
    import pymc3 as pm

import preprocessing
import models
import plotting


_log = logging.getLogger(__file__)


DP_RAW = pathlib.Path(__file__).parent.absolute() / "raw_data"


def preprocess_glucose_calibration(wd: pathlib.Path):
    X, Y = preprocessing.read_glucose_x_y(
        fp_dilutions=DP_RAW / "8EXA1W_dilution_factors_glc.xlsx",
        fp_rdata=DP_RAW / "8EXA1W_ReaderOutput_0_fresh.xml",
        stock_concentration=50.0,
    )
    df = pandas.DataFrame(data=dict(concentration=X, absorbance=Y)).set_index(
        "concentration"
    )
    df.to_excel(wd / "glucose_calibration_data.xlsx")
    return


def preprocess_biomass_calibration(wd: pathlib.Path):
    df_data = pandas.DataFrame(
        columns=["data_point", "runid", "independent", "dependent"]
    ).set_index(["data_point"])
    df_data.head()

    for runid in ["7MFD4H", "7N3HF5"]:
        # get stock CDW
        stock_mean, stock_sem = preprocessing.read_biomass_stock_concentration(
            DP_RAW / f"{runid}_weights_before.csv",
            DP_RAW / f"{runid}_weights_after.csv",
            eppi_from=7,
            eppi_to=12,
        )
        print(
            f"Run {runid} was performed with a stock of {stock_mean} ± {stock_sem} gCDW/L"
        )

        # and the dilution factors from this run
        df_dilutions = preprocessing.read_biomass_dilution_factors(
            DP_RAW / f"{runid}_dilution_factors_cdw.xlsx"
        )

        independent, dependent = preprocessing.read_biomass_x_and_y(
            fp_bldata=DP_RAW / f"{runid}_Pahpshmir.csv",
            df_dilutions=df_dilutions,
            rpm=1400,
            filterset="BS3",
            stock_concentration=stock_mean,
        )
        # collect into the DataFrame
        for ind, dep in zip(independent, dependent):
            df_data.loc[len(df_data)] = (runid, ind, dep)

    df_data.to_excel(wd / "biomass_calibration_data.xlsx")
    return


def preprocess_into_dataset(
    wd: pathlib.Path,
    trim_backscatter=False,
    force_glucose_zero=False,
    subset=None,
    dataset_id: str="8T1P5H",
):
    if dataset_id == "8T1P5H":
        dataset = preprocessing.create_cultivation_dataset(
            trim_backscatter=trim_backscatter,
            force_glucose_zero=force_glucose_zero,
        )
    elif dataset_id.startswith("hifreq"):
        dataset = preprocessing.create_cultivation_dataset_hifreq(
            fname_bldata=f"{dataset_id}.csv",
            trim_backscatter=trim_backscatter,
        )
    else:
        raise ValueError(f"Unknown dataset '{dataset_id}'.")
    if subset:
        _log.info("Filtering with subset %s", subset)
        filtered = murefi.Dataset()
        for rid, rep in dataset.items():
            if rid in subset:
                filtered[rid] = rep
        dataset = filtered
    _log.info("Saving dataset %s", dataset)
    dataset.save(wd / "cultivation_dataset.h5")
    return


def preprocess_parametermapping(wd: pathlib.Path):
    dataset = murefi.Dataset.load(wd / "cultivation_dataset.h5")
    model = models.MonodModel()

    df_mapping = pandas.DataFrame(columns=["rid"] + list(model.parameter_names)).set_index(
        "rid"
    )
    for rid in dataset.keys():
        df_mapping.loc[rid] = ("S0", f"X0_{rid}", "mu_max", "K_S", "Y_XS")
    df_mapping.to_excel(wd / "full_parameter_mapping.xlsx")
    df_mapping.head()
    return


def fit_glucose_calibration(wd: pathlib.Path):
    df_data = pandas.read_excel(wd / "glucose_calibration_data.xlsx", index_col=0)
    X = df_data.index.values
    Y = df_data.absorbance.values
    model = models.LogisticGlucoseCalibrationModelV1(
        independent_key="glc", dependent_key="A365"
    )
    theta_guess = [-3, 3, 2, 0.1, 3, 0.05, 0.01, 2]
    theta_bounds = [
        (-numpy.inf, 0.3),
        (2.5, 4),
        (-20, 20),
        (0, 1),
        (-3, 3),
        (1e-6, 0.1),
        (0, 0.05),
        (1, 20),
    ]

    theta_fitted, history_mle = calibr8.fit_scipy(
        model,
        independent=X,
        dependent=Y,
        theta_guess=theta_guess,
        theta_bounds=theta_bounds,
    )
    model.save(wd / "glucose_logistic.json")

    fig, axs = calibr8.plot_model(model)
    plotting.savefig(fig, "calibration_glucose", dp=wd)
    return


def fit_biomass_calibration(wd: pathlib.Path, contrib_model: str=None):
    if contrib_model:
        import calibr8_contrib
        cm = calibr8_contrib.get_model(contrib_model)
        cm.save(wd / "biomass.json")
    else:
        df_data = pandas.read_excel(wd / "biomass_calibration_data.xlsx", index_col=0)
        model = models.BLProCDWBackscatterModelV1()

        theta_fit, history = calibr8.fit_scipy(
            model,
            # pool the calibration data from all runs:
            independent=df_data.independent.values,
            dependent=df_data.dependent.values,
            theta_guess=[1.5, 400, 2, 500, 1, 0.2, 0.01, 3],
            theta_bounds=[
                (-numpy.inf, 5),
                (60, numpy.inf),
                (-4, 4),
                (100, 1000),
                (-5, 5),
                (0.001, 10),
                (0, 1),
                (1, 30),
            ],
        )
        # NOTE: The `df` hitting the upper limit is just an indication that the outcome distribution is flat-tailed.
        #       At `df=30` the distribution is already close to the Normal, but still has the useful properties of
        #       not going `nan` at extreme values.
        model.save(wd / "biomass.json")
    fig, axs = calibr8.plot_model(model)
    plotting.savefig(fig, "calibration_biomass", dp=wd)
    return


def fit_mle(wd: pathlib.Path):
    cm_biomass = models.get_biomass_model(wd)
    cm_glucose = models.get_glucose_model(wd)
    model = models.MonodModel()
    dataset = murefi.load_dataset(wd / "cultivation_dataset.h5")
    theta_mapping = models.get_parameter_mapping(dp=wd)

    # Create the objective function
    objective = murefi.objectives.for_dataset(
        dataset=dataset,
        model=model,
        parameter_mapping=theta_mapping,
        calibration_models=[cm_glucose, cm_biomass],
    )

    # If the guess has a NaN likelihood there's a problem.
    ll_guess = objective(theta_mapping.guesses)
    if not numpy.isfinite(ll_guess):
        raise ValueError(f"Your guess is shit. objective(guess)={ll_guess}")

    # Find the maximum with scipy
    fit_result = scipy.optimize.minimize(
        objective,
        x0=theta_mapping.guesses,
        bounds=theta_mapping.bounds,
        # The callback gives us a bit of a progress bar...
        callback=lambda x: sys.stdout.write("."),
    )
    if calibr8.optimization._warn_hit_bounds(
        fit_result.x, theta_mapping.bounds, theta_mapping.theta_names
    ):
        print(fit_result)
    print()
    for tf, tname in zip(fit_result.x, theta_mapping.theta_names):
        print(f"{tname: <10}{tf}")
    print()
    print(fit_result)
    with open(wd / "full_dataset_mle.json", "w") as jfile:
        json.dump(
            {k: v for k, v in zip(theta_mapping.parameters.keys(), fit_result.x)},
            jfile,
            indent=4,
        )
    return


def plot_mle(wd: pathlib.Path):
    cm_biomass = models.get_biomass_model(wd)
    cm_glucose = models.get_glucose_model(wd)
    model = models.MonodModel()
    dataset = murefi.load_dataset(wd / "cultivation_dataset.h5")
    theta_mapping = models.get_parameter_mapping(dp=wd)
    with open(wd / "full_dataset_mle.json") as jfile:
        theta_dict = json.load(jfile)

    template = murefi.Dataset.make_template_like(dataset, independent_keys=["S", "X"])
    prediction = model.predict_dataset(
        template=template, parameter_mapping=theta_mapping, parameters=theta_dict
    )
    fig = plotting.plot_mle(
        cm_biomass=cm_biomass,
        cm_glucose=cm_glucose,
        dataset=dataset,
        prediction=prediction,
    )
    plotting.savefig(fig, "MLE", dp=wd)
    return


def sample(wd: pathlib.Path):
    cm_biomass = models.get_biomass_model(wd)
    cm_glucose = models.get_glucose_model(wd)
    model = models.MonodModel()
    dataset = murefi.load_dataset(wd / "cultivation_dataset.h5")
    theta_mapping = models.get_parameter_mapping(dp=wd)
    with open(wd / "full_dataset_mle.json") as jfile:
        mle_dict = json.load(jfile)

    X_values = [item for (key, item) in mle_dict.items() if "X0" in key]
    mcmc_dict = {
        "X0_mu": numpy.mean(X_values),
        "F_offset": numpy.clip([v / numpy.mean(X_values) for v in X_values], 0.8, 1.5),
    }
    for key, item in mle_dict.items():
        if not "X0" in key:
            mcmc_dict[key] = item
    start_dict = mcmc_dict.copy()
    _log.info(start_dict)

    objective = murefi.objectives.for_dataset(
        dataset=dataset,
        model=model,
        parameter_mapping=theta_mapping,
        calibration_models=[cm_glucose, cm_biomass],
    )

    with pm.Model(coords={"replicate": list(dataset.keys())}) as pmodel:
        # Specify a hyperprior on the initial biomass group mean:
        # + centered on the planned inoculation density of 0.25 g/L in the main culture
        # + with a 10 % standard deviation to account for pipetting errors
        X0_mu = pm.Lognormal("X0_mu", mu=numpy.log(0.25), sd=0.10)

        # Model the relative offset of initial biomass between each well and the group mean
        # with a relative pipetting error of 20 %
        F_offset = pm.Lognormal("F_offset", mu=0, sd=0.20, dims=("replicate",))

        # Thereby, the initial biomass in each well is the product
        # of group mean and relative offset:
        X0 = pm.Deterministic("X0", X0_mu * F_offset, dims=("replicate",))

        # Combine the priors into a dictionary
        theta = {
            "S0": pm.Lognormal("S0", mu=numpy.log(20), sigma=0.10),
            "Y_XS": pm.Beta("Y_XS", mu=0.6, sd=0.05),
            "mu_max": pm.Beta("mu_max", mu=0.4, sd=0.1),
            "K_S": pm.HalfFlat("K_S", testval=0.02),
            # unpack the vector of initial biomasses into individual scalars
            **{f"X0_{rid}": X0[w] for w, rid in enumerate(dataset.keys())},
        }
        L = objective(theta)


    try:
        graph = pm.model_to_graphviz(pmodel)
        graph.render(filename=wd / "cultivation_model", format="pdf", cleanup=True)
    except:
        _log.warning("Failed to render model graph.")

    _log.info("Running MAP estimation")
    with pmodel:
        map_dict = pm.find_MAP(start=start_dict)

    # Save it, just like we did for the MLE.
    with open(wd / "full_dataset_map.json", "w") as jfile:
        json.dump(
            {k: numpy.array(v).tolist() for k, v in map_dict.items()},
            jfile,
            indent=4,
        )
    
    _log.info("Running MCMC")
    with pmodel:
        idata_full = pm.sample(
            step=pm.DEMetropolisZ(
                # 10x less than the default, because some posteriors are rather narrow.
                scaling=0.0001,
                # Only the first half of tuning steps is "contaminated" by initial convergence.
                # The default of 0.9 is quite conservative about this.
                tune_drop_fraction=0.8,
            ),
            return_inferencedata=True,
            tune=50_000,
            draws=500_000,
            discard_tuned_samples=False,
            start=start_dict,
            compute_convergence_checks=False,  # Can take rather long and we do this separately anyway.
            idata_kwargs={
                # Likelihood evaluation of ODE models is really expensive.
                # Enabling it doubles the runtime, but without a progress bar!
                "log_likelihood": False
            },
        )
    idata_full.to_netcdf(wd / "full_posterior.nc")
    return


def plot_diagnostics(wd: pathlib.Path):
    idata_full = arviz.from_netcdf(wd / "full_posterior.nc")

    _log.info("Plotting warmup draws")
    arviz.plot_trace(idata_full.warmup_posterior)
    pyplot.tight_layout()
    pyplot.savefig(wd / "plot_warmup.png")

    _log.info("Plotting posterior draws")
    arviz.plot_trace(idata_full)
    pyplot.tight_layout()
    pyplot.savefig(wd / "plot_trace.png")

    _log.info("Calculating summary statistics")
    df_summary = arviz.summary(idata_full)
    df_summary.to_excel(wd / "mcmc_diagnostics.xlsx")
    return


def plot_pair(wd: pathlib.Path):
    idata_full = arviz.from_netcdf(wd / "full_posterior.nc")

    arviz.rcParams["plot.max_subplots"] = 1000

    t_start = time.time()

    replacements = {
        "F_offset": "$F_{offset}$",
        "X0_mu": "$X_{0,\mu}$",
        "X0": "$X_{0}$",
        "S0": "$S_{0}$",
        "mu_max": "$\mu_{\max}$",
        "Y_XS": "$Y_{XS}$"
    }
    labeller = arviz.labels.MapLabeller(var_name_map=replacements)

    if len(idata_full.posterior.replicate.values) == 28:
        reps = idata_full.posterior.replicate.values.reshape(4, 7).flatten("F")
    else:
        reps = idata_full.posterior.replicate.values
    axs = arviz.plot_pair(
        idata_full,
        var_names=["~F_offset"],
        figsize=(40, 40),
        coords=dict(replicate=reps),
        kind='kde',
        labeller=labeller,
        backend_kwargs=dict(gridspec_kw=dict(hspace=0, wspace=0))
    )
    fig = pyplot.gcf()

    plotting.hdi_ticklimits(axs, idata_full, replacements, xlabelpad=-47, xrotate=True)
    t_end = time.time()
    _log.info(f"Plotting time: {t_end - t_start:.0f} seconds.")
    plotting.savefig(fig, "plot_pair", dp=wd, dpi=200, bbox_inches="tight")
    _log.info(f"Saving time: {time.time() - t_end:.0f} seconds.")
    return


def summarize_parameters(wd: pathlib.Path):
    with open(wd / "full_dataset_mle.json") as jfile:
        theta_dict = json.load(jfile)
    idata = arviz.from_netcdf(wd / "full_posterior.nc")

    df = arviz.summary(idata, var_names="S0,mu_max,Y_XS,K_S,X0_mu,X0".split(","), hdi_prob=0.9, round_to="none")
    rename_dict = {}
    for k in df.index.values:
        if "[" in k:
            rvname, cindex = k.strip("]").split("[")
            try:
                cindex = int(cindex)
                cname = tuple(idata.posterior[rvname].coords.keys())[2+0]
                cval = idata.posterior.coords[cname].values[cindex]
            except:
                _log.warning("Unexpected error in summary index interpretation. rvname: %s, cindex: %s, ArviZ version: %s", rvname, cindex, arviz.__version__)
                cval = cindex
            rename_dict[k] = f"{rvname}_{cval}"
    df = df.rename(index=rename_dict)
    df = df.assign(MLE=pandas.Series(theta_dict)).fillna("-")
    cols = "MLE,mean,sd,hdi_5%,hdi_95%,r_hat"#,mcse_mean,mcse_sd,ess_mean,ess_sd,ess_bulk,ess_tail"
    df = df[cols.split(",")]
    df.to_excel(wd / "summary.xlsx")
    return


def plot_ks_curvature(wd: pathlib.Path):
    idata = arviz.from_netcdf(wd / "full_posterior.nc")
    cm_biomass = models.get_biomass_model(wd)
    cm_glucose = models.get_glucose_model(wd)
    model = models.MonodModel()
    dataset = murefi.load_dataset(wd / "cultivation_dataset.h5")

    # Select replicates for which the backscatter is above 17
    rids = {
        rid
        for rid, rep in dataset.items()
        if rep["Pahpshmir_1400_BS3_CgWT"].y[-1] > 17
    }
    _log.info("Candidates for curvature plots: %s", rids)

    for rid in rids:
        _log.info("Preparing K_S curvature plot for %s", rid)
        theta_mapping = models.get_parameter_mapping(rids=[rid], dp=wd)

        fig, axs = plotting.plot_ks_curvature(
            idata,
            theta_mapping,
            model,
            dataset[rid],
            cm_biomass,
            cm_glucose,
        )
        plotting.savefig(fig, f"ks_curvature_{rid}", dp=wd)
        pyplot.close()
    return
