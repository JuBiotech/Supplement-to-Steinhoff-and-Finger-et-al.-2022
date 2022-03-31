"""
Unit operations of the full analysis pipeline.

Every unit operation requires a working directory argument ``wd``.
Some unit operations need additional kwargs.
"""

import logging
import pathlib

import arviz
import calibr8
import fastprogress
import json
import murefi
import numpy
import pandas
import scipy
import sys
import time
from matplotlib import pyplot

import bletl
try:
    import pymc as pm
except:
    import pymc3 as pm

import preprocessing
import models
import plotting


_log = logging.getLogger(__file__)


DP_RAW = pathlib.Path(__file__).parent.absolute() / "raw_data"


def prepare_glucose_calibration(wd: pathlib.Path):
    cm = models.get_glucose_model()
    cm.save(wd / "glucose.json")
    return


def prepare_biomass_calibration(wd: pathlib.Path):
    cm = models.get_biomass_model()
    cm.save(wd / "biomass.json")
    return


def prepare_dataset(
    wd: pathlib.Path,
    trim_backscatter=True,
    force_glucose_zero=True,
    fname_bldata:str="757-MO_Fast-MO-2022-02-08-20-59-49.csv",
    mask_from: float=numpy.inf,
    mask_to: float=0,
):
    dataset = preprocessing.create_cultivation_dataset(
        fname_bldata=fname_bldata,
        trim_backscatter=trim_backscatter,
        force_glucose_zero=force_glucose_zero,
        mask_from=mask_from,
        mask_to=mask_to,
    )
    _log.info("Saving dataset %s", dataset)
    dataset.save(wd / "cultivation_dataset.h5")
    return


def prepare_parametermapping(wd: pathlib.Path):
    dataset = murefi.Dataset.load(wd / "cultivation_dataset.h5")
    model = models.MonodModel()

    df_mapping = pandas.DataFrame(columns=["rid", *model.parameter_names]).set_index("rid")
    for rid in dataset.keys():
        df_mapping.loc[rid] = ("S0", "X0", "mu_max", "K_S", "Y_XS")
    df_mapping.to_excel(wd / "full_parameter_mapping.xlsx")
    return


def plot_glucose_calibration(wd: pathlib.Path):
    cm = models.LogisticGlucoseCalibrationModelV1.load(wd / "glucose.json")

    fig = pyplot.figure(figsize=(12*1.3, 3.65*1.3), dpi=120)
    gs1 = fig.add_gridspec(1, 3, wspace=0.05, width_ratios=[1.125, 1.125, 1.5])
    gs2 = fig.add_gridspec(1, 3, wspace=0.5, width_ratios=[1.125, 1.125, 1.5])
    axs = []
    axs.append(fig.add_subplot(gs1[0, 0]))
    axs.append(fig.add_subplot(gs1[0, 1], sharey=axs[0]))
    pyplot.setp(axs[1].get_yticklabels(), visible=False)
    axs.append(fig.add_subplot(gs2[0, 2]))
    calibr8.plot_model(cm, fig=fig, axs=axs)
    xlabel = r"$\mathrm{glucose\ concentration\ /\ g\ L^{-1}}$"
    axs[0].set(
        ylabel=r"$\mathrm{absorbance_{365\ nm}}\ /\ -$",
        xlabel=xlabel,
    )
    axs[1].set(
        xlabel=xlabel,
    )
    axs[2].set(
        ylabel=r"$\mathrm{absolute\ residual\ /\ -}$",
        xlabel=xlabel,
    )
    axs[2].legend(loc="upper left")
    plotting.savefig(fig, "calibration_glucose", dp=wd)
    return


def plot_biomass_calibration(wd: pathlib.Path):
    cm = models.BLProCDWBackscatterModelV1.load(wd / "biomass.json")

    fig = pyplot.figure(figsize=(12*1.3, 3.65*1.3), dpi=120)
    gs1 = fig.add_gridspec(1, 3, wspace=0.05, width_ratios=[1.125, 1.125, 1.5])
    gs2 = fig.add_gridspec(1, 3, wspace=0.5, width_ratios=[1.125, 1.125, 1.5])
    axs = []
    axs.append(fig.add_subplot(gs1[0, 0]))
    axs.append(fig.add_subplot(gs1[0, 1], sharey=axs[0]))
    pyplot.setp(axs[1].get_yticklabels(), visible=False)
    axs.append(fig.add_subplot(gs2[0, 2]))
    calibr8.plot_model(cm, fig=fig, axs=axs)
    xlabel = r"$\mathrm{biomass\ concentration\ /\ g\ L^{-1}}$"
    axs[0].set(
        ylabel=r"$\mathrm{backscatter\ /\ a.u.}$",
        xlabel=xlabel,
    )
    axs[1].set(
        xlabel=xlabel,
    )
    axs[2].set(
        ylabel=r"$\mathrm{absolute\ residual\ /\ a.u.}$",
        xlabel=xlabel,
    )
    axs[2].legend(loc="upper left")
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
    mcmc_dict = {}
    for key, item in mle_dict.items():
        if not "K_S" in key:
            mcmc_dict[key] = item
    start_dict = mcmc_dict.copy()
    _log.info(start_dict)

    objective = murefi.objectives.for_dataset(
        dataset=dataset,
        model=model,
        parameter_mapping=theta_mapping,
        calibration_models=[cm_glucose, cm_biomass],
    )

    with pm.Model() as pmodel:
        # Combine the priors into a dictionary
        theta = {
            "S0": pm.Lognormal("S0", mu=numpy.log(20), sigma=0.10),
            "X0": pm.Lognormal("X0", mu=numpy.log(0.25), sd=0.10),
            "mu_max": pm.Beta("mu_max", mu=0.4, sd=0.1),
            "K_S": pm.HalfFlat("K_S", initval=0.02),
            "Y_XS": pm.Beta("Y_XS", mu=0.6, sd=0.05),
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
            tune=20_000,
            draws=50_000,
            discard_tuned_samples=False,
            initvals=start_dict,
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
        "X0": "$X_{0}$\n$g\ L^{-1}$",
        "S0": "$S_{0}$\n$g\ L^{-1}$",
        "mu_max": "$\mu_{\max}$\n$\mathrm{h}$",
        "K_S": "$K_S$\n$g\ L^{-1}$",
        "Y_XS": "$Y_{XS}$\n$g\ L^{-1}$",
    }
    labeller = arviz.labels.MapLabeller(var_name_map=replacements)

    axs = arviz.plot_pair(
        idata_full,
        figsize=(8, 8),
        kind='kde',
        labeller=labeller,
        backend_kwargs=dict(gridspec_kw=dict(hspace=0, wspace=0))
    )
    fig = pyplot.gcf()

    plotting.hdi_ticklimits(axs, idata_full, replacements, xlabelpad=-47, xrotate=True)
    newlim = axs[-1, -1].get_xlim()[1] * 0.7
    for ax in axs[-2, :]:
        ax.set_ylim(0, newlim)
    axs[-1, -1].set_xlim(0, newlim)
    t_end = time.time()
    _log.info(f"Plotting time: {t_end - t_start:.0f} seconds.")
    plotting.savefig(fig, "plot_pair", dp=wd, dpi=200, bbox_inches="tight")
    _log.info(f"Saving time: {time.time() - t_end:.0f} seconds.")
    return


def summarize_parameters(wd: pathlib.Path):
    with open(wd / "full_dataset_mle.json") as jfile:
        theta_dict = json.load(jfile)
    idata = arviz.from_netcdf(wd / "full_posterior.nc")

    df = arviz.summary(idata, var_names="X0,S0,mu_max,K_S,Y_XS".split(","), hdi_prob=0.9, round_to="none")
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
        theta_mapping = models.get_parameter_mapping(wd, rids)

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


def plot_mcmc(wd: pathlib.Path):
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
    _log.info("Candidates for kinetics plots: %s", rids)

    for rid in rids:
        _log.info("Preparing full kinetics plot for %s", rid)
        theta_mapping = models.get_parameter_mapping(wd, rids)

        fig, axs = plotting.plot_full_kinetics(
            idata,
            theta_mapping,
            model,
            dataset[rid],
            cm_biomass,
            cm_glucose,
        )
        plotting.savefig(fig, f"kinetics_{rid}", dp=wd)
        pyplot.close()
    return


def report_KS(wd: pathlib.Path):
    idata = arviz.from_netcdf(wd / "full_posterior.nc")
    with (wd / "summary.txt").open("w") as file:
        ksmg = idata.posterior.K_S.stack(sample=("chain", "draw")).values * 1000
        perc_50 = numpy.percentile(ksmg, 50)
        perc_95 = numpy.percentile(ksmg, 95)
        plt001 = numpy.mean(ksmg < 10)
        file.writelines([
            f"With 50 % probability: K_S < {perc_50:.1f} mg/L.\n",
            f"With 95 % probability: K_S < {perc_95:.1f} mg/L.\n",
            f"K_S < 10 mg/L with {plt001 * 100:.1f} % probability.\n"
        ])
    return


def plot_monod_schematic(wd: pathlib.Path):
    """Makes a quadratic visualizatoin of the Monod kinetics with an exaggerated K_S value."""
    model = models.MonodModel()
    params = dict(S0=20, X0=0.25, mu_max=0.5, K_S=3, Y_XS=0.6)
    rep = model.predict_replicate(
        parameters=[
            params[pname]
            for pname in model.parameter_names
        ],
        template=murefi.Replicate.make_template(0, 11, independent_keys="SX")
    )

    fig, ax = pyplot.subplots(figsize=(4, 4), dpi=200)

    ax.plot(rep["S"].t, rep["S"].y, color="red", label="substrate")
    ax.plot(rep["X"].t, rep["X"].y, color="green", label="biomass")
    ax.annotate(
        "",
        xy=(9, 12),
        xytext=(8, 14),
        arrowprops=dict(color="black"),
    )

    params_textemplate = {
        "X0": r"$\mathrm{X_0=%s~g/L}$",
        "S0": r"$\mathrm{S_0=%s~g/L}$",
        "mu_max": r"$\mathrm{\mu_{max}}=%s~1/h}$",
        "Y_XS": r"$\mathrm{Y_{XS}=%s~g/g}$",
        "K_S": r"$\mathrm{K_{S}=%s~g/L}$",
    }
    for p, (pname, template) in enumerate(params_textemplate.items()):
        pval = params.get(pname)
        ax.text(5, 13 - (p * 2), template % pval, fontsize="x-small", ha="right")

    ax.legend()
    ax.set(
        ylabel="concentration / $g\ L^{-1}$",
        xlabel="time / h",
        ylim=(0, None),
        xlim=(0, None),
    )
    fig.tight_layout()
    plotting.savefig(fig, "plot_monod_schematic", dp=wd)
    return


def plot_raw_data(wd: pathlib.Path):
    bldata = bletl.parse(preprocessing.DP_DATA / "757-MO_Fast-MO-2022-02-08-20-59-49.csv")

    fig, ax1 = pyplot.subplots(figsize=(6, 4), dpi=200, facecolor="white")
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()

    ax3.spines.right.set_position(("axes", 1.3))

    # Backscatter
    ax = ax1
    ax.scatter(
        *bldata.get_timeseries("BS3", "A01"),
        marker=".",
        s=3,
        facecolor="green",
        edgecolor="none"
    )
    ax.set(
        ylabel="backscatter / a.u.",
        xlabel="time / h",
        ylim=(0, None),
        xlim=(0, 12),
        yticks=[0, 10, 20, 30],
        xticks=[0, 5, 10],
    )
    ax.yaxis.label.set_color("green")
    ax.tick_params(axis="y", colors="green")

    # DO
    ax = ax2
    ax.plot(
        *bldata.get_timeseries("DO", "A01"),
        color="blue",
    )
    ax.set(
        ylabel="dissolved $\mathrm{O_2}$ / %",
        ylim=(0, 110)
    )
    ax.yaxis.label.set_color("blue")
    ax.tick_params(axis="y", colors="blue")

    # pH
    ax = ax3
    ax.plot(
        *bldata.get_timeseries("pH", "A01"),
        color="orange",
    )
    ax.set(
        ylabel="pH / -",
        ylim=(6, 7)
    )
    ax.yaxis.label.set_color("orange")
    ax.tick_params(axis="y", colors="orange")

    fig.tight_layout()
    plotting.savefig(fig, "plot_raw_data", dp=wd)
    return


def plot_raw_data_zoom(wd: pathlib.Path):
    bldata = bletl.parse(preprocessing.DP_DATA / "757-MO_Fast-MO-2022-02-08-20-59-49.csv")

    fig, ax1 = pyplot.subplots(figsize=(4.5, 4), dpi=200, facecolor="white")
    ax2 = ax1.twinx()

    # Backscatter
    ax = ax1

    t, bs = bldata.get_timeseries("BS3", "A01")
    tmax = t[numpy.argmax(bs)]
    mask = t <= tmax
    ax.plot(t[mask], bs[mask], marker="x", color="green")
    ax.scatter(t[~mask], bs[~mask], marker=".", facecolor="green", edgecolor="none")
    ax.annotate(
        "max(backscatter)", color="green",
        fontsize=8,
        xy=(tmax, 31.1), xytext=(tmax, 29.8), arrowprops=dict(color="green", width=1, headwidth=5, headlength=5)
    )
    ax.set(
        ylabel="backscatter / a.u.",
        xlabel="time / h",
        ylim=(25, 32),
        xlim=(9.7, 10.3),
        yticks=[25, 28, 31],
    )
    ax.yaxis.label.set_color("green")
    ax.tick_params(axis="y", colors="green")

    # DO
    ax = ax2
    ax.plot(
        *bldata.get_timeseries("DO", "A01"),
        color="blue",
        marker="x",
    )
    ax.annotate(
        "trend deviation", color="blue",
        fontsize=8,
        xy=(10, 31.5), xytext=(10.05, 29), arrowprops=dict(color="blue", width=1, headwidth=5, headlength=5)
    )
    ax.set(
        ylabel="dissolved $\mathrm{O_2}$ / %",
        ylim=(25, 45)
    )
    ax.yaxis.label.set_color("blue")
    ax.tick_params(axis="y", colors="blue")

    fig.tight_layout()
    plotting.savefig(fig, "plot_raw_data_zoom", dp=wd)
    return


def plot_kinetics_with_insets(wd: pathlib.Path):
    idata = arviz.from_netcdf(wd / "full_posterior.nc")
    cm_biomass = models.get_biomass_model(wd)
    cm_glucose = models.get_glucose_model(wd)
    model = models.MonodModel()
    dataset = murefi.load_dataset(wd / "cultivation_dataset.h5")

    rid = "A01"
    theta_mapping = models.get_parameter_mapping(wd, [rid])
    replicate = dataset[rid]

    ipeak = numpy.argmax(replicate["Pahpshmir_1400_BS3_CgWT"].y)

    inferred_posteriors = {}
    ts: murefi.Timeseries = replicate['Pahpshmir_1400_BS3_CgWT']
    for i, y in fastprogress.progress_bar(list(enumerate(ts.y))):
        inferred_posteriors[(rid, i)] = cm_biomass.infer_independent(y, lower=0, upper=20, ci_prob=0.9)

    tmin, tmax = (9.95, 10.1)
    # Make a kinetics prediction only for the zoomed-in time range
    ds_prediction = plotting.predict_kinetics(
        idata=idata,
        model=model,
        theta_mapping=theta_mapping,
        subset={rid: replicate},
        predict_kwargs=dict(template={
            rid : murefi.Replicate.make_template(tmin, tmax, "SX", rid=rid, N=300)
        }),
    )

    fig, axs = pyplot.subplots(ncols=2, sharex=True, figsize=(12, 6), dpi=200)

    plotting.plot_kinetics(
        axs[0],
        idata,
        theta_mapping,
        model,
        {rid: replicate},
        cm_biomass,
        cm_glucose,
        inferred_posteriors,
        annotate=False,
        predict_kwargs=dict(template={
            rid : murefi.Replicate.make_template(0, replicate.t_max + 0.25, "SX", rid=rid, N=300)
        }),
        ax_glucose=axs[1],
        biomass_violins=False,
        violin_shrink=500,
        biomass_scatter=False,
    )

    ax = axs[0]
    ax.set(
        xlim=(0, 10.5),
        ylim=(0, None),
    )
    axi = ax.inset_axes([0.12, 0.47, 0.5, 0.5])
    axi.set(
        xlim=(tmin, tmax),
        ylim=(12.75, 13.25),
    )
    ax.indicate_inset_zoom(axi, edgecolor="black")
    ts: murefi.Timeseries = replicate['Pahpshmir_1400_BS3_CgWT']
    for i, (t, y) in fastprogress.progress_bar(list(enumerate(zip(ts.t, ts.y)))):
        if t > tmin and t < tmax:
            pst = inferred_posteriors[(rid, i)]
            pst = inferred_posteriors[(rid, i)]
            axi.scatter(t, pst.median, s=400, color="green", marker="_")
            axi.fill_betweenx(
                y=pst.hdi_x,
                x1=t - pst.hdi_pdf/500,
                x2=t + pst.hdi_pdf/500,
                color="green", alpha=0.5,
                edgecolor=None,
            )
    plotting.plot_density(
        ax=axi,
        x=ds_prediction["A01"]["X"].t,
        samples=ds_prediction["A01"]["X"].y,
        palette=pyplot.cm.Greens,
        plot_samples=True,
    )

    ax = axs[1]
    ax.set(
        ylim=(0, None),
    )
    axi = ax.inset_axes([0.15, 0.1, 0.47, 0.47])
    tmin, tmax = (9.95, 10.1)
    axi.set(
        xlim=(tmin, tmax),
        ylim=(0, 0.2),
    )
    ax.indicate_inset_zoom(axi, edgecolor="black")
    plotting.plot_density(
        ax=axi,
        x=ds_prediction["A01"]["S"].t,
        samples=ds_prediction["A01"]["S"].y,
        palette=pyplot.cm.Reds,
        plot_samples=True,
    )
    ts = replicate["A365"]
    assert len(ts.t) == 1
    t = ts.t[-1]
    pst = cm_glucose.infer_independent(ts.y, lower=0, upper=20, ci_prob=0.9)
    axi.scatter(t, pst.median, s=400, color="blue", marker="_")
    axi.fill_betweenx(
        y=pst.hdi_x,
        x1=t - pst.hdi_pdf/4000,
        x2=t + pst.hdi_pdf/4000,
        color="blue", alpha=1,
        edgecolor=None,
    )

    fig.tight_layout()
    plotting.savefig(fig, "plot_kinetics_with_insets", dp=wd)
    return
