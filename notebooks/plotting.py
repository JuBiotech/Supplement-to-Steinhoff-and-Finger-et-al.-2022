import fastprogress
import matplotlib
from matplotlib import cm, colors, pyplot
import numpy
import pathlib
import arviz
import pymc as pm


import calibr8
import murefi

import models


params = {
    "text.latex.preamble": "\\usepackage{gensymb}",
    "image.origin": "lower",
    "image.interpolation": "nearest",
    "axes.grid": False,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "font.size": 16,
    "legend.fontsize": 11,
    "legend.frameon": False,
    "legend.fancybox": False,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Noteworthy", "DejaVu Sans", "Lucida Grande", "Verdana"],
    "lines.markersize": 3,
}

DP_ROOT = pathlib.Path(__file__).absolute().parent.parent
DP_RESULTS = DP_ROOT / "results"
DP_RESULTS.mkdir(exist_ok=True)

matplotlib.rcParams.update(params)


def savefig(fig, name: str, *, dp: pathlib.Path=DP_RESULTS, facecolor="white", **kwargs):
    """Saves a bitmapped and vector version of the figure.

    Parameters
    ----------
    fig
        The figure object.
    name : str
        Filename without extension.
    dp : pathlib.Path
        Target directory. Defaults to the "figures" subfolder next to this script.
    **kwargs
        Additional kwargs for `pyplot.savefig`.
    """
    if not "facecolor" in kwargs:
        kwargs["facecolor"] = facecolor
    max_pixels = numpy.array([2250, 2625])
    max_dpi = min(max_pixels / fig.get_size_inches())
    if not "dpi" in kwargs:
        kwargs["dpi"] = max_dpi
    fig.savefig(dp / f"{name}.png", **kwargs)
    fig.savefig(dp / f"{name}.pdf", **kwargs)
    # Save with & without border to measure the "shrink".
    # This is needed to rescale the dpi setting such that we get max pixels also without the border.
    tkwargs = dict(
        pil_kwargs={"compression": "tiff_lzw"},
        bbox_inches="tight",
        pad_inches=0.01,
    )
    tkwargs.update(kwargs)
    fp = str(dp / f"{name}.tif")
    fig.savefig(fp, **tkwargs)
    # Measure the size
    actual = numpy.array(pyplot.imread(fp).shape[:2][::-1])
    tkwargs["dpi"] = int(tkwargs["dpi"] * min(max_pixels / actual))
    fig.savefig(fp, **tkwargs)
    return


def to_colormap(dark):
    N = 256
    dark = numpy.array((*dark[:3], 1))
    white = numpy.ones(4)
    cvals = numpy.array([
        (1 - n) * white + n * dark
        for n in numpy.linspace(0, 1, N)
    ])
    # add transparency
    cvals[:, 3] = numpy.linspace(0, 1, N)
    return colors.ListedColormap(cvals)


def transparentify(cmap: colors.Colormap) -> colors.ListedColormap:
    """Creates a transparent->color version from a standard colormap.
    
    Stolen from https://stackoverflow.com/a/37334212/4473230
    
    Testing
    -------
    x = numpy.arange(256)
    fig, ax = pyplot.subplots(figsize=(12,1))
    ax.scatter(x, numpy.ones_like(x) - 0.01, s=100, c=[
        cm.Reds(v)
        for v in x
    ])
    ax.scatter(x, numpy.ones_like(x) + 0.01, s=100, c=[
        redsT(v)
        for v in x
    ])
    ax.set_ylim(0.9, 1.1)
    pyplot.show()
    """
    # Get the colormap colors
    #cm_new = numpy.zeros((256, 4))
    #cm_new[:, :3] = numpy.array(cmap(cmap.N))[:3]
    cm_new = numpy.array(cmap(numpy.arange(cmap.N)))
    cm_new[:, 3] = numpy.linspace(0, 1, cmap.N)
    return colors.ListedColormap(cm_new)


redsT = transparentify(cm.Reds)
greensT = transparentify(cm.Greens)
bluesT = transparentify(cm.Blues)
orangesT = transparentify(cm.Oranges)
greysT = transparentify(cm.Greys)


class FZcolors:
    red = numpy.array((191, 21, 33)) / 255
    green = numpy.array((0, 153, 102)) / 255
    blue = numpy.array((2, 61, 107)) / 255
    orange = numpy.array((220, 110, 0)) / 255


class FZcmaps:
    red = to_colormap(FZcolors.red)
    green = to_colormap(FZcolors.green)
    blue = to_colormap(FZcolors.blue)
    orange = to_colormap(FZcolors.orange)
    black = transparentify(cm.Greys)


def extract_parameters(
    idata: arviz.InferenceData,
    theta_mapping: murefi.ParameterMapping,
    nmax: int = 1_000,
):
    posterior = idata.posterior.stack(sample=("chain", "draw"))
    parameters = {}
    nsamples = idata.posterior.dims["chain"] * idata.posterior.dims["draw"]
    nmax = min(nmax, nsamples)
    # randomly shuffle the samples using the following indices
    idxrnd = numpy.random.permutation(numpy.arange(nmax))
    for pname, pkind in theta_mapping.parameters.items():
        with_coords = pname != pkind
        if with_coords:
            coord = tuple(posterior[pkind].coords.keys())[0]
            pvals = posterior[pkind].sel({coord: pname.replace(f"{pkind}_", "")}).values
        else:
            pvals = posterior[pname].values
        parameters[pname] = pvals[idxrnd]
    return parameters


def plot_mle(
    *,
    cm_biomass: calibr8.CalibrationModel,
    cm_glucose: calibr8.CalibrationModel,
    dataset: murefi.Dataset,
    prediction: murefi.Dataset,
):
    for rid, rep in prediction.items():
        for dk, ts in rep.items():
            ts.y = numpy.clip(ts.y, 0, numpy.inf)

    fig, (left, right) = pyplot.subplots(
        dpi=120,
        figsize=(12, 6),
        ncols=2,
        sharex=True
    )
    left2 = left.twinx()

    # plot the data
    for i, (rid, rep) in enumerate(dataset.items()):
        # X
        ts_t = rep[cm_biomass.dependent_key].t
        ts_y = rep[cm_biomass.dependent_key].y
        left.scatter(
            ts_t,
            ts_y,
            marker="x",
            s=1,
            color="green",
        )
        x, y = ts_t[-1], ts_y[-1]

        # CDW transformation
        cdw = cm_biomass.predict_independent(ts_y)
        right.scatter(ts_t, cdw, marker="x", s=1, color="green")

        # S
        if cm_glucose.dependent_key in rep:
            ts_t = rep[cm_glucose.dependent_key].t
            ts_y = rep[cm_glucose.dependent_key].y
            left2.scatter(
                ts_t,
                ts_y,
                marker="x",
                s=20,
                color="blue",
            )
            # concentrations
            glc = cm_glucose.predict_independent(ts_y)
            right.scatter(ts_t, glc, marker="x", s=20, color="blue")

        # pred-X
        ts_t = prediction[rid]["X"].t
        ts_y = prediction[rid]["X"].y
        right.plot(ts_t, ts_y, color="green", linestyle="-", lw=0.5)

        # pred-S
        ts_t = prediction[rid]["S"].t
        ts_y = prediction[rid]["S"].y
        right.plot(ts_t, ts_y, color="blue", linestyle=":")
    # Make correct data labels
    left.scatter([], [], marker="x", s=20, color="blue", label="simulated $A_\mathrm{365}$ observation (glucose)")
    left.scatter([], [], marker="x", s=1, color="green", label="observed backscatter (biomass)")
    right.plot([], [], color="blue", linestyle=":", label="modeled glucose")
    right.plot([], [], color="green", linestyle="-", label="modeled biomass")
    right.scatter([], [], marker="x", s=20, color="blue", label="inferred glucose")
    right.scatter([], [], marker="x", s=1, color="green", label="inferred biomass")

    # Set axis labels and lims
    left.set_ylabel("backscatter   [a.u]")
    left.set_xlabel("time   [h]")
    left2.set_ylabel(r"A$_{\mathrm{365nm}}$   [-]")
    right.set_ylabel("concentration   [g/L]")
    right.set_xlabel("time [h]")
    right.set_xlim(None, None)
    right.set_ylim(0, 21)
    left.set_ylim(0, 40)
    left2.set_ylim(0, 1.8)

    # Make legends
    legend1 = left.legend(loc="upper right")
    pyplot.setp(legend1.get_texts(), multialignment="center")
    right.legend()

    # Mark subplots with A, B, ...
    for i, ax in enumerate((left, right)):
        ax.text(0.03, 0.93, "AB"[i], transform=ax.transAxes, size=20, weight="bold")


    pyplot.tight_layout()
    pyplot.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.45, hspace=None)
    return fig


def hdi_ticklimits(axs, idata, replacements, ylabelpad=-50, xlabelpad=-15, xrotate=False):
    posterior = idata.posterior.stack(sample=("chain", "draw"))
    hdi90 = arviz.hdi(idata, hdi_prob=0.9)
    hdi98 = arviz.hdi(idata, hdi_prob=0.98)
    inv_replacements = {val: key for key, val in replacements.items()}
    def _edit(axs, which: str):
        get_label = getattr(axs[0], f"get_{which}label")
    
        label = get_label()
        rvname = label.split("\n")[0]
        rvname = inv_replacements.get(rvname, rvname)
        rv = posterior[rvname]
        sel = {}
        if "\n" in label:
            for cname, cval in zip(rv.coords, label.split("\n")[1:]):
                # TODO: typecast cval to match the coord type
                sel[cname] = cval
        limits = hdi98[rvname].sel(sel).values
        hdi = hdi90[rvname].sel(sel).values
        # Round the HDI such that it differs in the last 2 decimals.
        # This way the label looks nice but is only positioned wrong by <1%
        # of the plot width.
        sigdigits = max(0, numpy.ceil(numpy.log10(1 / numpy.abs(hdi[1] - hdi[0]))) + 1)
        hdi = numpy.round(hdi, decimals=int(sigdigits))
        for i, ax in enumerate(axs):
            if i == 0:
                get_label = getattr(ax, f"get_{which}label")
                set_label = getattr(ax, f"set_{which}label")
                set_label(get_label(), labelpad=ylabelpad if which == "y" else xlabelpad)
            ax.set(**{f"{which}lim": limits, f"{which}ticks": hdi})
            if xrotate and which == "x":
                ax.tick_params("x", labelrotation=90)

    for r, axr in enumerate(axs[:]):
        _edit(axr, which="y")
    for c, axc in enumerate(axs[:, :].T):
        _edit(axc[::-1], which="x")
    # move labels back
    fig = pyplot.gcf()
    fig.align_ylabels()
    fig.align_xlabels()
    return


def plot_kinetics(
    ax,
    idata_full,
    theta_mapping,
    model,
    subset,
    cm_biomass,
    cm_glucose,
    inferred_posteriors,
    *,
    annotate=True,
    predict_kwargs,
    ax_glucose=None,
    biomass_violins=False,
    violin_shrink=20,
    biomass_scatter=True,
):
    # Default glucose to the same axis as biomass
    axb = ax
    axg = (ax_glucose or ax)

    parameters_sample = extract_parameters(idata_full, theta_mapping, nmax=1000)

    if not predict_kwargs:
        predict_kwargs = {}
    predict_kwargs.setdefault("template", murefi.Dataset.make_template_like(subset, independent_keys='SX'))
    predict_kwargs.setdefault("parameter_mapping", theta_mapping)
    predict_kwargs.setdefault("parameters", parameters_sample)
    ds_prediction = model.predict_dataset(**predict_kwargs)

    red = cm.Reds(0.9)
    green = cm.Greens(0.9)

    axmap = {
        "X": axb,
        "S": axg,
    }

    for rid, rep in ds_prediction.items():
        for ikey, ts in rep.items():
            pm.gp.util.plot_gp_dist(
                ax=axmap[ts.independent_key],
                x=ts.t,
                samples=ts.y,
                palette='Reds' if ikey == 'S' else 'Greens',
                plot_samples=True
            )

    for r, (rid, replicate) in enumerate(subset.items()):
        # biomass
        ts = replicate['Pahpshmir_1400_BS3_CgWT']
        for i, (t, y) in fastprogress.progress_bar(list(enumerate(zip(ts.t, ts.y)))):
            if (rid, i) in inferred_posteriors:
                pst = inferred_posteriors[(rid, i)]
                if biomass_scatter:
                    axb.scatter(t, pst.median, s=400, color=green, marker="_")
                if biomass_violins:
                    axb.fill_betweenx(
                        y=pst.hdi_x,
                        x1=t - pst.hdi_pdf/violin_shrink,
                        x2=t + pst.hdi_pdf/violin_shrink,
                        color=green, alpha=0.5,
                        edgecolor=None,
                    )
                else:
                    axb.plot(
                        [t, t],
                        [pst.hdi_lower, pst.hdi_upper],
                        color=green, alpha=0.5, linewidth=0.5
                    )

        # glucose
        if "A365" in replicate:
            ts = replicate['A365']
            assert len(ts.t) == 1
            t = ts.t[-1]
            pst = cm_glucose.infer_independent(ts.y, lower=0, upper=20, ci_prob=0.9)
            axg.scatter(t, pst.median, s=400, color=red, marker="_")
            axg.fill_betweenx(
                y=pst.hdi_x,
                x1=t - pst.hdi_pdf/violin_shrink,
                x2=t + pst.hdi_pdf/violin_shrink,
                color=red, alpha=0.5,
                edgecolor=None,
            )
    # annotations
    if annotate:
        for rid, replicate in subset.items():
            x = replicate['Pahpshmir_1400_BS3_CgWT'].t[-1]
            y = replicate['Pahpshmir_1400_BS3_CgWT'].y[-1]
            y = cm_biomass.predict_independent(replicate['Pahpshmir_1400_BS3_CgWT'].y[-1])
            
            if not numpy.isfinite(y):
                y = 0
            axg.annotate(
                rid, xy=(x, y), xytext=(x, y+1.5), 
                arrowprops=dict(arrowstyle='-|>', facecolor='black'),
                horizontalalignment='center',
            )

    axb.set_ylabel('concentration   [g/L]')
    axb.set_xlabel('time   [h]')
    if ax_glucose:
        axb.set_ylabel('biomass   [g/L]')
        axg.set_ylabel('glucose   [g/L]')
        axg.set_xlabel('time   [h]')
    return ds_prediction


def plot_ks_curvature(
    idata,
    theta_mapping,
    model,
    replicate: murefi.Replicate,
    cm_biomass,
    cm_glucose,
):
    rid = replicate.rid
    ipeak = numpy.argmax(replicate["Pahpshmir_1400_BS3_CgWT"].y)
    t = replicate["Pahpshmir_1400_BS3_CgWT"].t[ipeak]
    tmin, tmax = (t - 0.2, t + 0.15)
    del t

    inferred_posteriors = {}
    ts = replicate['Pahpshmir_1400_BS3_CgWT']
    for i, (t, y) in fastprogress.progress_bar(list(enumerate(zip(ts.t, ts.y)))):
        if t > tmin and t < tmax:
            inferred_posteriors[(rid, i)] = cm_biomass.infer_independent(y, lower=0, upper=20, ci_prob=0.9)

    fig, (axb, axg) = pyplot.subplots(ncols=2, sharex=True, figsize=(12, 6), dpi=200)

    template = {
        rid : murefi.Replicate.make_template(tmin, tmax, "SX", rid=rid, N=400)
    }
    ds_prediction = plot_kinetics(
        axb,
        idata,
        theta_mapping,
        model,
        {rid: replicate},
        cm_biomass,
        cm_glucose,
        inferred_posteriors,
        annotate=False,
        predict_kwargs=dict(template=template),
        ax_glucose=axg,
        biomass_violins=True,
        violin_shrink=100,
    )

    y = ds_prediction[rid]["X"].y
    ymin, ymax = numpy.percentile(y, [5, 95])
    ymin = ymin - 0.2
    ymax = ymax + 0.2

    axb.set(
        xlim=(tmin, tmax),
        ylim=(ymin, ymax),
    )
    axg.set(
        ylim=(0, None),
    )
    fig.tight_layout()

    return fig, (axb, axg)


def plot_full_kinetics(
    idata,
    theta_mapping,
    model,
    replicate: murefi.Replicate,
    cm_biomass,
    cm_glucose,
):
    rid = replicate.rid
    ipeak = numpy.argmax(replicate["Pahpshmir_1400_BS3_CgWT"].y)

    inferred_posteriors = {}
    ts: murefi.Timeseries = replicate['Pahpshmir_1400_BS3_CgWT']
    for i, y in fastprogress.progress_bar(list(enumerate(ts.y))):
        inferred_posteriors[(rid, i)] = cm_biomass.infer_independent(y, lower=0, upper=20, ci_prob=0.9)

    fig, (axb, axg) = pyplot.subplots(ncols=2, sharex=True, figsize=(12, 6), dpi=200)

    template = {
        rid : murefi.Replicate.make_template(0, replicate.t_max, "SX", rid=rid, N=400)
    }
    ds_prediction = plot_kinetics(
        axb,
        idata,
        theta_mapping,
        model,
        {rid: replicate},
        cm_biomass,
        cm_glucose,
        inferred_posteriors,
        annotate=False,
        predict_kwargs=dict(template=template),
        ax_glucose=axg,
        biomass_violins=False,
        violin_shrink=500,
        biomass_scatter=False,
    )

    y = ds_prediction[rid]["X"].y
    ymin, ymax = numpy.percentile(y, [5, 95])
    ymin = ymin - 0.2
    ymax = ymax + 0.2

    axb.set(
        #xlim=(tmin, tmax),
        ylim=(0, None),
    )
    axg.set(
        ylim=(0, None),
    )
    fig.tight_layout()

    return fig, (axb, axg)
