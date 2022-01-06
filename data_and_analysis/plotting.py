import fastprogress
import matplotlib
from matplotlib import cm, colors, pyplot
import numpy
from operator import itemgetter
import pathlib
import matplotlib.image as mpimg
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition, mark_inset
import scipy
import string
import typing
import arviz
try:
    import pymc as pm
except ModuleNotFoundError:
    import pymc3 as pm


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

DP_FIGURES = pathlib.Path(__file__).parent / "figures"
DP_FIGURES.mkdir(exist_ok=True)

matplotlib.rcParams.update(params)


def savefig(fig, name: str, *, dp: pathlib.Path=DP_FIGURES, facecolor="white", **kwargs):
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


def plot_glucose_cmodels(fn_out=None, *, residual_type="relative"):
    model_lin = models.get_glucose_model_linear()
    model_asym = models.get_glucose_model()
    X = model_asym.cal_independent
    Y = model_asym.cal_dependent
    fig, axs = pyplot.subplots(
        nrows=2, ncols=3, figsize=(16, 10), dpi=120, sharex="col", sharey="col"
    )
    calibr8.plot_model(model_lin, fig=fig, axs=axs[0, :], residual_type=residual_type)
    calibr8.plot_model(model_asym, fig=fig, axs=axs[1, :], residual_type=residual_type)
    for i, row in enumerate(axs):
        for j, ax in enumerate(row):
            if i == 1:
                ax.set_xlabel("glucose concentration [g/L]")
            else:
                ax.set_xlabel("")
            ax.scatter([], [], label="calibration data", color="#1f77b4")
            ax.plot([], [], label="$\mu_\mathrm{dependent}$", color="green")
            if j in [0, 1]:
                ax.set_ylabel("absorbance$_{365 \mathrm{nm}}$ [a.u.]")
                if i == 0:
                    ax.scatter(
                        X[X > 20],
                        Y[X > 20],
                        color="#1f77b4",
                        marker="x",
                        label="calibration data (ignored)",
                        s=25,
                    )
            ax.text(
                0.03,
                0.93,
                string.ascii_uppercase[j + i * len(row)],
                transform=ax.transAxes,
                size=20,
                weight="bold",
            )
    handles, labels = axs[0, 0].get_legend_handles_labels()
    handles2, labels2 = (
        (itemgetter(1, 2, 3, 0, -2, -1)(handles)),
        (itemgetter(1, 2, 3, 0, -2, -1)(labels)),
    )
    axs[0, 0].legend(handles2, labels2, loc="lower right")
    if residual_type == "relative":
        axs[0, 2].set_ylim(-0.1, 0.2)
        axs[1, 2].set_ylim(-0.1, 0.2)
    pyplot.tight_layout()
    if fn_out:
        savefig(fig, fn_out)


def plot_biomass_cmodel(fn_out=None, *, residual_type="relative"):
    btm_model = models.get_biomass_model()
    fig, axs = pyplot.subplots(nrows=1, ncols=3, figsize=(16, 6.5), dpi=120)
    calibr8.plot_model(btm_model, fig=fig, axs=axs, residual_type=residual_type)
    for i, ax in enumerate(axs):
        ax.set_xlabel("biomass concentration [g/L]")
        ax.scatter([], [], label="calibration data", color="#1f77b4")
        if i in [0, 1]:
            ax.set_ylabel("backscatter [a.u.]")
        ax.plot([], [], label="$\mu_\mathrm{dependent}$", color="green")
        # if i in [0,1]:
        #   ax.set_ylabel('absorbance$_{365 \mathrm{nm}}$ [a.u.]')
        ax.text(
            0.03,
            0.93,
            string.ascii_uppercase[i],
            transform=ax.transAxes,
            size=20,
            weight="bold",
        )
    handles, labels = axs[0].get_legend_handles_labels()
    handles2, labels2 = (
        (itemgetter(1, 2, 3, 0, -1)(handles)),
        (itemgetter(1, 2, 3, 0, -1)(labels)),
    )
    axs[0].legend(handles2, labels2, loc="lower right")
    pyplot.tight_layout()
    if fn_out:
        savefig(fig, fn_out)


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


def plot_residuals_pp(
    ax,
    cmodel,
    tsobs: murefi.Timeseries,
    tspred: murefi.Timeseries,
    *,
    color,
    palette,
    tspred_extra: typing.Optional[murefi.Timeseries] = None,
):
    assert isinstance(cmodel, calibr8.BaseModelT)
    numpy.testing.assert_array_equal(tspred.t, tsobs.t)

    # for each of the 9000 posterior samples, draw 1 observation
    mu, scale, df = cmodel.predict_dependent(tspred.y)
    ppbs = scipy.stats.t.rvs(loc=mu, scale=scale, df=df)
    median = numpy.median(ppbs, axis=0)

    if tspred_extra is not None:
        # tspred_extra may be used to plot a higher resolution or extrapolated density
        mu, scale, df = cmodel.predict_dependent(tspred_extra.y)
        ppbs_extra = scipy.stats.t.rvs(loc=mu, scale=scale, df=df)
        pm.gp.util.plot_gp_dist(
            ax=ax,
            x=tspred_extra.t,
            samples=ppbs_extra - numpy.median(ppbs_extra, axis=0),
            palette=palette,
            plot_samples=False,
        )
    else:
        # plot the density from the data-like prediction
        pm.gp.util.plot_gp_dist(
            ax=ax, x=tsobs.t, samples=ppbs - median, palette=palette, plot_samples=False
        )
    yres = tsobs.y - median
    ax.scatter(
        tsobs.t,
        yres,
        marker="x",
        color=color,
    )
    return numpy.abs(yres).max()


def plot_mle(
    *,
    cm_biomass: calibr8.CalibrationModel,
    cm_glucose: calibr8.CalibrationModel,
    dataset: murefi.Dataset,
    prediction: murefi.Dataset,
):
    fig, (left, right) = pyplot.subplots(
        dpi=120,
        figsize=(12, 6),
        ncols=2,
        sharex=True
    )
    left2 = left.twinx()

    # ___________________________________Inset plots___________________________________#

    # Create a set of inset Axes: these should fill the bounding box allocated to
    # them.
    ax_inset = pyplot.axes([0, 0, 1, 1])
    # Manually set the position and relative size of the inset axes within ax1
    ip = InsetPosition(right, [0.63, 0.14, 0.33, 0.45])
    ax_inset.set_axes_locator(ip)
    # inset plot for left plot (FP)
    ax_inset2 = pyplot.axes([1, 1, 0, 0])
    # Manually set the position and relative size of the inset axes within ax
    ip = InsetPosition(left, [0.63, 0.14, 0.30, 0.4])
    ax_inset2.set_axes_locator(ip)
    # _________________________________END: Inset plots_________________________________#

    # plot the Flowerplate icon
    img = mpimg.imread(DP_FIGURES / "4.2.2 FlowerPlate_named.png")
    ax_inset2.imshow(img[::-1, ...])
    ax_inset2.xaxis.set_visible(False)
    ax_inset2.yaxis.set_visible(False)
    ax_inset2.spines["bottom"].set_visible(False)
    ax_inset2.spines["top"].set_visible(False)
    ax_inset2.spines["right"].set_visible(False)
    ax_inset2.spines["left"].set_visible(False)

    # plot the data
    for i, (rid, rep) in enumerate(dataset.items()):
        color = cm.tab10("ABCDEF".index(rid[0]) / 10)

        # X
        ts_t = rep[cm_biomass.dependent_key].t
        ts_y = rep[cm_biomass.dependent_key].y
        left.scatter(
            ts_t,
            ts_y,
            marker="x",
            s=1,
            color=color,
        )
        x, y = ts_t[-1], ts_y[-1]

        ## annotate X
        if x < 7.2:
            x_offset = 0
            y_offset = 1.7

        elif 7.2 < x < 9.2:
            x_offset = -1.7
            y_offset = -0.52

        else:
            x_offset = 0
            if i % 2 == 0:
                y_offset = -2.4
            else:
                y_offset = -3.6
        left.annotate(
            rid,
            xy=(x, y),
            xytext=(x + x_offset, y + y_offset),
            arrowprops=dict(arrowstyle="-|>", facecolor=color, edgecolor=color),
            horizontalalignment="center",
            fontsize=9.5,
            rotation=-45,
            color=color,
        )
        # CDW transformation
        cdw = cm_biomass.predict_independent(ts_y)
        right.scatter(ts_t, cdw, marker="x", s=1, color=color)
        ## inset X
        mask = numpy.logical_and(ts_t > 6.0, ts_t < 7.2)
        ax_inset.scatter(ts_t[mask], cdw[mask], marker="x", s=1, color=color)

        # S
        ts_t = rep[cm_glucose.dependent_key].t
        ts_y = rep[cm_glucose.dependent_key].y
        left2.scatter(
            ts_t,
            ts_y,
            marker="x",
            s=20,
            color=color,
        )
        # concentrations
        glc = cm_glucose.predict_independent(ts_y)
        right.scatter(ts_t, glc, marker="x", s=20, color=color)

        # pred-X
        ts_t = prediction[rid]["X"].t
        ts_y = prediction[rid]["X"].y
        right.plot(ts_t, ts_y, color=color, linestyle="-", lw=0.5)

        ## inset pred X
        mask = numpy.logical_and(ts_t > 6.0, ts_t < 7.2)
        ax_inset.plot(ts_t[mask], ts_y[mask], linestyle="-", lw=0.5, color=color)

        # pred-S
        ts_t = prediction[rid]["S"].t
        ts_y = prediction[rid]["S"].y
        right.plot(ts_t, ts_y, color=color, linestyle=":")
    right.annotate(
        "global $S_0$",
        xy=(0, 17),
        xytext=(2, 18),
        arrowprops=dict(arrowstyle="-|>", facecolor="k", edgecolor="k"),
        horizontalalignment="left",
        fontsize=12,
        color="k",
    )
    # Make correct data labels
    right.plot([], [], color="r", linestyle=":", label=r"MLE$_{\mathrm{glucose}}$")
    right.plot([], [], color="r", linestyle="-", label=r"MLE$_{\mathrm{biomass}}$")
    left.scatter([], [], marker="x", s=20, color="r", label="glucose (absorbance)")
    left.scatter([], [], marker="x", s=1, color="r", label="biomass (backscatter)")

    # Set axis labels and lims
    left.set_ylabel("backscatter [a.u]")
    left.set_xlabel("time [h]")
    left2.set_ylabel(r"absorbance$_{\mathrm{365nm}}$")
    right.set_ylabel("concentration [g/L]")
    right.set_xlabel("time [h]")
    right.set_xlim(-1.3, 18)
    right.set_ylim(-1, 19)
    left.set_ylim(0, 26)
    left2.set_ylim(0, 1.8)
    ax_inset.set_ylim(3.4, 5.2)
    ax_inset.set_xlim(6.4, 7.1)

    # Make legends
    legend1 = left.legend(loc="upper right")
    pyplot.setp(legend1.get_texts(), multialignment="center")
    right.legend()

    # Inset plot box
    # Mark the region corresponding to the inset axes on right and draw lines
    # in grey linking the two axes.
    mark_inset(
        right, ax_inset, loc1=2, loc2=3, fc="none", ec="0.5", zorder=1, linestyle="dashed"
    )
    pyplot.setp(ax_inset.get_xticklabels(), backgroundcolor="w")
    pyplot.setp(ax_inset.get_yticklabels(), backgroundcolor="w")
    ax_inset.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

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
    t = replicate["Pahpshmir_1400_BS3_CgWT"].t[-1]
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
