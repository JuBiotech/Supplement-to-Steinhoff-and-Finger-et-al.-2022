"""
The code in this file is used to transform experimental data from
their original data formatats to Excel sheets.

This transformation is performed with our internal packages `bletl` and `retl`.
"""
import os
import logging
import numpy
import pandas
import pathlib
import scipy
import scipy.stats

import bletl
import retl

import murefi

DP_RAW = pathlib.Path(__file__).parent / "raw_data"
_log = logging.getLogger(__file__)


def read_glucose_x_y(
    fp_dilutions: os.PathLike, fp_rdata: os.PathLike, stock_concentration: float
):
    """Extracts X (concentration factor or glucose) and Y (absorbance reading.

    Parameters
    ----------
    fp_dilutions : path-like
        Filepath to an Excel sheet with dilutions factors sorted in rows A-H and columns 1-12.
    fp_rdata : path-like
        Filepath to the plate reader XML output.
    stock_concentration : float
        Concentration of glucose stock solution.

    Returns
    -------
    X : numpy.ndarray
        Glucose concentration, 1-dimensional.
    Y : numpy.ndarray
        Absorbance readings, 1-dimensional.
    """

    X = (
        pandas.read_excel(fp_dilutions, index_col=0).values.flatten()
        * stock_concentration
    )
    rdata = retl.parse(fp_rdata)
    Y = numpy.array(rdata["A365"].value.loc[1]).reshape(8, 12).flatten()

    return X, Y


def read_biomass_dilution_factors(fp_xlsx: os.PathLike) -> pandas.DataFrame:
    """Parses the Excel-Sheet of dilution factors in plate layout into a long-form DataFrame.

    Parameters
    ----------
    fp_xlsx : path-like
        Filepath to an Excel sheet with rows A-F and columns 1-8.

    Returns
    -------
    df : pandas.DataFrame
        Hass well IDs as the index and one column "value" indicating the concentration factor.
    """
    df = pandas.read_excel(fp_xlsx).set_index("Unnamed: 0")
    df.index.name = "row"
    df = df.reset_index().melt(id_vars=["row"])
    df["well"] = [f"{row.row}{int(row.variable):02d}" for row in df.itertuples()]
    return df[["well", "value"]].set_index("well")


def read_biomass_stock_concentration(
    fp_before: os.PathLike,
    fp_after: os.PathLike,
    *,
    eppi_from: int,
    eppi_to: int,
    v_eppi: float = 800,
):
    """Parses CSV files of before and after Eppi weights and summarizes them into a biomass concentation.

    Parameters
    ----------
    fp_before : str
        Path to tab-separated CSV of weights before addition of biomass.
    fp_after : str
        Path to tab-separated CSV of weights after addition of biomass.
    eppi_from : int
        Eppi number of the first replicate.
    eppi_to : int
        Eppi number of the last replicate.
    v_eppi : float
        Volume of biomass [µL] that was centrifuged into each tube.

    Returns
    -------
    mean : float
        Arithmetic mean of CDW over the replicates.
    sem : float
        Standard error of the mean.
    """
    df_before = pandas.read_csv(
        fp_before, sep="\t", decimal=",", header=None, names=["tube", "weight"]
    ).set_index("tube")
    df_after = pandas.read_csv(
        fp_after, sep="\t", decimal=",", header=None, names=["tube", "weight"]
    ).set_index("tube")
    cdw = (df_after - df_before).weight
    stock_cdw = cdw.loc[eppi_from:eppi_to] / (v_eppi / 1000 / 1000)
    stock_mean = numpy.mean(stock_cdw)
    stock_sem = scipy.stats.sem(stock_cdw)
    return stock_mean, stock_sem


def read_biomass_x_and_y(
    fp_bldata: os.PathLike,
    df_dilutions: pandas.DataFrame,
    rpm: int,
    filterset: str,
    stock_concentration: float = 1,
):
    """Extracts X (concentration factor or CDW) and Y (backscatter reading) for a given RPM.

    Parameters
    ----------
    fp_bldata : path-like
        Path to the BioLector CSV with calibration measurements.
    df_dilutions : pandas.DataFrame
        Concentration factors for each well (see `read_biomass_dilution_factors`).
    rpm : int
        RPM of interest.
    stock_concentration : float
        CDW at dilution factor 1 (stock concentration).

    Returns
    -------
    X : numpy.ndarray
        Biomass concentration (factors), 1-dimensional.
    Y : numpy.ndarray
        Backscatter readings, 1-dimensional.
    """
    bldata = bletl.parse(fp_bldata)
    # parse the system comments to find out start and end times for each rpm
    df_slicing = pandas.DataFrame(columns=["rpm", "t_start", "t_end"]).set_index("rpm")
    if bldata.model in [bletl.BioLectorModel.BL1]:
        rpms = list(bldata.environment.shaker_setpoint.unique())
        setpoints = list(bldata.environment.shaker_setpoint)
        t_starts = [bldata.environment.time[setpoints.index(rpm)] for rpm in rpms]
    else:
        rpms = [
            int(row.sys_comment.strip(" rpm").strip("SET SHAKER SPEED TO "))
            for row in bldata.comments.itertuples()
            if "rpm" in row.sys_comment
        ]
        t_starts = [
            row.time for row in bldata.comments.itertuples() if "rpm" in row.sys_comment
        ]
    if len(rpms) == 1:
        rpms.append(rpms[-1])
        t_starts.append(bldata.environment.time.max())
    for rpm_, t_start, t_end in zip(rpms, t_starts, t_starts[1:]):
        df_slicing.loc[rpm_] = (t_start, t_end)

    t_start, t_end = df_slicing.loc[rpm]

    # aggregate
    X = []
    Y = []
    for row in df_dilutions.reset_index().itertuples():
        x, y = bldata[filterset].get_timeseries(row.well)
        mask = numpy.bitwise_and(t_start < x, x < t_end)
        y = y[mask]
        X += list(numpy.repeat(row.value, len(y)))
        Y += list(y)

    return numpy.array(X) * stock_concentration, numpy.array(Y)


def create_cultivation_dataset(
    with_pca=True, dkey_x="Pahpshmir_1400_BS3_CgWT", dkey_s="A365",
    trim_backscatter=False,
) -> murefi.Dataset:
    WELLS_WITH_PCA = [f"{r}{c:02d}" for r in "ABCD" for c in [2, 3, 4, 5, 6, 7, 8]]
    # The wells without PCA were not used for the analysis in the manuscript.
    WELLS_WITHOUT_PCA = [f"{r}{c:02d}" for r in "EF" for c in [2, 3, 4, 5, 6, 7, 8]]
    bldata = bletl.parse(DP_RAW / "8T1P5H_Ms000388.LG.csv")
    glcdata = retl.parse(DP_RAW / "8T1P5H_ReaderOutput.xml")

    df_inoculations = pandas.read_excel(
        DP_RAW / "8T1P5H_events.xlsx", sheet_name="inoculations", index_col=0
    ).set_index("fp_well")
    df_events = pandas.read_excel(
        DP_RAW / "8T1P5H_events.xlsx", sheet_name="sacrifices", index_col=0
    ).set_index("fp_well")
    # bundle into one dataframe
    df_events["inoculation_cycle"] = df_inoculations["cycle"]
    df_events["inoculation_time"] = df_inoculations["time"]
    df_events["glc_A365"] = [
        glcdata["A365"].value.loc[1, swell] for swell in df_events.supernatant_well
    ]
    df_events.head()

    assert len(set(df_events.inoculation_time.round(2))) == 1

    # transform into purely dict based data structure
    wells = WELLS_WITH_PCA if with_pca else WELLS_WITHOUT_PCA
    dataset = murefi.Dataset()
    for well in wells:
        X_t, X_y = bldata["BS3"].get_timeseries(
            well, last_cycle=df_events.loc[well, "cycle"]
        )
        # take only between inoculation & harvest
        X_t = X_t[df_events.loc[well, "inoculation_cycle"] :]
        X_t = tuple(X_t - X_t[0])
        X_y = tuple(X_y[df_events.loc[well, "inoculation_cycle"] :])

        # Optional: Take only up to the maximum
        if trim_backscatter:
            ipeak = numpy.argmax(X_y)
            ypeak = X_y[ipeak]
            if ypeak > 10:
                _log.info("Peak backscatter of %s in cycle %i at %f.", well, ipeak, ypeak)
                X_t = X_t[:ipeak + 1]
                X_y = X_y[:ipeak + 1]
            else:
                _log.info("Peak backscatter of %s in cycle %i at %f is too low to be the entry into stationary phase. NOT trimming.", well, ipeak, ypeak)

        # get timeseries for glucose
        S_t = tuple([X_t[-1]])
        S_y = tuple([df_events.loc[well, "glc_A365"]])

        # store into the dataset
        replicate = murefi.Replicate(well)
        replicate[dkey_x] = murefi.Timeseries(
            X_t, X_y, independent_key="X", dependent_key=dkey_x
        )
        replicate[dkey_s] = murefi.Timeseries(
            S_t, S_y, independent_key="S", dependent_key=dkey_s
        )
        dataset[well] = replicate

    return dataset
