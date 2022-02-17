"""
The code in this file is used to transform experimental data from
their original data formatats to Excel sheets.

This transformation is performed with our internal packages `bletl` and `retl`.
"""
import logging
import numpy
import pathlib

import bletl
import murefi

DP_ROOT = pathlib.Path(__file__).absolute().parent.parent
DP_DATA = DP_ROOT / "data"
_log = logging.getLogger(__file__)


def create_cultivation_dataset(
    *,
    fname_bldata: str,
    trim_backscatter=False,
    dkey_x="Pahpshmir_1400_BS3_CgWT",
    force_glucose_zero=False,
) -> murefi.Dataset:
    bldata = bletl.parse(DP_DATA / fname_bldata)
    dataset = murefi.Dataset()
    for well in bldata["BS3"].time.columns:
        X_t, X_y = bldata.get_timeseries("BS3", well)
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

        replicate = murefi.Replicate(well)
        replicate[dkey_x] = murefi.Timeseries(
            X_t, X_y, independent_key="X", dependent_key=dkey_x
        )
        if force_glucose_zero:
            replicate["A365"] = murefi.Timeseries(
                [X_t[-1]],
                # This value 👇 is the most likely observation of A365 according to our calibration
                [0.10961253928758818],
                independent_key="S",
                dependent_key="A365"
            )
        dataset[well] = replicate
    return dataset
