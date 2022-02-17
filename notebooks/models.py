from typing import Iterable, Optional
import numpy
import pathlib
import pandas
import pathlib

import calibr8
import murefi

DP_ROOT = pathlib.Path(__file__).absolute().parent.parent
DP_DATA = DP_ROOT / "data"


class LogisticGlucoseCalibrationModelV1(calibr8.BaseAsymmetricLogisticT):
    def __init__(self, *, independent_key:str='S', dependent_key:str='A365'):
        super().__init__(
            independent_key=independent_key, 
            dependent_key=dependent_key, 
            scale_degree=1
        )


class BLProCDWBackscatterModelV1(calibr8.BaseLogIndependentAsymmetricLogisticT):
    def __init__(self, *, independent_key:str='X', dependent_key:str='Pahpshmir_1400_BS3_CgWT'):
        super().__init__(independent_key=independent_key, dependent_key=dependent_key, scale_degree=1)


def get_biomass_model(dp=DP_DATA) -> BLProCDWBackscatterModelV1:
    return BLProCDWBackscatterModelV1.load(dp / "biomass.json")


def get_glucose_model(dp=DP_DATA) -> LogisticGlucoseCalibrationModelV1:
    return LogisticGlucoseCalibrationModelV1.load(dp / "glucose.json")


class MonodModel(murefi.BaseODEModel):
    """ Class specifying the model for parameter fitting as Monod kinetics. """

    def __init__(self):
        super().__init__(parameter_names=('S0', 'X0', 'mu_max', 'K_S', 'Y_XS'), independent_keys=['S', 'X'])

    def dydt(self, y, t, theta):
        """First derivative of the transient variables.
        Args:
            y (array): array of observables
            t (float): time since intial state
            theta (array): Monod parameters
        Returns:
            array: change in y at time t
        """
        # NOTE: this method has significant performance impact!
        S, X = y
        mu_max, K_S, Y_XS = theta
        dXdt = mu_max * S * X / (K_S + S)
    
        yprime = [
            -1/Y_XS * dXdt,
            dXdt,
        ]
        return yprime


def get_parameter_mapping(dp=DP_DATA, rids: Optional[Iterable[str]]=None) -> murefi.ParameterMapping:
    df_mapping = pandas.read_excel(dp / "full_parameter_mapping.xlsx", index_col=0)
    df_mapping.index.name = "rid"
    if rids:
        df_mapping = df_mapping.loc[list(rids)]
    theta_mapping = murefi.ParameterMapping(
        df_mapping,
        bounds={
            'S0': (18, 22),
            'X0': (0.01, 1),
            'mu_max': (0.4, 0.5),
            'Y_XS': (0.3, 1),
            'K_S': (1e-7, numpy.inf),
        },
        guesses={
            'S0': 20,
            'X0': 0.25,
            'mu_max': 0.42,
            'Y_XS': 0.6,
            'K_S': 0.02,
        }
    )
    return theta_mapping
