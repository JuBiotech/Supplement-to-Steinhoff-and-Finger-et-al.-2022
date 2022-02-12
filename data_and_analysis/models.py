import numpy
import pathlib
import pandas
import pathlib
import typing

import calibr8
import murefi

DP_PROCESSED = pathlib.Path(__file__).parent / "processed"


class LinearGlucoseCalibrationModelV1(calibr8.BasePolynomialModelT):
    def __init__(self, *, independent_key:str='S', dependent_key:str='A365'):
        super().__init__(
            independent_key=independent_key, 
            dependent_key=dependent_key, 
            mu_degree=1, 
            scale_degree=1
        )


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



def get_biomass_model(dp=DP_PROCESSED) -> BLProCDWBackscatterModelV1:
    try:
        import calibr8_contrib
        return calibr8_contrib.models.BLProCDWBackscatterModelV1.load(dp / "biomass.json")
    except calibr8.utils.CompatibilityException:
        return BLProCDWBackscatterModelV1.load(dp / "biomass.json")


def get_glucose_model(dp=DP_PROCESSED) -> LogisticGlucoseCalibrationModelV1:
    return LogisticGlucoseCalibrationModelV1.load(dp / "glucose_logistic.json")


def get_glucose_model_linear(dp=DP_PROCESSED) -> LinearGlucoseCalibrationModelV1:
    return LinearGlucoseCalibrationModelV1.load(dp / "glucose_linear.json")


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


def get_parameter_mapping(rids: typing.Optional[typing.Iterable[str]]=None, dp=DP_PROCESSED) -> murefi.ParameterMapping:
    df_mapping = pandas.read_excel(dp / "full_parameter_mapping.xlsx", index_col="rid")
    if rids:
        df_mapping = df_mapping.loc[list(rids)]
    model = MonodModel()
    theta_mapping = murefi.ParameterMapping(
        df_mapping,
        bounds={
            'S0': (15, 20),
            'X0': (0.01, 1),
            'mu_max': (0.4, 0.5),
            'Y_XS': (0.3, 1),
            'K_S': (1e-7, numpy.inf),
        },
        guesses={
            'S0': 17,
            'X0': 0.25,
            'mu_max': 0.42,
            'Y_XS': 0.6,
            'K_S': 0.02,
        }
    )
    return theta_mapping


class LinearCM(calibr8.BaseModelT):
    def __init__(self, blank=0):
        self.blank = blank
        theta_names = ('slope', 'scale', 'df')
        super().__init__(independent_key='X', dependent_key='Pahpshmir_1400_BS3_CgWT', theta_names=theta_names)

    def predict_dependent(self, x, *, theta=None):
        if theta is None:
            theta = self.theta_fitted
        slope, scale, df = theta
        mu = x * slope + self.blank
        return mu, scale, df

    def predict_independent(self, y, *, theta=None):
        if theta is None:
            theta = self.theta_fitted
        slope, scale, df = theta
        return (y - self.blank) / slope
