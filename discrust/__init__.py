from __future__ import annotations
from .discrust import Discretizer as RustDiscretizer
import numpy as np
import pandas as pd
from typing import List, Optional, Union

ArrayLike = Union[pd.Series, np.ndarray]


class Discretizer(RustDiscretizer):
    def __new__(
        cls,
        min_obs: Optional[float] = 5,
        max_bins: Optional[int] = 10,
        min_iv: Optional[float] = 0.001,
        min_pos: Optional[float] = 5,
        mono: Optional[int] = None,
    ):
        return super().__new__(
            cls,
            min_obs=min_obs,
            max_bins=max_bins,
            min_iv=min_iv,
            min_pos=min_pos,
            mono=mono,
        )

    def __init__(
        self,
        min_obs: Optional[float] = 5,
        max_bins: Optional[int] = 10,
        min_iv: Optional[float] = 0.001,
        min_pos: Optional[float] = 5,
        mono: Optional[int] = None,
    ):
        """Create a binary discretizer

        Args:
            min_obs (Optional[float], optional): Minimum number of observations required
                in a bin. Defaults to 5.
            max_bins (Optional[int], optional): Maximum number of bins to split the variable
                into. Defaults to 10.
            min_iv (Optional[float], optional): Minimum information value required to make a split.
                Defaults to 0.001.
            min_pos (Optional[float], optional): Minimum number of records with a value of one
                that should be present in a split. Defaults to 5.
            mono (Optional[int], optional): The monotonicity required between the binned variable and
                the binary performance outcome. A value of -1 will result in negative corrlation between
                the binned x and y variables, while a value of 1 will result in a positive correlation between the
                binned x variable and the y variable. Specifying a value of 0 will result in binning
                x, with no monotonicity constraint. If a value of None is specified the monotonicity
                will be determined the monotonicity of the first split. Defaults to None.
        """
        super().__init__()

    @staticmethod
    def _convert_array(x: ArrayLike) -> np.ndarray:
        # Relevant conversions, need to be a numpy array.
        if isinstance(x, pd.Series):
            x = x.to_numpy().astype(np.float64)

        # Check dtypes, need to be a 64 bit float.
        # TODO: Make rust code accept different numerical types.
        if not x.dtype == np.float64:
            x = x.astype(np.float64)

        return x

    def fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        exception_values: Optional[List[float]] = None,
    ) -> Discretizer:
        """Fit the discretizer.

        Args:
            x (ArrayLike): An arraylike numeric field that will be discretized based on
                the values of `y`.
            y (ArrayLike): An arraylike binary field.
            sample_weight (Optional[ArrayLike], optional): Optional sample weight column to be applied
                to be used when calculating the optimal breaks. Defaults to None.
            exception_values (Optional[List[float]], optional): Optional list specifying exception
                values. These values are held out of the binning process, additionally, their
                their respective weight of evidence, and summary information can be found in the
                `exception_values_` attribute once the discretizer has been fit.

        Returns:
            List[float]: A list of the optimal split values for the feature.
        """
        x = self._convert_array(x)
        y = self._convert_array(y)
        if sample_weight is not None:
            sample_weight = self._convert_array(sample_weight)

        super().fit(x, y, sample_weight, exception_values)
        return self

    def predict(self, x: ArrayLike, prediction_type: str = "woe") -> np.ndarray:
        """Convert provided variable to WOE given the predicted discretization
        scheme.

        Args:
            x (ArrayLike): An arraylike numeric field.
            prediction_type (str, optional): A string specifying which prediction
                type should be returned. The string specified must be one of
                "woe" or "index". Defaults to "woe".

                * If "woe" is supplied, weight of evidence substitution will
                be returned for each value. Exception values will use the
                calculated weight of evidence value, if any were present in
                the variable when it was fit. If none were present, a value of
                0.0 will be returned for the weight of evidence.
                * If "index" is specified, each value will be converted to the
                relevant bin value, These bins will be created from the `splits_`
                attribute and will zero indexed. Any exception values will be encoded
                starting with -1 to -N, where N is the number of exception values present
                in the `exception_values_` attribute. The order of the exception values
                will be equivalent to the `vals_` key in this attribute.
        Returns:
            np.ndarray: The x variable where each level is transformed to
                it's respective weight of evidence given the fitted binning
                scheme.
        """
        x = self._convert_array(x)
        if prediction_type == "woe":
            return super().predict_woe(x)
        if prediction_type == "index":
            return super().predict_idx(x)
        else:
            e_msg = (
                "The parameter `prediction_type` must be one of 'index' or 'woe', "
                + f"but {prediction_type} was passed."
            )
            raise ValueError(e_msg)
