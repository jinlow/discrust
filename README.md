# Discrust

## _Supervised discretization with Rust_

[![PyPI version shields.io](https://img.shields.io/pypi/v/discrust.svg)](https://pypi.python.org/pypi/discrust/)

The `discrust` package provides a supervised discretization algorithm. Under the hood it implements a decision tree, using information value to find the optimal splits, and provides several different methods to constrain the final discretization scheme.

_The package draws heavily from the [ivpy](https://github.com/gravesee/ivpy) package, both in the algorithm and the parameter controls. Why make another package? This package serves as a proof of
concept of building a python package using Rust and pyo3. Additionally the goal is for this
package to better align with the scikit-learn API, and possibly be used in other Rust based
credit score building tools._

## Usage

The package has a single user facing class, `Discretizer` that can be instantiated with the following arguments.

- `min_obs` **_(Optional[float], optional)_**: Minimum number of observations required
  in a bin. Defaults to 5.
- `max_bins` **_(Optional[int], optional)_**: Maximum number of bins to split the variable
  into. Defaults to 10.
- `min_iv` **_(Optional[float], optional)_**: Minimum information value required to make a split.
  Defaults to 0.001.
- `min_pos` **_(Optional[float], optional)_**: Minimum number of records with a value of one
  that should be present in a split. Defaults to 5.
- `mono` **_(Optional[int], optional)_**: The monotonicity required between the binned variable and
  the binary performance outcome. A value of -1 will result in negative correlation between
  the binned x and y variables, while a value of 1 will result in a positive correlation between the
  binned x variable and the y variable. Specifying a value of 0 will result in binning
  x, with no monotonicity constraint. If a value of None is specified the monotonicity
  will be determined the monotonicity of the first split. Defaults to None.

The `fit` method can be called on data and accepts the following parameters.

- `x` **_(ArrayLike)_**: An arraylike numeric field that will be discretized based on
  the values of `y`, and the constraints the `Discretizer` was initialized with.
- `y` **_(ArrayLike)_**: An arraylike binary field.
- `sample_weight` **_(Optional[ArrayLike], optional)_**: Optional sample weight array
  to be used when calculating the optimal breaks. Defaults to None.
- `exception_values` **_(Optional[List[float]], optional)_**: Optional list specifying exception
  values. These values are held out of the binning process, additionally, their
  respective weight of evidence, and summary information can be found in the
  `exception_values_` attribute once the discretizer has been fit.

A `np.nan` value may be present in the list of possible exception values. If there are `np.nan` values present in the `x` variable, and `np.nan` is not listed as a possible exception value, an error will be raised. Additionally, an error will be raised if `np.nan` is found to be in `y` or the `sample_weight` arrays.

This method will fit the decision tree and find the optimal split values for the feature given the constraints. After being fit the discretizer will have a `splits_` attribute with the optimal
split values.

```python
import seaborn as sns

df = sns.load_dataset("titanic")

from discrust import Discretizer

ds = Discretizer(min_obs=5, max_bins=10, min_iv=0.001, min_pos=1.0, mono=None)
ds.fit(df["fare"], df["survived"])
ds.splits_
# [-inf, 6.95, 7.125, 7.7292, 10.4625, 15.1, 50.4958, 52.0, 73.5, 79.65, inf]
```

Here we show what the results are if exception values are also specified. These exception values will be held out when calculating the bins.

```python
ds = Discretizer(min_obs=5, max_bins=10, min_iv=0.001, min_pos=1.0, mono=None)
ds.fit(df["age"], df["survived"], exception_values=[np.nan, 1.0])
ds.exception_values_
# {'vals_': [nan, 1.0],
#  'totals_ct_': [177.0, 7.0],
#  'iv_': [0.03054206173541801, 0.015253257689460616],
#  'ones_ct_': [52.0, 5.0],
#  'woe_': [-0.40378231427394834, 1.3895784363210804],
#  'zero_ct_': [125.0, 2.0]}
```

The `exception_values_` dictionary has the following keys.

- `vals_`: The exception values passed to the `Discretizer`.
- `totals_ct_`: The total number of each respective exception value present in the `x` variable used for fitting.
- `ones_ct_`: Total count of the positive class for each exception value.
- `zero_ct_`: Total count of zeros for each respective value.
- `woe_`: The weight of evidence for each respective exception value.
- `iv_`: The information value for each respective exception value.

The `predict` method can be called and will discretize the feature, and then perform either weight of evidence substitution on each binned level, or return the bin index. This method takes the following arguments.

- `x` **_(ArrayLike)_**: An arraylike numeric field.
- `prediction_type` **_(str, optional)_**: A string specifying which prediction
  type should be returned. The string specified must be one of
  "woe" or "index". Defaults to "woe".

  - If "woe" is supplied, weight evidence subtitution will be performed for each value, and the
    weight of evidence of the bin the value should fall in will be returned. For exception values found in `x`, the calculated weight of evidence for that exception value will be returned. If the exception value was never present in the `x` variable when the `Discretizer` was fit, then the returned weight of evidence will be zero for the exception value.
  - If "index" is specified, each value will be converted to the
    relevant bin index. These bins will be created from the `splits_`
    attribute and will be zero indexed. Any exception values will be encoded
    starting with -1 to -N, where N is the number of exception values present
    in the `exception_values_` attribute. The order of the exception values
    will be equivalent to the `vals_` key in this attribute.

```python
ds.predict(df["fare"])[0:5]
array([-0.84846814, 0.78344263, -0.787529, 0.78344263, -0.787529])
```

Specifying `prediction_type` to "index" will be equivalent to use the pandas `cut` method with the `splits_` on the `Discretizer` object used as the bins.

```python
import pandas as pd

ds = Discretizer(min_obs=5, max_bins=5, min_iv=0.001, min_pos=1.0, mono=None)
ds.fit(df["fare"], df["survived"])
pd.cut(df["fare"], bins=ds.splits_).value_counts().sort_index()
# (-inf, 6.95]        26
# (6.95, 7.125]       16
# (7.125, 10.462]    297
# (10.462, 73.5]     455
# (73.5, inf]         97
# Name: fare, dtype: int64

pd.value_counts(ds.predict(df["fare"], prediction_type="index")).sort_index()
# 0     26
# 1     16
# 2    297
# 3    455
# 4     97
# dtype: int64
```

One of the main benefits of using the `predict` method over the pandas cut function directly, is the built in support for exception values.

```python
ds = Discretizer(min_obs=5, max_bins=4, min_iv=0.001, min_pos=1.0, mono=None)
ds.fit(df["age"], df["survived"], exception_values=[np.nan, 1.0])

pd.value_counts(ds.predict(df["age"], prediction_type="index")).sort_index()
# -2      7
# -1    177
#  0      6
#  1     34
#  2    654
#  3     13
# dtype: int64

ds.exception_values_["vals_"]
# [nan, 1.0]
ds.exception_values_["totals_ct_"]
# [177.0, 7.0]
```

## Installation

### From PyPi

For Windows users, the package can be installed directly from pypi with the following command.

```shell
python -m pip install discrust
```

### Building from Source

The package can be built from source, it utalizes the [maturin](https://github.com/PyO3/maturin) tool as a build backend. This tool requires you have python, and a working Rust compiler installed, [see here for details](https://www.rust-lang.org/tools/install). If these two requirements are met, you can clone this repository, and run the following command in the repositories root directory.

```shell
python -m pip install . -v
```

This should invoke the `maturin` tool, which will handle the building of the Rust code and installation of the package. Alternativly, if you simply want to build a wheel, you can run the following command after installing `maturin`.

```shell
maturin build --release
```

_I have had some problems building packages with maturin directly in a conda environment, this is actually a bug on anaconda's side that will hopefully be resolved. If this does give you any problems, it's usually easiest to build a wheel inside of a `venv` and then install the wheel._
