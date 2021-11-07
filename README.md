# Discrust

## _Supervised discretization in Rust_

The `discrust` package provides a supervised discretization algorithm. Under the hood it implements a decision tree, using information value to find the optimal splits, and provides several different methods to constrain the final discretization scheme.

_The package draws heavily from the [ivpy](https://github.com/gravesee/ivpy) package, both in the algorithm and the parameter controls._

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

This method will return a list of the optimal split values for the feature given the constraints. After being fit the discretizer will have a `splits_` attribute with this list.

```python
import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")

from discrust import Discretizer

ds = Discretizer(min_obs=5, max_bins=10, min_iv=0.001, min_pos=1.0, mono=None)
ds.fit(df["fare"], df["survived"])
# [-inf, 6.95, 7.125, 7.7292, 10.4625, 15.1, 50.4958, 52.0, 73.5, 79.65, inf]
```

The `predict` method can be called and will discretize the feature, and then perform weight of evidence substitution on each binned level. This method takes the following arguments.

- `x` **_(ArrayLike)_**: An arraylike numeric field.

```python
ds.predict(df["fare"])[0:5]
array([-0.84846814, 0.78344263, -0.787529, 0.78344263, -0.787529])
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

### Additional TODOs

- [ ] Support for exception values
- [ ] Support for missing values in both the dependant and independent variables
