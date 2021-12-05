from discrust import Discretizer
import numpy as np


def test_Discretizer_fit(titanic):
    ds = Discretizer(min_obs=5, max_bins=10, min_iv=0.001, min_pos=1.0, mono=None)
    ds.fit(titanic["fare"], titanic["survived"])
    assert ds.splits_ == [
        -np.inf,
        6.95,
        7.125,
        7.7292,
        10.4625,
        15.1,
        50.4958,
        52.0,
        73.5,
        79.65,
        np.inf,
    ]


def test_Discretizer_fit_mono_negative(titanic):
    ds = Discretizer(mono=-1)
    ds.fit(titanic["fare"].mul(-1), titanic["survived"], exception_values=[np.nan])
    avg_bad = (
        titanic["survived"].groupby(ds.predict(titanic["fare"].mul(-1), "index")).mean()
    ).to_list()

    assert all([i >= j for i, j in zip(avg_bad, avg_bad[1:])])


def test_Discretizer_fit_mono_positive(titanic):
    ds = Discretizer(mono=1)
    ds.fit(titanic["fare"], titanic["survived"], exception_values=[np.nan])
    avg_bad = (
        titanic["survived"].groupby(ds.predict(titanic["fare"], "index")).mean()
    ).to_list()

    assert all([i <= j for i, j in zip(avg_bad, avg_bad[1:])])
