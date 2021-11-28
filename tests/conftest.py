from pytest import fixture
import os
import pandas as pd


@fixture(scope="session")
def tests_path():
    return os.path.dirname(__file__)


@fixture(scope="module")
def titanic(tests_path):
    return pd.read_csv(os.path.join(tests_path, "data/titanic.csv"))
