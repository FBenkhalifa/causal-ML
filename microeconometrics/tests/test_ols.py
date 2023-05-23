from typing import Tuple
import statsmodels.api as sm
from microeconometrics.ols import LinearModel

import numpy as np
import pandas as pd

import pytest


@pytest.fixture
def test_data() -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Simulates some testing data for estimating OLS."""
    params = np.random.random(10)[:, np.newaxis]
    x = np.random.random((1000, 10))
    y = (x @ params).flatten() + np.random.normal(0, 5, size=1000) + 5
    x = pd.DataFrame(x)

    return params, x, y


def test_coefs(test_data):
    """
    Test if custom OLS can reproduce statsmodels OLS
    """
    params, x, y = test_data

    test_ols = LinearModel()
    test_ols.fit(X=sm.add_constant(x), y=y)

    model_bench = sm.OLS(endog=y, exog=sm.add_constant(x))
    model_bench_fit = model_bench.fit()

    assert np.allclose(model_bench_fit.params, test_ols.coefs)
