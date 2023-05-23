from typing import List, Union

import pandas as pd
import numpy as np
from sklearn.utils import check_X_y
from scipy.stats import t, f
from scipy.linalg import qr


class LinearModel:
    def __init__(self, robust: bool = True):
        self.robust: bool = robust

        self._coefs: List[float] = None
        self._varnames: List[str] = None
        self._std_errors: List[float] = None
        self._t_stats: List[float] = None
        self._p_values: List[float] = None
        self._r_squared: float = None
        self._r_squared_adj: float = None
        self._f_stat: float = None
        self._f_stat_p_value: float = None
        self._is_fitted: bool = False
        self._degrees_of_freedom: int = None

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray, List]) -> None:
        # Validate input
        X = X.copy()

        # Store variable names
        self._varnames = X.columns.tolist()

        # Check input and transform to arrays so we can apply numpy operators
        X, y = check_X_y(X, y, y_numeric=True)

        # Store number of obs and vars
        n, p = X.shape

        # Get the coefficient estimates
        # coefs = np.linalg.inv(X.T @ X) @ X.T @ y

        # DO QR decomposition for numerical stability
        q_mat, r_mat = qr(X, mode="economic")

        # Solving the system of equations
        coefs = np.linalg.solve(r_mat, q_mat.T @ y)

        # Calculating residuals which we need for standard errors
        e = y - X @ coefs

        # Calculating res sum of squares
        rss = e.T @ e
        tss = np.sum((y - np.mean(y)) ** 2)

        # Calculating degrees of freedom
        df_reg = p
        df_resid = n - p - 1

        if self.robust:
            # Calculating robust standard errors using White covariance estimator
            cov_matrix = (
                np.linalg.inv(X.T @ X)
                @ X.T
                @ np.diag(e**2)
                @ X
                @ np.linalg.inv(X.T @ X)
            )

        else:
            # Get the estimated variance of the error term
            var_residuals = rss / df_resid
            cov_matrix = var_residuals * np.linalg.inv(X.T @ X)

        # Extracting standard errors
        standard_errors = np.sqrt(np.diag(cov_matrix))

        # Calculating t-stats and p-values
        t_stats = coefs / standard_errors
        p_values = 2 * (1 - t.cdf(np.abs(t_stats), df_resid))

        # Calculating R-squared
        r_squared = 1 - rss / tss
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

        # Calculating F-statistic and its p-value
        f_stat = (tss - rss) / df_reg / (rss / df_resid)
        f_stat_p_value = 1 - f.cdf(f_stat, df_reg, df_resid)

        self._coefs = coefs
        self._std_errors = standard_errors
        self._t_stats = t_stats
        self._p_values = p_values
        self._r_squared = r_squared
        self._r_squared_adj = adj_r_squared
        self._f_stat = f_stat
        self._f_stat_p_value = f_stat_p_value
        self._is_fitted = True
        self._degrees_of_freedom = df_resid

    @property
    def coefs(self) -> pd.Series:
        if not self._is_fitted:
            raise ValueError("Model not fitted yet.")

        return pd.Series(self._coefs, index=self._varnames)

    @property
    def pvalues(self) -> pd.Series:
        if not self._is_fitted:
            raise ValueError("Model not fitted yet.")

        return pd.Series(self._p_values, index=self._varnames)

    @property
    def coefs_summary(self) -> pd.DataFrame:
        if not self._is_fitted:
            raise ValueError("Model not fitted yet.")

        return pd.DataFrame(
            {
                "Coefficients": self.coefs,
                "Standard Errors": self._std_errors,
                "t-Values": self._t_stats,
                "p-Values": self._p_values,
            },
            index=self._varnames,
        )

    def summary(self) -> None:
        """Prints a model summary"""

        param_table = self.coefs_summary

        # Print formatted summary inserting the calculated values
        print(
            "==============================================================================="
        )
        print(f"Dependent Variable: {self._varnames[-1]}")
        print("Method: Least Squares")
        print(f"Date: {pd.Timestamp.now()}")
        print(f"Time: {pd.Timestamp.now().time()}")
        print(f"Number of observations: {len(self._varnames) - 1}")
        print(
            f"R-squared: {self._r_squared:.4f}, Adjusted R-squared: {self._r_squared_adj:.4f}"
        )
        print(f"F-statistic: {self._f_stat:.4f}, p-value: {self._f_stat_p_value:.4f}")
        print(
            "==============================================================================="
        )
        print(param_table.round(3).to_string())
        print(
            "==============================================================================="
        )
