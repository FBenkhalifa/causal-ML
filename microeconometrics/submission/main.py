import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from doubleml import DoubleMLData, DoubleMLPLR
from statsmodels import api as sm

from functions import (FixedEffectsPreprocessor, LinearModel,
                       make_combined_hist, prepare_data,
                       print_variable_summaries_latex)

# Set numpy seed for reproducibility (for DML)
np.random.seed(1995)

# 1. Load and prepare data ------------------------------------------------------
data_prepared, data_ols, data_panel = prepare_data()
data_panel.to_csv("data/data_panel.csv", index=True)
data_ols.to_csv("data/data_ols.csv", index=True)
data_prepared.to_csv("data/data_prepared.csv", index=True)

# data_panel = pd.read_csv("data/data_panel.csv").set_index(["name", "date"])
# data_ols = pd.read_csv("data/data_ols.csv", index_col=0)
# data_prepared = pd.read_csv("data/data_prepared.csv", index_col=0)

# Print summary for paper
print_variable_summaries_latex(data_prepared)

# 2. Describe data ------------------------------------------------------
data = pd.read_csv("data/data_group_1.csv", sep=";")

# Create combined histograms
make_combined_hist(data, save_path="data_combined_hist.pdf")

# See also the file plots.R for more plots

# 3. Fit models ------------------------------------------------------
target = "z_score"

# 3.1 Pooled OLS model -----------------------------------------------------------------

exog_ols = data_ols.drop(target, axis=1).assign(Intercept=1)
endog_ols = data_ols[target]

# Fit custom OLS
model_ols = LinearModel(robust=True)
model_ols.fit(X=exog_ols, y=endog_ols)
model_ols.summary()

# Check using statsmodels
check_ols = sm.OLS(exog=exog_ols, endog=endog_ols)
check_ols_fit = check_ols.fit(cov_type="HC0")
check_ols_fit.summary()

# Eyeball differences
params_frame = pd.concat([check_ols_fit.params, model_ols.coefs], axis=1)
pvals_frame = pd.concat([check_ols_fit.pvalues, model_ols.pvalues], axis=1)

# 3.3 Panel OLS model ------------------------------------------------------

# Conduct within transformation
data_panel = FixedEffectsPreprocessor().fit_transform(X=data_panel.reset_index())
exog_panels = data_panel.drop(target, axis=1).assign(Intercept=1)
endog_panels = data_panel[target]

# Fit custom OLS
model_panel = LinearModel(robust=True)
model_panel.fit(X=exog_panels, y=endog_panels)
model_panel.summary()

# Check using statsmodels
check_panel = sm.OLS(exog=exog_panels, endog=endog_panels)
check_panel_fit = check_panel.fit(cov_type="HC0")
check_panel_fit.summary()

# Eyeball differences
params_frame = pd.concat([check_panel_fit.params, model_panel.coefs], axis=1)
pvals_frame = pd.concat([check_panel_fit.pvalues, model_panel.pvalues], axis=1)


# 3.4 PLR --------------------------------------------------------------

data_container = DoubleMLData(
    data_panel.reset_index(drop=True),
    y_col=target,
    d_cols="treatment",
    force_all_x_finite="allow-nan",
)
# Fit PLR - we increase the number of repetitions and folds since the results seem to be quite unstable
double_ml = DoubleMLPLR(
    data_container,
    CatBoostRegressor(verbose=False),
    CatBoostRegressor(verbose=False),
    n_rep=5,
    n_folds=10,
)
double_ml.fit()
print(double_ml)
