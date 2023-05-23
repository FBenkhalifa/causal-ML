import pandas as pd
from doubleml import DoubleMLPLR
from catboost import CatBoostRegressor, CatBoostClassifier
from doubleml import DoubleMLData
from sklearn.experimental import enable_iterative_imputer  # noqa
from microeconometrics import preprocessing as prep
from statsmodels import api as sm
from microeconometrics.ols import LinearModel


# Load Data
data_ols = pd.read_csv("data/data_ols.csv", index_col=0)

# Convert one-hot to dummy
data_ols.drop(columns=["country_AUT", "skis_fischer"], inplace=True)

data_panel = pd.read_csv("data/data_panel.csv")

target = "z_score"

# 3.1 OLS model -----------------------------------------------------------------

ols = sm.OLS(
    exog=data_ols.drop(target, axis=1).assign(Intercept=1), endog=data_ols[target]
)

ols_fit = ols.fit()
ols_fit.summary()


model = LinearModel()
model.fit(X=data_ols.drop(target, axis=1).assign(Intercept=1), y=data_ols[target])
model.summary()



self = model

# 3,3 Panel data ------------------------------------------------------

# Conduct within transformation
data_panel = prep.FixedEffectsPreprocessor().fit_transform(X=data_panel.reset_index())

# Fixed effects model
ols_panel = sm.OLS(
    exog=sm.add_constant(data_panel.drop(target, axis=1)), endog=data_panel[target]
)
ols_panel_fit = ols_panel.fit()
ols_panel_fit.summary()


# 3.4 PLR --------------------------------------------------------------

data_container = DoubleMLData(
    data_panel.reset_index(drop=True),
    y_col=target,
    d_cols="treatment",
    force_all_x_finite="allow-nan",
)

double_ml = DoubleMLPLR(
    data_container,
    CatBoostRegressor(),
    CatBoostClassifier(),
)
double_ml.fit()
print(double_ml)
