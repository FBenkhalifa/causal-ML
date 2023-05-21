from typing import Union

import pandas as pd
from doubleml import DoubleMLDID
from catboost import CatBoostRegressor, CatBoostClassifier
from doubleml import DoubleMLData
from sklearn.pipeline import make_pipeline
import statsmodels.api as sm
import statsmodels.formula.api as smf
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import logging
from sklearn.linear_model import BayesianRidge
from linearmodels import PanelOLS
from microeconometrics import preprocessing as prep
import seaborn as sns

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# 0. LOAD THE DATA ------------------------------------------------------
data = pd.read_csv("data/data_group_1.csv", sep=";")
data.dropna(subset=["run_time"], inplace=True)
data["location"] = data["location"].str.replace(
    "CORTINA D' AMPEZZO", "CORTINA D'AMPEZZO"
)
data["run_time"] = data.run_time.str.split(":|\.").apply(prep.to_miliseconds) / 1000
data["treatment"] = data.coursesetter_nat == data.country
data["date"] = pd.to_datetime(data["date"])

# Standardize the data by group of details_competition_type, gender, date and location
data["z_score"] = data.groupby(['details_competition_type', 'gender', 'date',  'location'], group_keys=False).apply(lambda x: (x["run_time"] - x["run_time"].mean()) / x["run_time"].std())

 data.to_csv("data/data_z_score.csv",  index=False)

box = sns.boxplot(y=data.z_score, x=data.treatment)
box.figure.show()
means = data.groupby(['details_competition_type', 'gender', 'date',  'location']).z_score.mean()
means.hist(bins=50)
plt.show()

# For each name, get the mean of the z-score when the treatment is 1 na dthe mean of the z-score when the treatment is 0
z_scores_mean = data.groupby(['name', 'treatment']).z_score.mean()
# Plot the z-score means for each name
z_scores_mean.hist(bins=50)
plt.show()
# Get the difference between the z-score means for each name (true - false)
z_scores_mean_diff = z_scores_mean.unstack().diff(axis=1).rename(columns={1: 'z_score_diff'})
z_scores_mean_diff.iloc[:, 1].plot.kde()

merged = data.merge(z_scores_mean.reset_index(), on='name')
merged = data.merge(z_scores_mean_diff.iloc[:, 1], on='name')
# For each name make a barplot of the z-score means (true and false) with seaborn
box = sns.boxplot(y=z_scores_mean.values, x=z_scores_mean.index.get_level_values(1))
box.figure.show()
h = merged.groupby(['gender', 'details_competition_type']).z_score_diff.mean()
merged.groupby([ 'coursesetter']).z_score_diff.mean().plot.kde()



cat_features = data.select_dtypes(include=["object", "category"]).columns
data = pd.get_dummies(data, columns=cat_features, drop_first=True, dtype=int)
panel_data = data.groupby(['name', 'date']).transform(lambda x: (x - x.mean())/ x.std())
