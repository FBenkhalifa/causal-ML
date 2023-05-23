import datetime

import pandas as pd
from doubleml import DoubleMLPLR
from catboost import CatBoostRegressor, CatBoostClassifier
from doubleml import DoubleMLData
from sklearn.experimental import enable_iterative_imputer  # noqa
import numpy as np
from sklearn.impute import IterativeImputer
import logging
from sklearn.linear_model import BayesianRidge
from microeconometrics import preprocessing as prep
from statsmodels import api as sm

np.random.seed(13)

# Configs ------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

# 0. LOAD THE DATA ------------------------------------------------------
data_raw = pd.read_csv("data/data_group_1.csv", sep=";")
data = data_raw.copy()
data.dropna(subset=["run_time"])

# 1. FEATURES ------------------------------------------------------

# 1.0 Parse the target variable
data["run_time"] = pd.to_datetime(
    data["run_time"], format="%M:%S.%f", errors="coerce"
) - datetime.datetime(1900, 1, 1)
data["run_time"] = data["run_time"].dt.total_seconds() * 1_000
data.dropna(subset=["run_time"], inplace=True)


# 1.1 Transformations
data["date"] = pd.to_datetime(data["date"])
data["start_time"] = data["start_time"].apply(prep.time_string_to_minutes)

# 1.2 Inconsistencies
# Harmonize the names of the coursesetters
data["coursesetter"] = data["coursesetter"].replace(
    {
        "Anderson Pete": "Anderson Peter",
        "Brun Loïc": "Brun Loic",
        "Del Dio Simone": "Deldio Simone",
        "Evers Andy": "Evers Andreas",
        "FÜrbeck Andi": "FÜrbeck Andreas",
        "Girardi Valter": "Girardi Walter",
        "Glasse-davies Tristan": "Glasse Davies Tristan",
        "Krug Helmut": "Krug Helmuth",
        "Ominger Andreas": "Omminger Andreas",
        "PoljŠak Sergej": "Poljsak Sergej",
        "Thoule Nicola": "Thoule Nicolas",
        "Vuignier Julian": "Vuignier Julien",
    }
)
# Harmonize the names of the skis
data["skis"] = data["skis"].str.lower()

# Harmonize the locations
data["location"] = data["location"].str.replace(
    "CORTINA D' AMPEZZO", "CORTINA D'AMPEZZO"
)

# Make age granular
data["age"] = (
    data.date - pd.to_datetime(data["birthdate"])
).dt.days / 365.25  # Compute age of athlete since age stays constant in the orignal dataset

# Delete values that are not possible
data = data.query("vertical_drop > 0")

# 1.2 Missing values
# Recover the genders for the names with NA,
female_coursesetter = data.query('gender == "Female"').coursesetter.unique()
male_coursesetter = data.query('gender == "Male"').coursesetter

# Get the coursesetters that have set courses in both male and female tournaments
both_coursesetter = male_coursesetter[
    male_coursesetter.isin(female_coursesetter)
].unique()

# Remove the entries from both_coursesetter from male_coursesetter and female_coursesetter
male_coursesetter = np.setdiff1d(male_coursesetter, both_coursesetter)
female_coursesetter = np.setdiff1d(female_coursesetter, both_coursesetter)

# Create a mask for rows where 'gender' is NA and 'coursesetter' is in 'male_coursesetter'
mask_male = data["gender"].isna() & data["coursesetter"].isin(male_coursesetter)
mask_female = data["gender"].isna() & data["coursesetter"].isin(female_coursesetter)

# Assign 'Male' to 'gender' where the mask is True
data.loc[mask_male, "gender"] = "Male"
data.loc[mask_female, "gender"] = "Female"

# Quick eyeball check if names seem male and female
# data[data["gender"] == "Male"]["name"].unique()
# data[data["gender"] == "Female"]["name"].unique()

# Make unique race id
data["race_id"] = data[
    [
        "details_competition_type",
        "run",
        "gender",
        "coursesetter",
        "date",
        "number_of_gates",
    ]
].apply(lambda row: "_".join(row.values.astype(str)), axis=1)

# 1.3. Duplicated values ----------------------------------------------------------
duplicated = data.duplicated(subset=["name", "date", "start_time"], keep=False)
duplicated_df = (
    data[duplicated]
    .sort_values(by=["name", "start_time"])[
        [
            "name",
            "date",
            "start_time",
            "season",
            "run",
            "run_time",
            "total_wcp_nat_last",
        ]
    ]
    .head(6)
)
# print(duplicated_df.astype(str).to_latex(escape=True, index=False))

# Fix season
data["season"] = data["date"].apply(prep.assign_season)
data = data.drop_duplicates(subset=["name", "date", "start_time"], keep="first")

# Recalculate the total_wcp_nat_last
wpc_last_mask = data.season.isin(["2020_2021"])
wpc_last_nat = (
    data.groupby(["season", "country", "gender"])
    .wpc.sum()
    .reset_index()
    .replace(
        {
            "2022_2023": "2023_2024",
            "2021_2022": "2022_2023",
            "2020_2021": "2021_2022",
        }
    )
    .rename(columns={"wpc": "total_wcp_nat_last"})
)

# Replace the total_wcp_nat_last_y with the total_wcp_nat_last_x where the total_wcp_nat_last_y is NA
total_wcp_nat_last = (
    data[wpc_last_mask]
    .groupby(["season", "country", "gender"])
    .total_wcp_nat_last.max()
    .reset_index()
)
total_wcp_nat_last = pd.concat([wpc_last_nat, total_wcp_nat_last], axis=0)
data = data.merge(total_wcp_nat_last, how="left", on=["season", "country", "gender"])

# Eyeball the values
data.query('country == "AUT"')[
    ["season", "total_wcp_nat_last_x", "total_wcp_nat_last_y", "country", "gender"]
].drop_duplicates().sort_values(["season", "gender"])
data.pop("total_wcp_nat_last_x")
data.rename(columns={"total_wcp_nat_last_y": "total_wcp_nat_last"}, inplace=True)

# 1.1 Construct target and treatment ------
data["treatment"] = data.coursesetter_nat == data.country
data["z_score"] = data.groupby("race_id", group_keys=False).apply(
    lambda x: (x["run_time"] - x["run_time"].mean()) / x["run_time"].std()
)

# 2 Feature engineering--------------------------------------------------
# 2.1 Athlete specific features --------
# Current form of athlete measured in world cup points
data = prep.convert_bib_to_start_list(data)
data = prep.assign_wcp_accumulated(
    df=data, target="wpc", col_prefix="wcp"
)  # wpc overall
data = prep.assign_wcp_accumulated(
    df=data,
    target="wpc",
    col_prefix="discipline_wcp",
    additional_group_vars=["details_competition_type", "season"],
)  # wpc per discipline


# Form features
for lag in [15, 30, 60]:
    data = (
        data.sort_values(["name", "date"])
        .groupby("name", group_keys=False)
        .apply(lambda df: prep.rolling_mean_last_n_days(df, "rank", lag))
    )

data.groupby(["name"])["details_competition_type"].value_counts()

# Add variables measuring experience in this discipline
data["discipline_count"] = (
    data.groupby(["name", "details_competition_type"], group_keys=False)[
        "details_competition_type"
    ].transform("cumcount")
    + 1
)

# Divide by the total number of races the athlete has participated in up until that point
data["discipline_frac"] = data["discipline_count"] / (
    data.groupby(["name"], group_keys=False)["details_competition_type"].transform(
        "cumcount"
    )
    + 1
)


#  2.1 Coach specific features (influence on the treatment) ---------------

# Get the distance to the tournament according to the home country of the athlete
data = prep.retrieve_distance_to_tournament(data)
data["distance_to_tournament"] = data["distance_to_tournament"] / 100

# Repeat for the discipline wpc points
wpc_last_nat_discipline = (
    data.groupby(["season", "country", "gender", "details_competition_type"])
    .wpc.sum()
    .reset_index()
    .replace(
        {
            "2022_2023": "2023_2024",
            "2021_2022": "2022_2023",
            "2020_2021": "2021_2022",
        }
    )
    .rename(columns={"wpc": "total_wcp_discipline_nat_last"})
)
data = data.merge(wpc_last_nat_discipline, how="left")

# 1.3 Tournament specific features -----------------------------------------------------
data.to_csv("data/data_preprocessed.csv", index=False)

# 2. Missing values ------------------------------------------------------------
# Filter dataset for columns that are not in to_remove
missing_summary = prep.missing_values_summary(
    df=data.loc[
        :,
        ~data.columns.isin(
            [
                "wpc_fake",
                "total_rank",
                "total_time",
                "wpc",
                "bib",
                "birthdate",
                "total_wcp_discipline_nat_last",
            ]
        ),
    ]
)
# print(
#     missing_summary[missing_summary.iloc[:, 1] > 0].to_latex(
#         escape=True, float_format="%.2f"
#     )
# )

# Detect multicollinearity
multi_col = prep.detect_multicollinearity(df=data, threshold=0.8, target="run_time")
print(multi_col)

features_athletes_variant = [
    "z_score",
    "start_time",  # Should be left, to incorporate time effects during one dqy
    "age",  # Stays in favor of birthdate
    "total_wcp_nat_last",  # High correlation with treatment
    "treatment",
    "approx_world_cup_start_list",
    "rank_mean_last_15_days",
    "rank_mean_last_60_days",
    "distance_to_tournament",  # Accounts for proximity advantages
    "discipline_frac",
    "discipline_count",
]

features_athletes_invariant = [
    "country",  # Stays to account for general strength of country
    "skis",  # Replaced by sponsor
]

features_run = [
    "gender",
    "number_of_gates",
    "details_competition_type",
    "vertical_drop",
    "season",
    "gate_per_vertical_meter",
    "day_since_season_start",
]

features_athlete_id = ["name", "date"]



from sklearn.impute import SimpleImputer




# data_imputed = pd.read_csv("data/data_imputed.csv")

# 3. MODELLING ------------------------------------------------------------------
target = "z_score"

# 3.1 OLS model -----------------------------------------------------------------
data_ols = prepare_data(
    data.copy().filter(features_athletes_invariant + features_athletes_variant)
)

ols = sm.OLS(
    exog=sm.add_constant(data_ols.drop(target, axis=1)), endog=data_ols[target]
)
ols_fit = ols.fit()
ols_fit.summary()

# 3,3 Panel data ------------------------------------------------------
data_panel = prepare_data(
    data.copy().set_index(features_athlete_id).filter(features_athletes_variant)
)

# Conduct within transformation
data_panel = prep.FixedEffectsPreprocessor().fit_transform(X=data_panel.reset_index())

# Fixed effects model
ols_panel = sm.OLS(
    exog=sm.add_constant(data_panel.drop(target, axis=1)), endog=data_panel[target]
)
ols_panel_fit = ols_panel.fit()
ols_panel_fit.summary()


# 3.4 PLR --------------------------------------------------------------
data_plr_imputed = prepare_data(
    data.copy()
    .set_index(features_athlete_id)
    .filter(features_athletes_invariant + features_athletes_variant)
)

data_plr_panel = prep.FixedEffectsPreprocessor().fit_transform(
    X=data_plr_imputed[features_athletes_variant].reset_index()
)

# Insert the transformed features
data_plr = data_plr_imputed.copy()
data_plr = data_plr.assign(
    **{col: data_plr_panel[col] for col in data_plr_panel.columns if col != "treatment"}
)

# Get original treatment variable ()
data_container = DoubleMLData(
    data_plr.reset_index(drop=True),
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


# Out-of-sample Performance:
# Learner ml_l RMSE: [[0.69205823]]
# Learner ml_m RMSE: [[0.26788376]]


# 5. Interpretation ------------------------------------------------------
# Simulate treatment effect on the whole dataset
# rank_ols = prep.simulate_treatment_effect(data=data,
#                                           treatment_effect=prep.extract_treatment_effect(ols).loc["beta"])
# rank_ols.rank_change.mean()
# rank_ols.rank_change.hist(bins=30)
# plt.show()
#
# # Rank changes for panel
# rank_panel = prep.simulate_treatment_effect(
#     data, prep.extract_treatment_effect(panel_ols_summary).loc["beta"]
# )
# rank_panel.rank_change.mean()
# rank_panel.rank_change.hist(bins=30)
# plt.show()
#
# # Rank changes for double ml
# rank_double_ml = prep.simulate_treatment_effect(data, double_ml.coef[0])
# rank_double_ml.rank_change.mean()
# rank_double_ml.rank_change.hist(bins=30)
# plt.show()
#
#
# plot_data = pd.concat(
#     [
#         rank_ols.assign(model='ols'),
#         rank_panel.assign(model='panel-ols'),
#         rank_double_ml.assign(model='double-ml'),
#     ],
#     axis=0,
# )
# plot_data['model'] = plot_data.model.values + plot_data.treatment_effect.round(3).str.to_list()
# # Plot the rank change for each model in one plot
# plot = sns.boxplot(data=plot_data,
#             x='model',
#             y='rank_change',
#             palette='Set3',
#             linewidth=1.2,
#             fliersize=2,
# flierprops=dict(marker='o', markersize=4)
#             )
# plt.show()
# plot.figure.savefig('plots/rank_change.pdf')
# # Descriptive analysis -------------------------------------------------------
# summary = data.groupby(['details_competition_type', 'gender']).describe().T
# summary = data.describe().T
#
