import datetime

import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
import numpy as np
from sklearn.impute import IterativeImputer
import logging
from microeconometrics import preprocessing as prep
from sklearn.linear_model import BayesianRidge


np.random.seed(13)


# 0. LOAD THE DATA ------------------------------------------------------
data_raw = pd.read_csv("data/data_group_1.csv", sep=";")
data = data_raw.copy()

# 1. FEATURES ------------------------------------------------------

# 1.0 Parse the target variable
data.dropna(subset=["run_time"])
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

# Drop blizzard observation - they are colinear with the country
data = data.query('skis != "blizzard"')

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

data.pop("total_wcp_nat_last_x")
data.rename(columns={"total_wcp_nat_last_y": "total_wcp_nat_last"}, inplace=True)

# 1.1 Construct target and treatment ------
data["treatment"] = data.coursesetter_nat == data.country
data["z_score"] = data.groupby("race_id", group_keys=False).apply(
    lambda x: (x["run_time"] - x["run_time"].mean()) / x["run_time"].std()
)

# 2 Feature engineering--------------------------------------------------

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

# Define feature sets for modeling

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


def prepare_data(df):
    data_imputed = pd.get_dummies(df, drop_first=False, dtype=int)
    data_imputed.columns = data_imputed.columns.str.replace(" ", "_")
    if "country" in df:
        data_imputed.drop(columns=["country_SUI"], inplace=True)

    if "skis" in df:
        data_imputed.drop(columns=["skis_fischer"], inplace=True)

    data_imputed_matrix = IterativeImputer(
        estimator=BayesianRidge(), random_state=0, verbose=2, max_iter=10
    ).fit_transform(X=data_imputed)

    data_imputed = pd.DataFrame(
        data_imputed_matrix, columns=data_imputed.columns, index=data_imputed.index
    )
    return data_imputed


# Impute and save prepared data

data_prepared = data.copy().filter(
    features_athletes_invariant + features_athletes_variant
)
data_prepared.to_csv("data/data_prepared.csv", index=True)

data_ols = prepare_data(
    data.copy().filter(features_athletes_invariant + features_athletes_variant)
)
data_ols.to_csv("data/data_ols.csv", index=True)

data_panel = prepare_data(
    data.copy().set_index(features_athlete_id).filter(features_athletes_variant)
)
data_panel.to_csv("data/data_panel.csv", index=True)

# Create variable summaries
data_summary = data_prepared.dtypes.to_frame(name="dtype")
data_summary["panel"] = data_prepared.columns.isin(features_athletes_invariant)
data_summary["ols"] = data_prepared.columns.isin(features_athletes_variant)
data_summary["Description"] = [
    "Country of athlete",
    "Ski brand of athlete",
    "The target variable. Z-score of athlete",
    "Start time of athlete in the race in minutes since midnight",
    "Age of athlete in years at the time of the race",
    "Total world cup points of the athlete's nation during the last season",
    "Binary variable indicating whether the athlete was treated",
    "Binary variable indicating whether the athlete was in the world cup start list",
    "Mean rank of athlete in the last 15 days",
    "Mean rank of athlete in the last 60 days",
    "Distance to the tournament in kilometers from the athlete's home country",
    "The fraction of past races the athlete has participated in this discipline",
    "The count of past races the athlete has participated in this discipline",
]
data_summary = data_summary[["Description", "dtype", "panel", "ols"]]
data_summary.columns = ["Description", "Type", "Panel", "Pooled"]
print(
    data_summary.to_latex(float_format="{:0.2f}".format, column_format="lp{5cm}lrr")
    .replace("_", "\_")
    .replace("object", "cat.")
    .replace("int64", "int.")
    .replace("float64", "cont.")
)
