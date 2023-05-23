import datetime
import logging
import os
from datetime import datetime, timedelta
from typing import Any, List, Tuple, Union

import geopy.distance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycountry
from geopy.geocoders import Nominatim
from pandas import DataFrame
from scipy.linalg import qr
from scipy.stats import f, pointbiserialr, t
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LassoCV
from sklearn.utils import check_X_y
from tqdm import tqdm

np.random.seed(13)


# Data preprocessing ---------------------------------------------------------------------------------------------------


# 0. FUNCTIONS ---------------------------------------------------------------------------------------
def to_miliseconds(x: List[str]) -> float:
    """Converts the run_time into miliseconds."""
    # If x is an NA value, return NA
    if x is np.nan:
        return np.nan
    if len(x) == 2:
        x = ["0"] + x
    return float(x[0]) * 60 * 1000 + float(x[1]) * 1000 + float(x[2])


def time_string_to_minutes(time_str: str) -> float:
    """Converts the format HH:MM to minutes since midnight."""
    hours, minutes = map(int, time_str.split(":"))
    total_minutes = hours * 60 + minutes
    return total_minutes


def convert_bib_to_start_list(data: pd.DataFrame) -> pd.DataFrame:
    """Converts the bib number into a world cup rank according to the data description."""
    newCol = "approx_world_cup_start_list"
    data[newCol] = np.nan

    for index, row in data.iterrows():
        bib = row["bib"]
        discipline = row["details_competition_type"]
        run = row["run"]
        season = row["season"]

        if discipline in ["Slalom", "Giant Slalom"]:
            # For the first run, athletes are grouped into buckets based on the world cup start list
            if run == 1:
                # The top 7 athletes can start with bib numbers 1-7
                if 1 <= bib <= 7:
                    start_list = (1 + 7) / 2  # Assign the mean rank
                # athletes 8-15, who can start with numbers 8-15
                elif 8 <= bib <= 15:
                    start_list = (8 + 15) / 2  # Assign the mean rank
                # Remaining athletes are then assigned according to the world cup start list
                else:
                    start_list = bib
            # In the second run, only the top 30 athletes participate, with the 30th starting first and the first starting last
            elif run == 2:
                start_list = 31 - bib

        elif discipline == "Super G":
            # System implemented this season
            if season == "2022_2023":
                # The top ten athletes start with bib numbers 6-15
                if 6 <= bib <= 15:
                    start_list = (1 + 10) / 2  # Assign the mean rank
                # Athletes 11-20 can start with bib numbers 1-5 and 16-20
                elif 1 <= bib <= 5 or 16 <= bib <= 20:
                    start_list = (11 + 20) / 2  # Assign the mean rank
                # Remaining athletes start with their world cup starting list number.
                else:
                    start_list = bib
            # System implemented the  previous two seasons
            else:
                # Top 20 can choose their bib number
                if bib <= 20:
                    # Top 10 can choose an odd number between 1 and 20
                    if bib % 2 == 1:
                        start_list = (1 + 10) / 2
                    # Top 10 can choose an even number between 1 and 20
                    else:
                        start_list = (11 * 20) / 2
                # Remaining athletes start with their world cup starting list number.
                else:
                    start_list = bib

        data.at[index, newCol] = start_list

    return data


def assign_season(date: datetime) -> str:
    """Assigns the season to a date."""
    if date.month >= 10:
        season = f"{date.year}_{date.year + 1}"
    elif date.month <= 3:
        season = f"{date.year - 1}_{date.year}"
    else:
        season = None  # or assign a value for dates outside the ski season
    return season


def missing_values_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Creates a summary of missing values in a dataframe."""
    df = df.copy()
    run_time = df.pop("run_time")
    z_score = df.pop("z_score")
    missing = df.isnull().sum()
    missing_percent = df.isnull().mean() * 100

    # Create a dataframe to hold missing value indicators
    df_missing = df.isnull().astype(int)

    # Calculate correlation of missing indicators with outcome
    run_time_corr = df_missing.assign(outcome=run_time).corr()["outcome"]
    z_score_corr = df_missing.assign(outcome=z_score).corr()["outcome"]

    missing_df = pd.DataFrame(
        {
            "Missing Values": missing,
            "Missing (%)": missing_percent,
            "Corr(Missing = 1, run_time)": run_time_corr,
            "Corr(Missing = 1, z_score)": z_score_corr,
        }
    )

    # Sort by the number of missing values
    missing_df = missing_df.sort_values(by="Missing Values", ascending=False)
    return missing_df


# relative strenghts
# seasonal_relative_strengths
# lifetime_relative_strengths
def assign_wcp_accumulated(
    df: pd.DataFrame,
    target: str,
    col_prefix: str,
    additional_group_vars: str = ["season"],
) -> pd.DataFrame:
    """Computes the accumulated world cup points for an athlete per season (and an additional group)."""
    group_vars = ["name"]

    # Add the additional group if specified
    if additional_group_vars is not None:
        group_vars.extend(additional_group_vars)

    col_accumulated = f"{col_prefix}_accumulated"
    col_relative_rank = f"{col_prefix}_relative_rank_on_race"

    for _, grouped_df in df.groupby(group_vars):
        grouped_df = grouped_df.sort_values(["date", "run"])
        df.loc[grouped_df.index, col_accumulated] = (
            grouped_df[target].fillna(0).cumsum().shift(1).fillna(0)
        )

    df[col_relative_rank] = df.groupby("race_id")[col_accumulated].rank(ascending=False)
    return df


def rolling_mean_last_n_days(
    group: pd.DataFrame, mean_var: str, days: int
) -> pd.DataFrame:
    rolling_mean = []
    for i, row in group.iterrows():
        one_month_ago = row["date"] - timedelta(days=days)
        last_month_rows = group[
            (group["date"] >= one_month_ago)
            & (group["date"] <= (row["date"] - timedelta(days=1)))
        ]
        rolling_mean.append(last_month_rows[mean_var].mean())
    group[f"{mean_var}_mean_last_{days}_days"] = rolling_mean
    group[f"{mean_var}_mean_last_{days}_days"].ffill(inplace=True)
    return group


def assign_wcp_rank(df: pd.DataFrame, new_col: str, target_wpc: str) -> pd.DataFrame:
    # For each date in a season, take the acc_wpc of each athlete, rank them and assign the rank to the athlete
    for group, df_group in df.groupby(["season", "date", "gender", "run"]):
        ranks = df_group[target_wpc].rank(ascending=False).astype(int)
        df.loc[df_group.index, new_col] = ranks
    return df


def retrieve_distance_to_tournament(
    data: pd.DataFrame,
    from_disk=True,
) -> pd.DataFrame:
    """Retrieves the distance (in km) between the home country of an athlete to the tournament hosting country using the NOMINATIM API."""
    data = data.copy()
    file_path = "data/distance_to_tournament.parquet"
    # Check if there is a parquet file in the data folder
    if from_disk and os.path.isfile(file_path):
        logging.warning(f"Retrieving distance_to_tournament from disk.")
        # If there is, load it
        df_distance_map = pd.read_parquet(
            "data/distance_to_tournament.parquet", engine="fastparquet"
        )

        # Join by country, location and add distance_to_tournament as new variable
        return data.merge(
            df_distance_map,
            on=["country", "location_country"],
            how="left",
        )

    replacements = {
        "SUI": "CHE",  # Switzerland
        "GER": "DEU",  # Germany
        "SLO": "SVN",  # Slovenia
        "BUL": "BGR",  # Bulgaria
        "DEN": "DNK",  # Denmark
        "MAD": "MDG",  # Madagascar (assuming 'MAD' refers to Madagascar)
        "NED": "NLD",  # Netherlands
        "GRE": "GRC",  # Greece
        "CRO": "HRV",  # Croatia
        "CHI": "CHL",  # Chile
        "LAT": "LVA",  # Latvia
    }
    # invalid_countries = []
    # for code in data.country.unique():
    #     country = pycountry.countries.get(alpha_3=code)
    #     if not country:
    #         invalid_countries.append(code)

    # Instantiate geolocator which resolves the coordinates for each country
    geolocator = Nominatim(user_agent="myGeocoder")

    def _get_country_coordinates(country_code: str):
        """Get the coordinates of a country"""
        # Get country name which is needed for geolocator
        country = pycountry.countries.get(alpha_3=country_code)
        # Since the country_codes in the dataset are not consistently alpha_3, some countries are not found
        if not country:
            # In this case, use the replacements dictionary which maps the country codes to alpha_3
            country = pycountry.countries.get(
                alpha_3=replacements.get(country_code, None)
            )
        location = geolocator.geocode(country.name)
        return location.latitude, location.longitude

    def _get_distance(country_code_1: str, country_code_2: str):
        """Measure the distance between two countries in km"""
        coords_1 = _get_country_coordinates(country_code_1)
        coords_2 = _get_country_coordinates(country_code_2)
        distance = geopy.distance.distance(coords_1, coords_2).km
        return distance

    # Retrieve the distance for each country combination from the API -----
    # Initialize an empty dictionary to store calculated distances
    calculated_distances = {}
    for group, df in tqdm(data.groupby(["location_country", "country"])):
        print(group)
        # Check if the combination or its reverse is in the dictionary
        if (
            group[0] in calculated_distances
            and group[1] in calculated_distances[group[0]]
        ):
            distance = calculated_distances[group[0]][group[1]]
        elif (
            group[1] in calculated_distances
            and group[0] in calculated_distances[group[1]]
        ):
            distance = calculated_distances[group[1]][group[0]]
        else:
            distance = _get_distance(group[0], group[1])
            if group[0] not in calculated_distances:
                calculated_distances[group[0]] = {}
            calculated_distances[group[0]][group[1]] = distance
        data.loc[df.index, "distance_to_tournament"] = distance

    # Save to disk -----
    # Create new data folder if not already existent
    if not os.path.exists("data"):
        logging.warning(f"Creating new folder 'data'")
        os.makedirs("data")
    logging.warning(f"Saving distance data to 'data/distance_to_tournament.parquet'")
    data[
        ["country", "location_country", "distance_to_tournament"]
    ].drop_duplicates().to_parquet("data/distance_to_tournament.parquet")

    return data


def detect_multicollinearity(df: pd.DataFrame, threshold=0.8, target=None):
    """Returns a list of feature pairs with a correlation greater than the threshold. Also detects numeric-categorical pairs with Point-Biserial correlation."""
    # Compute the correlation matrix for numeric columns
    corr_matrix = df.select_dtypes(include=[np.number]).corr().abs()

    # Find feature pairs with correlation greater than the threshold
    multicoll_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] >= threshold:
                multicoll_pairs.append(
                    dict(
                        var_1=corr_matrix.columns[i],
                        var_2=corr_matrix.columns[j],
                        corr=corr_matrix.iloc[i, j],
                    )
                )

    # Compute Point-Biserial correlation for numeric-categorical pairs
    num_features = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if target:
        num_features.remove(target)

    for num_feature in num_features:
        for cat_feature in cat_features:
            # Convert categorical feature to numeric
            available_data = df.dropna(subset=[num_feature, cat_feature])
            pb_corr, _ = pointbiserialr(
                available_data[cat_feature].astype("category").cat.codes,
                available_data[num_feature],
            )
            if abs(pb_corr) >= threshold:
                multicoll_pairs.append(
                    dict(
                        var_1=num_feature,
                        var_2=cat_feature,
                        corr=pb_corr,
                    )
                )

    return pd.DataFrame.from_records(multicoll_pairs).assign(threshold=threshold)


class DoubleSelection(BaseEstimator, TransformerMixin):
    def __init__(
        self, target: str, treatment: str, alpha=0.0, controls: List[str] = None
    ):
        self.alpha = alpha
        self.controls = controls if controls else []
        self.target = target
        self.treatment = treatment
        self._lasso_y = None
        self._lasso_z = None

        self._columns = None

    def fit(self, X: pd.DataFrame, y=None) -> None:
        X = X.copy()
        y = X.pop(self.target)
        d = X.pop(self.treatment)

        # Step 1: Lasso regression of y on X
        self._lasso_y = LassoCV(cv=5, random_state=13).fit(X, y)

        # Step 2: Lasso regression of Z on X
        self._lasso_z = LassoCV(cv=5, random_state=13).fit(X, d)

        # Collect the variables
        self._columns = X.columns

    @property
    def selected_variables(self) -> pd.Index:
        selected_y = np.abs(self._lasso_y.coef_) > self.alpha
        selected_z = np.abs(self._lasso_z.coef_) > self.alpha
        selected = np.logical_or(selected_y, selected_z)
        return self._columns[selected].union(self.controls + [self.treatment])

    def transform(self, X: pd.DataFrame) -> tuple[DataFrame, Any]:
        X = X.copy()
        y = X.pop(self.target)
        return X.filter(self.selected_variables), y


class FixedEffectsPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, entity_var="name", time_var="date"):
        self.entity_var = entity_var
        self.time_var = time_var
        self.entity_means = None
        self.time_std = None

    def fit(self, X, y=None):
        # Check if the MultiIndex has the required levels
        if not {self.entity_var, self.time_var}.issubset(X.columns):
            raise ValueError("The MultiIndex of X must contain entity and time levels.")
        return self

    def transform(self, X, y=None):
        # Demean the data
        X = X.copy().set_index([self.entity_var, self.time_var])
        X = pd.get_dummies(X, drop_first=True, dtype=int)
        X_demeaned = X - X.groupby(level=self.entity_var).transform("mean")
        return X_demeaned


def add_missing_indicator(X: pd.DataFrame) -> pd.DataFrame:
    """For each variable in the dataset, adds an indicator whether the value is missing."""
    # Check in each column if there is a missing value
    # Make empty dataframe
    missing_df = pd.DataFrame()
    for col in X.columns:
        if X[col].isna().any():
            missing_df[f"missing_{col}"] = X[col].isna().astype(int)
    return missing_df


def extract_treatment_effect(model):
    """Helper function to print the treatment effect from a regression model."""
    # regex the treatment variable
    treatment_beta = [col for col in model.params.index if "treatment" in col][0]
    treatment_pval = [col for col in model.pvalues.index if "treatment" in col][0]
    return pd.Series(
        {"beta": model.params[treatment_beta], "pval": model.pvalues[treatment_pval]}
    )


def simulate_treatment_effect(
    data: pd.DataFrame, treatment_effect: float, target="run_time"
):
    """
    Simulate the treatment effect on the outcome and show the change in ranking for each athlete.
    Parameters:
    - data (pd.DataFrame): The input data containing athlete, outcome, and treatment columns.
    - treatment_effect (float): The estimated treatment effect (beta) from the regression model.
    Returns:
    - ranking_changes (pd.DataFrame): A DataFrame containing the change in ranking for each athlete.
    """
    data = data.copy()
    rank_df = []

    def _maybe_add_or_remove_treatment_effect(row: pd.Series, treatment_nat: str):
        """This function adds the treatment effect to the outcome if the athlete was treated."""
        new_treatment = row.country == treatment_nat
        if row.treatment:
            return row[target] - treatment_effect
        if new_treatment:
            return row[target] + treatment_effect
        return row[target]

    for group, df in tqdm(
        list(
            data.dropna(subset=["date", "gender", "details_competition_type"]).groupby(
                ["date", "gender", "details_competition_type"]
            )
        )
    ):
        for index, row in df.iterrows():
            if row.treatment:
                # Sample one country from the other countries except the one of the athlete
                other_countries = df.country.unique().tolist()
                other_countries.remove(row.country)
                treatment_nat = np.random.choice(other_countries)
            else:
                treatment_nat = row.country
                # Get the adjusted run times
            simulated_run_times = df.drop(index=index).apply(
                lambda x: _maybe_add_or_remove_treatment_effect(x, treatment_nat),
                axis=1,
            )
            simulated_athlete_run_time = _maybe_add_or_remove_treatment_effect(
                row, treatment_nat
            )

            new_row = dict(
                rank_before=(df.drop(index=index).run_time < row.run_time).sum() + 1,
                rank_after=(simulated_run_times < simulated_athlete_run_time).sum() + 1,
                name=row["name"],
                date=row.date,
                gender=row.gender,
                details_competition_type=row.details_competition_type,
                treatment_effect=treatment_effect,
                average_rank_difference=df[target].sort_values().diff().mean(),
            )
            # Append to rank_df
            rank_df.append(new_row)
    return (
        pd.DataFrame.from_records(rank_df)
        .assign(rank_change=lambda x: x.rank_after - x.rank_before)
        .set_index(["name", "date"])
    )


# Data handling --------------------------------------------------------------------------------------------------------

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


def impute_data(df: pd.DataFrame) -> pd.DataFrame:
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


def prepare_data(
    path: str = "data/data_group_1.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    This function prepares the data for the modeling part.
    """
    data_raw = pd.read_csv(path, sep=";")
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
    data["start_time"] = data["start_time"].apply(time_string_to_minutes)

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
    data["season"] = data["date"].apply(assign_season)
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
    data = data.merge(
        total_wcp_nat_last, how="left", on=["season", "country", "gender"]
    )

    data.pop("total_wcp_nat_last_x")
    data.rename(columns={"total_wcp_nat_last_y": "total_wcp_nat_last"}, inplace=True)

    # 1.1 Construct target and treatment ------
    data["treatment"] = data.coursesetter_nat == data.country
    data["z_score"] = data.groupby("race_id", group_keys=False).apply(
        lambda x: (x["run_time"] - x["run_time"].mean()) / x["run_time"].std()
    )

    # 2 Feature engineering--------------------------------------------------

    # Current form of athlete measured in world cup points
    data = convert_bib_to_start_list(data)
    data = assign_wcp_accumulated(
        df=data, target="wpc", col_prefix="wcp"
    )  # wpc overall
    data = assign_wcp_accumulated(
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
            .apply(lambda df: rolling_mean_last_n_days(df, "rank", lag))
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
    data = retrieve_distance_to_tournament(data)
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
    missing_summary = missing_values_summary(
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

    features_athlete_id = ["name", "date"]

    # Impute and save prepared data

    data_prepared = data.copy().filter(
        features_athletes_invariant + features_athletes_variant
    )

    data_ols = impute_data(
        data.copy().filter(features_athletes_invariant + features_athletes_variant)
    )

    data_panel = impute_data(
        data.copy().set_index(features_athlete_id).filter(features_athletes_variant)
    )

    return data_prepared, data_ols, data_panel


def print_variable_summaries_latex(data_prepared: pd.DataFrame) -> pd.DataFrame:
    """
    Print a summary of the variables in the data set in latex format and print it to the console.
    """
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


# Custom OLS -----------------------------------------------------------------------------------------------------------


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


# Describe data --------------------------------------------------------------------------------------------------------


def our_stats_num(data):
    """
    Calculate comparative statics on float64 and int64 variables.

    Parameters
    ----------
    data : TYPE pd.DataFrame
        DESCRIPTION. dataframe to get variables and perform comparative stats on
    """

    # empty dictionary with Key-Value pairs for every column in data
    stats_dict = {}

    # additional col_stats dictionary to store statistics of each column
    # loop over the single columns
    for col in data.columns:
        if data[col].dtype in [
            np.float64,
            np.int64,
        ]:  # check if the column contains numeric values
            col_stats = {
                "mean": round(data[col].mean(), 3),
                "var": round(data[col].var(), 3),
                "sd": round(data[col].std(), 3),
                "max": round(data[col].max(), 3),
                "min": round(data[col].min(), 3),
                "miss_vals": round(data[col].isna().sum(), 3),
                "uni_vals": data[col].nunique(),
                "num_obs": data[col].count(),
            }
            # add key-value pair (column-statistics pair) to the dictionary
            stats_dict[col] = col_stats

    # create a pandas DataFrame from the statistics dictionary
    stats_df = pd.DataFrame.from_dict(stats_dict, orient="index")

    # transpose for names on horizon and stats vertically
    stats_df = stats_df.transpose()

    # return the statistics in table
    return stats_df


def our_stats_string(data):
    """
    Calculate comparative statics on string variables.

    Parameters
    ----------
    data : TYPE pd.DataFrame
        DESCRIPTION. dataframe to get variables and perform comparative stats on
    """

    # empty dictionary with Key-Value pairs for every column in data
    stats_dict = {}

    # additional col_stats dictionary to store statistics of each column
    # loop over the single columns
    for col in data.columns:
        if data[col].dtype not in [
            np.float64,
            np.int64,
        ]:  # check the non numeric values
            col_stats = {
                # unique values
                "uni_vals": data[col].nunique(),
                # dictionary of count for each unique value
                "val_counts": dict(data[col].value_counts()),
                # missing values
                "miss_vals": round(data[col].isna().sum(), 3),
                # most common value
                "mode": data[col].mode().values[0],
            }
            # add key-value pair (column-statistics pair) to the dictionary
            stats_dict[col] = col_stats

    # create a pandas DataFrame from the statistics dictionary
    stats_df = pd.DataFrame.from_dict(stats_dict, orient="index")

    # transpose for names on horizon and stats vertically
    stats_df = stats_df.transpose()

    # return the statistics in table
    return stats_df


# =============================================================================
# Histograms of numeric data
# =============================================================================


def hists_individual(data):
    """
    Plots histogram of continuous variables with more than two unique values.
    Size of bins determined by Scott's rule (based on the sd and sample size).

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe to get data from.

    """
    # randomly select a color
    color = np.random.rand(
        3,
    )

    for col in data.columns:
        # check if the column is continuous and not a dummy
        if data[col].dtype in [float, int] and data[col].nunique() > 2:
            # calculate the bin width using Scott's rule
            std = data[col].dropna().std()
            bin_width = 3.5 * std / (len(data[col]) ** (1 / 3))

            # calculate the number of bins
            min_val = data[col].dropna().min()
            max_val = data[col].dropna().max()
            num_bins = int((max_val - min_val) / bin_width)

            # plot the histogram
            plt.hist(
                data[col].dropna(),
                facecolor=color,
                edgecolor="black",
                alpha=0.7,
                bins=num_bins,
            )

    # set the title
    plt.title(col)

    # show the plot
    plt.show()


def hists_combined(data_sets, save_path: str = None) -> None:
    """
    Plots histograms of continuous variables with more than two unique values
    for multiple datasets.

    Parameters
    ----------
    data_sets : dict
                Dictionary of pd.DataFrame objects to get data from.
                Keys = dataset names, values = corresponding dataframes.
    save_path : str, optional
                Path to save the plot to. The default is None.

    """
    # randomly select a color for each dataset
    colors = [
        np.random.rand(
            3,
        )
        for _ in range(len(data_sets))
    ]
    fig, axs = plt.subplots(ncols=3, nrows=4, figsize=(25, 20))

    idx = 0
    for col in data_sets[list(data_sets.keys())[0]].columns:
        # check if the column is continuous and not a dummy
        if (
            data_sets[list(data_sets.keys())[0]][col].dtype in [float, int]
            and data_sets[list(data_sets.keys())[0]][col].nunique() > 2
        ):
            # get the axis index
            plot_col_idx = idx % 3
            plot_row_idx = idx // 3

            ax = axs[plot_row_idx, plot_col_idx]

            # set the plot title
            ax.set_title(col)

            for i, (data_name, data) in enumerate(data_sets.items()):
                # plot the histogram for the current dataset
                ax.hist(
                    data[col].dropna(),
                    facecolor=colors[i],
                    edgecolor="black",
                    alpha=0.7,
                    bins=30,
                    label=data_name,
                )

            # add a legend to the plot
            ax.legend()
            idx += 1

    if save_path:
        plt.savefig(save_path)
    # show the plot
    plt.show()


def make_combined_hist(data: pd.DataFrame, save_path: str = None) -> None:
    """
    Makes a combined histogram of the numeric covariates in the dataframe and saves it as a png file.
    """
    # conversion to dummy variables + int64
    data_gender = pd.get_dummies(data["gender"]).astype("int64")
    # concatenate the dummy variables with the original dataframe
    data = pd.concat([data, data_gender], axis=1)
    # drop the original gender column and one dummy
    data = data.drop(["gender", "Female"], axis=1)

    # split data into slalom, super-G, giant slalom
    # empty dictionary
    data_sets = {}
    # loop through unique values
    for val in data["details_competition_type"].unique():
        # new dataframe for each unique value
        df_name = "data_" + str(val)
        vars()[df_name] = data[data["details_competition_type"] == val].copy()
        data_sets[df_name] = vars()[df_name]

    # check combined histograms of the numeric covariates for comparatives
    hists_combined(data_sets, save_path=save_path)


# additional for plotting all histograms on one pane
# used for illustration in the paper, same results as previuos function


def hists_combi_one_pane(data_sets, num_cols=4):
    """
    Plots histograms of continuous variables with more than two unique values
    for multiple datasets on one single pane.

    Parameters
    ----------
    data_sets : dict
                Dictionary of pd.DataFrame objects to get data from.
                Keys = dataset names, values = corresponding dataframes.

    """
    # randomly select a color for each dataset
    colors = [
        np.random.rand(
            3,
        )
        for _ in range(len(data_sets))
    ]

    # get the columns that satisfy the condition for plotting a histogram
    cols_to_plot = [
        col
        for col in data_sets[list(data_sets.keys())[0]].columns
        if data_sets[list(data_sets.keys())[0]][col].dtype in [float, int]
        and data_sets[list(data_sets.keys())[0]][col].nunique() > 2
    ]

    # calculate the number of rows and columns for the subplots
    num_plots = len(cols_to_plot)
    num_rows = int(np.ceil(num_plots / num_cols))

    # create the subplots
    fig, axes = plt.subplots(
        nrows=num_rows, ncols=num_cols, figsize=(num_cols * 5, num_rows * 5)
    )

    # flatten the axes array for easier indexing
    axes = axes.flatten()

    for i, col in enumerate(cols_to_plot):
        # set the plot title
        axes[i].set_title(col)

        for j, (data_name, data) in enumerate(data_sets.items()):
            # plot the histogram for the current dataset
            axes[i].hist(
                data[col].dropna(),
                facecolor=colors[j],
                edgecolor="black",
                alpha=0.7,
                bins=30,
                label=data_name,
            )

        # add a legend to the subplot
        axes[i].legend()

    # hide any unused subplots
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    # show the plot
    plt.show()


# =============================================================================
# Correlation matrix
# =============================================================================


def corr_matrix(data):
    """
    Correlation matrix for numeric variables.

    Parameters
    ----------
    data: TYPE. pd.DataFrame
        DESCRIPTION. df containing the variables to calculate correlation of

    Returns
    -------
    pd.DataFrame containing the Pearson correlation coefficients between each
    pair of numeric variables.
    """

    # select only numeric data types from input DataFrame
    numeric_data = data.select_dtypes(include=["float64", "int"])

    # create empty DataFrame with index and columns set to the columns of the numeric data
    corr_df = pd.DataFrame(index=numeric_data.columns, columns=numeric_data.columns)

    # loop over each column in the correlation DataFrame
    for col in corr_df.columns:
        # loop over each other column in the correlation DataFrame
        for other_col in corr_df.columns:
            # calculate the Pearson correlation between the two columns using the corr method of pandas Series
            corr_df[col][other_col] = numeric_data[col].corr(
                numeric_data[other_col], method="pearson"
            )

    # return the completed correlation DataFrame
    return corr_df
