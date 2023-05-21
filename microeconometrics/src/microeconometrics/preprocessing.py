import os
from typing import List, Tuple, Any

import pandas as pd
from datetime import timedelta, datetime

from pandas import DataFrame
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import numpy as np
from scipy.stats import pointbiserialr
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.pipeline import Pipeline
import logging
import pycountry
from tqdm import tqdm
import geopy.distance
from geopy.geocoders import Nominatim
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LassoCV
from scipy.stats import zscore


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


def rolling_mean_rank_last_month(group, days: int) -> pd.DataFrame:
    """Computes mean rankings of the athlete in the last month since an event."""
    rolling_mean = []
    for i, row in group.iterrows():
        one_month_ago = row["date"] - timedelta(days=days)
        last_month_rows = group[
            (group["date"] >= one_month_ago)
            & (group["date"] <= (row["date"] - timedelta(days=1)))
        ]
        rolling_mean.append(last_month_rows["rank"].mean())
    group[f"rolling_mean_rank_last_{days}_days"] = rolling_mean
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
        df_distance_map["distance_to_tournament"] = (
            df_distance_map["distance_to_tournament"] / 1000
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
