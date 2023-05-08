import os
from typing import List, Tuple

import pandas as pd
from datetime import timedelta
from doubleml import DoubleMLData, DoubleMLPLR, DoubleMLIRM
from catboost import CatBoostRegressor, CatBoostClassifier
from doubleml import DoubleMLData
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import statsmodels.api as sm
import numpy as np
from scipy.stats import pointbiserialr
import statsmodels.formula.api as smf
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
import logging
from sklearn.linear_model import LassoCV, BayesianRidge
from linearmodels import PanelOLS
import pycountry
from tqdm import tqdm
import geopy.distance
from geopy.geocoders import Nominatim

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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


def _compute_acc_wpc(group: pd.DataFrame, name: str):
    group = group.sort_values(["date", "run"])
    group[name] = group["wpc"].fillna(0).cumsum().shift(1).fillna(0)
    return group


def assign_wcp_accumulated(
    df: pd.DataFrame, new_col: str, additional_group: str = None
) -> pd.DataFrame:
    """Computes the accumulated world cup points for an athlete per season (and an additional group)."""
    group = ["name", "season"]
    # Add the additional group if specified
    if additional_group is not None:
        group.append(additional_group)
    # print(df.assign(acc_wpc=acc_wpc)[["name", "wpc", "acc_wpc", "date", "run"]])
    return (
        df.groupby(group)
        .apply(lambda df: _compute_acc_wpc(df, new_col))
        .reset_index(drop=True)
    )


def rolling_mean_rank_last_month(group) -> pd.DataFrame:
    """Computes mean rankings of the athlete in the last month since an event."""
    rolling_mean = []
    for i, row in group.iterrows():
        one_month_ago = row["date"] - timedelta(days=30)
        last_month_rows = group[
            (group["date"] >= one_month_ago)
            & (group["date"] <= (row["date"] - timedelta(days=1)))
        ]
        rolling_mean.append(last_month_rows["rank"].mean())
    group["rolling_mean_rank"] = rolling_mean
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
        df_distance_map = pd.read_parquet("data/distance_to_tournament.parquet")

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
    multicollinear_pairs = pd.DataFrame()
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] >= threshold:
                multicollinear_pairs = multicollinear_pairs.append(
                    dict(
                        var_1=corr_matrix.columns[i],
                        var_2=corr_matrix.columns[j],
                        corr=corr_matrix.iloc[i, j],
                    ),
                    ignore_index=True,
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
                multicollinear_pairs = multicollinear_pairs.append(
                    dict(
                        var_1=num_feature,
                        var_2=cat_feature,
                        corr=pb_corr,
                    ),
                    ignore_index=True,
                )

    return multicollinear_pairs.assign(threshold=threshold)


def _revert_target_encoding(
    data_original,
    data_transformed,
    target,
    encoded_columns,
    tol=0.1,
):
    data_reverted = data_transformed.copy()
    data_original_with_target = data_original.assign(target=target)

    for column in encoded_columns:
        original_column = data_original[column]
        target_column = data_transformed[column]
        class_means = data_original_with_target.groupby(column)["target"].mean()

        for cls, mean in class_means.to_dict().items():
            data_reverted.loc[np.abs(target_column - mean) <= tol, column] = cls

    return data_reverted


from sklearn.linear_model import LassoCV


def _post_lasso(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Get the features selected by the lasso model."""
    X = X.copy()
    y = y.copy()
    # Fit the lasso model
    lasso = LassoCV(cv=5, random_state=42).fit(X, y)
    # Get the coefficients
    coef = lasso.coef_
    return pd.Series(coef[coef != 0], index=X.columns[coef != 0])


def _encode_categorical(X, **kwargs):
    """Encodes categorical features and adds a constant column."""
    cat_features = X.select_dtypes(include=["object", "category"]).columns
    X = pd.get_dummies(X, columns=cat_features, **kwargs)
    X = sm.add_constant(X)
    # Make sure that columns do not contain spaces
    X.columns = [col.replace(" ", "_") for col in X.columns]
    return X


def _get_features_from_post_lasso(X: pd.DataFrame, y: pd.Series, d="treatment"):
    """Extracts the union of features left by a lasso regression of (1) the treatment on the covariates and (2) the outcome on the covariates."""
    # Fit the lasso model on the target
    non_shrunk_target = _post_lasso(X, y)
    non_shrunk_treatment = _post_lasso(X.drop(columns=[d]), X[d])
    non_shrunk = (
        pd.concat([non_shrunk_target, non_shrunk_treatment], axis=0)
        .append(pd.Series({d: np.nan}))
        .index.unique()  # Make sure that treatment is never shrunk
    )
    # Filter out the non-shrunk parameters from the treatment prediction
    logging.warning(
        f"Discarding {', '.join(X.columns.difference(non_shrunk).tolist())} from the treatment prediction"
    )
    return non_shrunk


def prepare_X_y(
    data: pd.DataFrame,
    target: str = "run_time",
    preprocess_pipe: Pipeline = None,
    dummy_encode_and_constant: bool = False,
    add_shadow_features: bool = False,
    post_lasso: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Helper function to preprocess the dataset."""
    X = data.copy().dropna(subset=[target]).set_index(["name", "date"])
    y = X.pop(target)

    if dummy_encode_and_constant:
        X = _encode_categorical(X, drop_first=True)

    if add_shadow_features:
        X = _add_shadow_variable(X)

    if post_lasso:
        X_one_hot = _encode_categorical(X, drop_first=False)
        X_train, X_test, y_train, y_test = train_test_split(
            X_one_hot, y, test_size=0.5, random_state=13
        )
        X_preprocessed = pd.DataFrame(
            preprocess_pipe.fit_transform(X_train, y_train),
            columns=X_train.columns,
            index=X_train.index,
        )
        features = _get_features_from_post_lasso(X=X_preprocessed, y=y_train)
        X = X_test[features]
        y = y_test

    return X, y


def _add_shadow_variable(X: pd.DataFrame):
    """For each variable in the dataset, adds a shadow variable that indicates whether the value is missing."""
    # Check in each column if there is a missing value
    # Make empty dataframe
    X = X.copy()
    for col in X.columns:
        if X[col].isna().any():
            X[f"shadow_{col}"] = X[col].isna().astype(int)
    return X


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
    rank_df = pd.DataFrame()

    def _maybe_add_treatment_effect(row: pd.Series):
        """This function adds the treatment effect to the outcome if the athlete was treated."""
        if row.treatment:
            return row[target]
        else:
            return row[target] + treatment_effect

    def _maybe_remove_treatment_effect(row: pd.Series):
        """This function removes the treatment effect from a competing athlete if the athlete was treated."""
        if row.treatment:
            return row[target] - treatment_effect
        else:
            return row[target]

    for group, df in tqdm(
        list(data.groupby(["date", "gender", "details_competition_type"]))
    ):
        for index, row in df.iterrows():
            # Get the adjusted run times
            run_times = df.drop(index=index).apply(_maybe_add_treatment_effect, axis=1)
            # Get rank before treatment by comparing target
            run_time_athlete = _maybe_remove_treatment_effect(row)

            new_row = dict(
                rank_before=(run_times < run_time_athlete).sum() + 1,
                rank_after=(run_times < (run_time_athlete + treatment_effect)).sum()
                + 1,
                name=row["name"],
                date=row.date,
                gender=row.gender,
                details_competition_type=row.details_competition_type,
                treatment_effect=treatment_effect,
                average_rank_difference=df[target].sort_values().diff().mean(),
            )
            # Append to rank_df
            rank_df = rank_df.append(new_row, ignore_index=True)
    return rank_df.assign(rank_change=lambda x: x.rank_after - x.rank_before).set_index(
        ["name", "date"]
    )


# 0. LOAD THE DATA ------------------------------------------------------
data = pd.read_csv("data/data_group_1.csv", sep=";")

# 1. FEATURE ENGINEERING ------------------------------------------------------
# Prepare the treatment and run time
data["run_time"] = data.run_time.str.split(":|\.").apply(to_miliseconds) / 1000
data.dropna(subset=["run_time"], inplace=True)
data["treatment"] = data.coursesetter_nat == data.country
data["date"] = pd.to_datetime(data["date"])

# 2. OUTLIER DETECTION ----------------------------------------------------------
data = data.query("vertical_drop > 0")
duplicated = data.duplicated(subset=["name", "date", "start_time"])
data = data[~duplicated]

# 1.1 Athlete specific features --------------------------------------------------
# Current form of athlete measured in world cup points
data = convert_bib_to_start_list(data)
data = assign_wcp_accumulated(data, "acc_wpc")  # wpc overall
data = assign_wcp_accumulated(
    data, "acc_discipline_wpc", "details_competition_type"
)  # wpc per discipline
data["sponsor"] = data[
    "skis"
].str.lower()  # Rename skis to sponsor (to make it more distinguishable) and make them lowercase
data["age"] = (
    data.date - pd.to_datetime(data["birthdate"])
).dt.days / 365.25  # Compute age of athlete since age stays constant in the orignal dataset

# Make rolling mean of discipline wpc points of the last month (current form indicator)
data = (
    data.sort_values(["name", "date"])
    .groupby("name")
    .apply(rolling_mean_rank_last_month)
)

# Forward fill the rolling mean rank
data["rolling_mean_rank"] = (
    data.sort_values(["name", "date"])
    .groupby("name")[["rolling_mean_rank"]]
    .fillna(method="ffill")
)

# Add number of events in the last season/months
# [...]

# 1.2 Country specific performances (influence on the treatment) ---------------
# Assign for each date the accumulated wcp points per country and gender
for group, df in data.groupby(["season", "date", "gender", "country"]):
    wpc_country = df.acc_wpc.sum()
    data.loc[df.index, "acc_country_wpc"] = wpc_country

# Assign for each date the accumulated wcp points per country and competition type
for group, df in data.groupby(
    ["season", "date", "gender", "country", "details_competition_type"]
):
    wpc_discipline = df.acc_discipline_wpc.sum()
    data.loc[df.index, "acc_country_wpc_discipline"] = wpc_discipline

# 1.3 Tournament specific features -----------------------------------------------------
# Measure how mature the season is, this might have general effects on the performance, motivation etc., importance
for group, df in data.groupby(["season", "gender"]):
    season_start = df.date.min()
    data.loc[df.index, "day_since_season_start"] = (df.date - season_start).dt.days

# World cup points overall
# [...]

# Get the distance to the tournament according to the home country of the athlete
data = retrieve_distance_to_tournament(data)

# 1.4 Course specifics ---------------------------------------------------------
data["gate_per_vertical_meter"] = data["number_of_gates"] / data["vertical_drop"]
data["start_time"] = data["start_time"].apply(time_string_to_minutes)
# Rename the category 'CORTINA D' AMPEZZO' to 'CORTINA D'AMPEZZO'
data["location"] = data["location"].str.replace(
    "CORTINA D' AMPEZZO", "CORTINA D'AMPEZZO"
)


# 2. FEATURE SELECTION -------------------------------------------------------
# 2.1 Inspect multicollinearity
correlation_matrix = data.corr()
print("Investigate correlation of numeric features: ")
correlation_matrix

correlation_matrix_performance = data.filter(
    regex="acc|wpc|treatment|run_time|total_rank|total_wcp_nat_last|approx|rank"
).corr()  # Finding: discipline wpc have higher correlation with run_time than wpc overall while having the similar correlation with treatment
correlation_matrix_performance
print(correlation_matrix_performance)

# Detect multicollinearity
multi_col = detect_multicollinearity(df=data, threshold=0.8, target="run_time")
print("Detect features with a correlation higher than 0.8: ")
print(multi_col)

# Inspect missing values
msno.matrix(data.sample(1000))
plt.show()
msno.bar(data.sample(1000))
plt.show()
msno.heatmap(data)
plt.show()

# Selected features
features = [
    "run_time",  # Outcome
    "details_competition_type",  # Important because indicates the discipline
    "date",
    "start_altitude",  # Stays in favor of finish_altitude, high correlation
    "vertical_drop",  # Very high correlation with run_time
    "number_of_gates",  # Very high correlation, since it determines how fast one slits through the gates
    "start_time",  # Should be left, to incorporate time effects during one dqy
    "name",
    "season",
    "age",  # Stays in favor of birthdate
    "gender",  # Makes difference
    "run",  # High correlation with treatment indicating that the 'cut/prefiltering' only allows the best athletes to run
    "location",  # Accounts for general characteristics of the location
    "total_wcp_nat_last",  # High correlation with treatment
    "treatment",
    "approx_world_cup_start_list",
    "acc_discipline_wpc",
    "acc_country_wpc_discipline",
    "distance_to_tournament",  # Accounts for proximity advantages
    "day_since_season_start",  # In favor of acc_country_wpc
    # "finish_altitude", # In favor of finish altitude
    # "lenght",  # Very high correlation with run_time but many NAs (consider binning)
    # "rolling_mean_rank",  # Current form of athlete
    # "coursesetter",
    # "coursesetter_nat",  # Might be interesting
    # "total_rank", # Not pre treatment
    # "bib", # Indicator oc current form, dropped in favor of acc_wpc
    # "country",  # Stays to account for general strength of country
    # "total_time", # not pre treatment
    # "birthdate",
    # "sponsor",  # Stays in favor of skis
    # "skis", # Replaced by sponsor
    # "boots", # Replace by sponsor
    # "poles", # Replaced by sponsor
    # "rank", # Pre treatment
    # "location_country",  # Replaced by distance_to_tournament
    # "wpc", # Not pre treatment
    # "wpc_fake", # Not pre treatment
    # "acc_wpc",   # In favor of acc_discipline_wpc
    # "acc_country_wpc",  # In favor of acc_country_wpc_discipline
]

# 3. MODELLING ------------------------------------------------------------------
target = "run_time"
pipe = make_pipeline(
    IterativeImputer(estimator=BayesianRidge(), random_state=0),
)
X, y = prepare_X_y(
    data=data[data.columns.intersection(features)],
    target="run_time",
    preprocess_pipe=pipe,
    dummy_encode_and_constant=False,
    add_shadow_features=True,
    post_lasso=True,
)

# 3.1 OLS model -----------------------------------------------------------------
formula = f"{target} ~ " + " + ".join(X.columns)
ols = smf.ols(formula=formula, data=X.assign(**{target: y})).fit()
print("OLS model: ")
print(ols.summary().as_latex())

print("Print as latext version for overleaf:")
print(ols.summary().as_latex())

print("Treatment effect: ")
extract_treatment_effect(ols)

# 3.2 Gamma GLM with log ---------------------------------------------------------
glm = smf.glm(
    formula=formula,
    data=X.assign(**{target: y}),
    family=sm.families.Gamma(link=sm.families.links.log()),
).fit()
print("GLM model: ")
print(glm.summary())

print("Print as latext version for overleaf:")
print(glm.summary().as_latex())

print("Treatment effect: ")
extract_treatment_effect(glm)

# 3,3 Panel data ------------------------------------------------------
X_panel, y_panel = prepare_X_y(
    data=data[data.columns.intersection(features)],
    target="run_time",
    preprocess_pipe=pipe,
    add_shadow_features=True,
    dummy_encode_and_constant=False,
    post_lasso=True,
)
panel_ols = PanelOLS(
    y_panel,
    X_panel,
    entity_effects=True,
    drop_absorbed=True,
    singletons=True,
)
panel_ols_summary = panel_ols.fit()
print("Panel model:")
print(panel_ols_summary)

print("Print as latext version for overleaf:")
print(panel_ols_summary.summary.as_latex())

print("Treatment effect: ")
extract_treatment_effect(panel_ols_summary)

# 3.4 Double ML approach -------------------------------------------
features = [
    "run_time",  # Outcome
    "details_competition_type",  # Important because indicates the discipline
    "date",
    "start_altitude",  # Stays in favor of finish_altitude, high correlation
    "rolling_mean_rank",  # Current form of athlete
    "vertical_drop",  # Very high correlation with run_time
    "number_of_gates",  # Very high correlation, since it determines how fast one slits through the gates
    "start_time",  # Should be left, to incorporate time effects during one dqy
    "name",
    "country",  # Stays to account for general strength of country
    "season",
    "age",  # Stays in favor of birthdate
    "gender",  # Makes difference
    "sponsor",  # Stays in favor of skis
    "run",  # High correlation with treatment indicating that the 'cut/prefiltering' only allows the best athletes to run
    "location",  # Accounts for general characteristics of the location
    "distance_to_tournament",  # Accounts for proximity advantages
    "total_wcp_nat_last",  # High correlation with treatment
    "treatment",
    "approx_world_cup_start_list",
    "acc_wpc",  # In favor of acc_discipline_wpc
    "acc_discipline_wpc",
    "acc_country_wpc_discipline",
    "acc_country_wpc",  # In favor of acc_country_wpc_discipline
    "day_since_season_start",  # In favor of acc_country_wpc
    # "finish_altitude", # In favor of finish altitude
    # "lenght",  # Very high correlation with run_time but many NAs (consider binning)
    # "coursesetter",
    # "coursesetter_nat",  # Might be interesting
    # "total_rank", # Not pre treatment
    # "bib", # Indicator oc current form, dropped in favor of acc_wpc
    # "total_time", # not pre treatment
    # "birthdate",
    # "skis", # Replaced by sponsor
    # "boots", # Replace by sponsor
    # "poles", # Replaced by sponsor
    # "rank", # Pre treatment
    # "location_country",  # Replaced by distance_to_tournament
    # "wpc", # Not pre treatment
    # "wpc_fake", # Not pre treatment
]
X_double, y_double = prepare_X_y(
    data=data[data.columns.intersection(features)],
    preprocess_pipe=pipe,
    target="run_time",
    add_shadow_features=True,
    dummy_encode_and_constant=True,
    post_lasso=False,
)
data_container = DoubleMLData(
    X_double.assign(**{target: y_double}).reset_index(drop=True),
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

# Best out-of-sample Performance (as reference)
# Learner ml_l RMSE: [[1.41758144]]
# Learner ml_m RMSE: [[0.26943345]]

# 5. Interpretation ------------------------------------------------------
# Simulate treatment effect on the whole dataset
rank_ols = simulate_treatment_effect(data, extract_treatment_effect(ols).loc["beta"])
print(
    "Hypothetical changes in the ranking if the treatment would be applied to every athlete in a race"
)
rank_ols.rank_change.mean()
rank_ols.rank_change.hist(bins=30)
plt.show()

# Rank changes for panel
rank_panel = simulate_treatment_effect(
    data, extract_treatment_effect(panel_ols_summary).loc["beta"]
)
rank_panel.rank_change.mean()
rank_panel.rank_change.hist(bins=30)
plt.show()

# Rank changes for double ml
rank_double_ml = simulate_treatment_effect(data, double_ml.coef[0])
rank_double_ml.rank_change.mean()
rank_double_ml.rank_change.hist(bins=30)
plt.show()

# Descriptive analysis -------------------------------------------------------


# Check missing at random
# plot_category_counts(data, "boots")
# plot_category_counts(data, "country")
# plot_category_counts(data, "location_country")
# plot_all_categories(
#     data, columns=cat_features, ncols=4, title_fontsize=8, label_fontsize=8
# )
