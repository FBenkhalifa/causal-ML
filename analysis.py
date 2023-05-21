import pandas as pd
from doubleml import DoubleMLPLR
from catboost import CatBoostRegressor, CatBoostClassifier
from doubleml import DoubleMLData
import statsmodels.formula.api as smf
from sklearn.experimental import enable_iterative_imputer  # noqa
import numpy as np
from sklearn.impute import IterativeImputer
import logging
from sklearn.linear_model import BayesianRidge
from microeconometrics import preprocessing as prep

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

# 1. FEATUREs ------------------------------------------------------
# 1.1 Transformations
data["run_time"] = data.run_time.str.split(":|\.").apply(prep.to_miliseconds) / 1000
data.dropna(subset=["run_time"], inplace=True)
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
print(data.loc[mask_male, "name"].unique())
print(data.loc[mask_female, "name"].unique())

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
data = (
    data.sort_values(["name", "date"])
    .groupby("name", group_keys=False)
    .apply(lambda df: prep.rolling_mean_rank_last_month(df, 30))
)
# Forward fill the rolling mean rank
data["rolling_mean_rank_last_30_days"] = (
    data.sort_values(["name", "date"])
    .groupby("name")[["rolling_mean_rank_last_30_days"]]
    .fillna(method="ffill")
)

#  2.1 Coach specific features (influence on the treatment) ---------------
# Assign the cumulative count a coursesetter has been appointed
data["coursesetter_count"] = data.groupby("coursesetter").cumcount()
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
# Measure how mature the season is, this might have general effects on the performance, motivation etc., importance
for group, df in data.groupby(["season", "gender"]):
    season_start = df.date.min()
    data.loc[df.index, "day_since_season_start"] = (df.date - season_start).dt.days

# 1.4 Course specifics ---------------------------------------------------------
data["gate_per_vertical_meter"] = data["number_of_gates"] / data["vertical_drop"]
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

# Selected features
features = [
    # "run_time",  # Outcome
    "z_score",
    "details_competition_type",  # Important because indicates the discipline
    "date",
    "start_altitude",  # Stays in favor of finish_altitude, high correlation
    # "vertical_drop",  # Very high correlation with run_time
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
    # "total_wcp_discipline_nat_last",
    "wcp_relative_to_field",
    "discipline_wcp_relative_to_field",
    "coursesetter_count",
    "rolling_mean_rank_last_30_days",
    "gate_per_vertical_meter",
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
    "country",  # Stays to account for general strength of country
    # "total_time", # not pre treatment
    # "birthdate",
    "skis",  # Replaced by sponsor
    # "boots", # Replace by sponsor
    # "poles", # Replaced by sponsor
    # "rank", # Pre treatment
    # "location_country",  # Replaced by distance_to_tournament
    # "wpc", # Not pre treatment
    # "wpc_fake", # Not pre treatment
    # "acc_wpc",   # In favor of acc_discipline_wpc
    # "acc_country_wpc",  # In favor of acc_country_wpc_discipline
]

# 2.1 Imputation -------
data_imputed = data.copy().set_index(["name", "date"]).filter(features)
data_imputed = pd.get_dummies(data_imputed, drop_first=False, dtype=int)
data_imputed.columns = data_imputed.columns.str.replace(" ", "_")

data_imputed_matrix = IterativeImputer(
    estimator=BayesianRidge(), random_state=0, verbose=2, max_iter=3
).fit_transform(X=data_imputed)
data_imputed = pd.DataFrame(
    data_imputed_matrix, columns=data_imputed.columns, index=data_imputed.index
)
# data_imputed = pd.read_csv("data/data_imputed.csv")

# 3. MODELLING ------------------------------------------------------------------
target = "z_score"

# 3.1 OLS model -----------------------------------------------------------------
data_ols = data_imputed.copy()

# Feature selection
ols_selector = prep.DoubleSelection(
    target=target, treatment="treatment", alpha=0, controls=["total_wcp_nat_last"]
)
ols_selector.fit(X=data_ols)
X_ols, y_ols = ols_selector.transform(X=data_ols)
X_ols.assign(**{target: y_ols}).reset_index().to_csv("data/ols_data.csv", index=False)
X_ols_corr = (
    X_ols.assign(**{target: y_ols}).reset_index().select_dtypes(include="number").corr()
)
print(X_ols_corr)

# OLS model
formula = f"{target} ~  {' + '.join(X_ols.columns)}"
ols = smf.ols(formula=formula, data=X_ols.assign(**{target: y_ols})).fit()
print(ols.summary())

# 3,3 Panel data ------------------------------------------------------
data_panel = data_imputed.copy()

# Conduct within transformation
data_panel = prep.FixedEffectsPreprocessor().fit_transform(X=data_panel.reset_index())

# Feature selection
panel_selector = prep.DoubleSelection(
    target=target, treatment="treatment", alpha=0, controls=["total_wcp_nat_last"]
)
panel_selector.fit(X=data_panel)
X_panel, y_panel = panel_selector.transform(X=data_panel)
X_panel.assign(**{target: y_panel}).reset_index().to_csv(
    "data/panel_data.csv", index=False
)
X_panel_corr = (
    X_panel.assign(**{target: y_panel})
    .reset_index()
    .select_dtypes(include="number")
    .corr()
)
print(X_panel_corr)

# Fixed effects model
formula = f"{target} ~  {' + '.join(X_panel.columns)}"
fe_ols = smf.ols(formula=formula, data=X_panel.assign(**{target: y_panel})).fit()
print(fe_ols.summary())

# 3.4 PLR --------------------------------------------------------------
data_plr = data_imputed.copy()

# Get original treatment variable ()
d = data_plr.pop("treatment").astype(int)
data_plr = (
    prep.FixedEffectsPreprocessor()
    .fit_transform(X=data_plr.reset_index())
    .assign(treatment=d)
    .reset_index(drop=True)
)
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
