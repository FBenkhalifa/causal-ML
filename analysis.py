import pandas as pd
from doubleml import DoubleMLPLR
from catboost import CatBoostRegressor, CatBoostClassifier
from doubleml import DoubleMLData
from sklearn.pipeline import make_pipeline
import statsmodels.api as sm
import statsmodels.formula.api as smf
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer  # noqa
import numpy as np
from sklearn.impute import IterativeImputer
import logging
from sklearn.linear_model import LassoCV
from sklearn.linear_model import BayesianRidge, Lasso
from linearmodels import PanelOLS, PooledOLS
from sklearn.preprocessing import OneHotEncoder

from microeconometrics import preprocessing as prep
import seaborn as sns
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# 0. LOAD THE DATA ------------------------------------------------------
data = pd.read_csv("data/data_group_1.csv", sep=";")
data.dropna(subset=["run_time"])

# 1. FEATURE ENGINEERING ------------------------------------------------------
# Prepare the treatment and run time
data["run_time"] = data.run_time.str.split(":|\.").apply(prep.to_miliseconds) / 1000
data.dropna(subset=["run_time"], inplace=True)
data["treatment"] = data.coursesetter_nat == data.country
data["date"] = pd.to_datetime(data["date"])
# Sort string columns by last name
data["coursesetter"] = data["coursesetter"].replace({
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

# Recover the genders for the names with NA,
female_coursesetter = data.query('gender == "Female"').coursesetter.unique()
male_coursesetter = data.query('gender == "Male"').coursesetter

# Get the coursesetters that have set courses in both male and female tournaments
both_coursesetter = male_coursesetter[male_coursesetter.isin(female_coursesetter)].unique()

# Remove the entries from both_coursesetter from male_coursesetter and female_coursesetter
male_coursesetter = np.setdiff1d(male_coursesetter, both_coursesetter)
female_coursesetter = np.setdiff1d(female_coursesetter, both_coursesetter)
# Remove the entries from both_coursesetters from the female and male coursesetters


# Create a mask for rows where 'gender' is NA and 'coursesetter' is in 'male_coursesetter'
mask_male = data['gender'].isna() & data['coursesetter'].isin(male_coursesetter)
mask_female = data['gender'].isna() & data['coursesetter'].isin(female_coursesetter)

# Assign 'Male' to 'gender' where the mask is True
data.loc[mask_male, 'gender'] = 'Male'
data.loc[mask_female, 'gender'] = 'Female'

# Quick eyeball check if names seem male and female
print(data.loc[mask_male, 'name'].unique())
print(data.loc[mask_female, 'name'].unique())

# Get names with gender as NA
names_with_bday_na = data[data['age'].isna()]['name'].unique()
for _, df in data.query('name.isin(@names_with_bday_na)').groupby('name'):
    print(df.boots.unique())

# Make id by concatenating the columns as string 'details_competition_type', 'gender', 'date', 'location', 'country'
data['race_id'] = data[['details_competition_type', 'run', 'gender', 'coursesetter', 'date', 'number_of_gates']].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
data['race_id'].nunique()

# 2. OUTLIER DETECTION ----------------------------------------------------------
data = data.query("vertical_drop > 0")
duplicated = data.duplicated(subset=["name", "date", "start_time"], keep=False)
duplicated_df = data[duplicated].sort_values(by=["name", "start_time"])[["name", "date", "start_time", 'season', 'run', 'run_time', 'total_wcp_nat_last']].head(6)
print(duplicated_df.astype(str).to_latex(escape=True, index=False))
def assign_season(date):
    if date.month >= 10:
        season = f"{date.year}_{date.year + 1}"
    elif date.month <= 3:
        season = f"{date.year - 1}_{date.year}"
    else:
        season = None  # or assign a value for dates outside the ski season
    return season
data['season'] = data['date'].apply(assign_season)
data["start_time"] = data["start_time"].apply(prep.time_string_to_minutes)
data = data.drop_duplicates(subset=["name", "date", "start_time"], keep="first")

#
wpc_last_mask = data.season.isin([ "2020_2021"])
wpc_last_nat = data.groupby(["season", "country", "gender"]).wpc.sum().reset_index().replace(
    {
        '2022_2023': '2023_2024',
        '2021_2022': '2022_2023',
        '2020_2021': '2021_2022',
     }
).rename(columns={"wpc": "total_wcp_nat_last"})

# Replace the total_wcp_nat_last_y with the total_wcp_nat_last_x where the total_wcp_nat_last_y is NA
total_wcp_nat_last = data[wpc_last_mask].groupby(['season', 'country', 'gender']).total_wcp_nat_last.max().reset_index()
total_wcp_nat_last = pd.concat([wpc_last_nat, total_wcp_nat_last], axis=0)
data = data.merge(total_wcp_nat_last, how='left', on=['season', 'country', 'gender'])

# Eyeball the values
data.query('country == "AUT"')[['season', 'total_wcp_nat_last_x', 'total_wcp_nat_last_y', 'country', 'gender']].drop_duplicates().sort_values(['season', 'gender'])
data.pop('total_wcp_nat_last_x')
data.rename(columns={'total_wcp_nat_last_y': 'total_wcp_nat_last'}, inplace=True)

# Repeat for the discipline wpc points
wpc_last_nat_discipline = data.groupby(['season', "country", "gender", "details_competition_type"]).wpc.sum().reset_index().replace(
    {
        '2022_2023': '2023_2024',
        '2021_2022': '2022_2023',
        '2020_2021': '2021_2022',
     }
).rename(columns={"wpc": "total_wcp_discipline_nat_last"})
data = data.merge(wpc_last_nat_discipline, how='left')

# 1.1 Athlete specific features --------------------------------------------------
# Current form of athlete measured in world cup points
data = prep.convert_bib_to_start_list(data)
data = prep.assign_wcp_accumulated(
    df=data,
    target='wpc',
    col_prefix="wcp"
)  # wpc overall
data = prep.assign_wcp_accumulated(
    df=data,
    target="wpc",
    col_prefix="discipline_wcp",
    additional_group_vars= ["details_competition_type", "season"]
)  # wpc per discipline
data["skis"] = data[
    "skis"
].str.lower()  # Rename skis to sponsor (to make it more distinguishable) and make them lowercase
data["age"] = (
    data.date - pd.to_datetime(data["birthdate"])
).dt.days / 365.25  # Compute age of athlete since age stays constant in the orignal dataset

# Assign the cumulative count a coursesetter has been appointed
data["coursesetter_count"] = data.groupby("coursesetter").cumcount()

# Make rolling mean of discipline wpc points of the last month (current form indicator)
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

# Add number of events in the last season/months
# [...]

# 1.2 Country specific performances (influence on the treatment) ---------------
# Assign for each date the accumulated wcp points per country and gender

# 1.3 Tournament specific features -----------------------------------------------------
# Measure how mature the season is, this might have general effects on the performance, motivation etc., importance
for group, df in data.groupby(["season", "gender"]):
    season_start = df.date.min()
    data.loc[df.index, "day_since_season_start"] = (df.date - season_start).dt.days

# World cup points overall
# [...]
# Get the distance to the tournament according to the home country of the athlete
data = prep.retrieve_distance_to_tournament(data)

# 1.4 Course specifics ---------------------------------------------------------
data["gate_per_vertical_meter"] = data["number_of_gates"] / data["vertical_drop"]
# Rename the category 'CORTINA D' AMPEZZO' to 'CORTINA D'AMPEZZO'
data["location"] = data["location"].str.replace(
    "CORTINA D' AMPEZZO", "CORTINA D'AMPEZZO"
)


# 2. FEATURE SELECTION -------------------------------------------------------
# 2.1 Inspect multicollinearity
# select only numeric features
correlation_matrix = data.select_dtypes(include='number').corr()
print("Investigate correlation of numeric features: ")
correlation_matrix

# Get a matrix of the columns indicating if the value is missing
missing = data.isna().astype(int)
# Get table of missing values per column and sort by the number of missing values
def missing_values_summary(df, outcome):
    df = df.copy()
    outcome = df.pop(outcome)
    missing_values = df.isnull().sum()
    missing_values_percent = df.isnull().mean() * 100

    # Create a dataframe to hold missing value indicators
    missing_indicators = df.isnull().astype(int)

    # Calculate correlation of missing indicators with outcome
    outcome_corr = missing_indicators.assign(outcome = outcome).corr()['outcome']

    missing_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Missing (%)': missing_values_percent,
        'Corr of Missing with run time': outcome_corr
    })

    # Sort by the number of missing values
    missing_df = missing_df.sort_values(by='Missing Values', ascending=False)
    return missing_df

# Call the function
missing_summary = missing_values_summary(df=data, outcome='run_time')
print(missing_summary.drop(
    [
        'total_wcp_discipline_nat_last',
        'wpc_fake',
        'rolling_mean_rank_last_30_days',
        'total_rank',
        'total_time',
        'day_since_season_start',
        'distance_to_tournament']
).head(12).to_latex(escape=True, float_format="%.2f"))


missing_values_summary(data, 'run_time')

# # Get the correlation matrix of the missing values with run_time
# missing["ranks"] = data["rank"]
# # Drop all columns that are constant
# missing = missing.loc[:, (missing != missing.iloc[0]).any()]
# correlation_matrix_missing = missing.corr()
# run_time_corr = correlation_matrix_missing.loc['rank']
#
# correlation_matrix_performance = data.filter(
#     regex="acc|wpc|treatment|run_time|total_rank|total_wcp_nat_last|approx|rank"
# ).corr()  # Finding: discipline wpc have higher correlation with run_time than wpc overall while having the similar correlation with treatment
# correlation_matrix_performance
# print(correlation_matrix_performance)
#
# # Detect multicollinearity
# multi_col = prep.detect_multicollinearity(df=data, threshold=0.8, target="run_time")
# print("Detect features with a correlation higher than 0.8: ")
# print(multi_col)
#
# # Inspect missing values
# msno.matrix(data.sample(1000))
# plt.show()
# msno.bar(data.sample(1000))
# plt.show()
# msno.heatmap(data)
# plt.show()

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
    "total_wcp_discipline_nat_last",
    "wcp_relative_to_field",
    'discipline_wcp_relative_to_field',
    'coursesetter_count',
    'rolling_mean_rank_last_30_days',
    # 'gate_per_vertical_meter',
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
# Imputation -------------------------------------------------------
data["z_score"] = data.groupby('race_id', group_keys=False).apply(lambda x: (x["run_time"] - x["run_time"].mean()) / x["run_time"].std())
target = "z_score"

data_imputed = data[data.columns.intersection(features)].dropna(subset=[target]).set_index(["name", 'date'])
data_imputed = pd.get_dummies(data_imputed, drop_first=False, dtype=int)
data_imputed = prep._add_shadow_variable(data_imputed)
data_imputed.columns = data_imputed.columns.str.replace(' ', '_')

data_imputed_matrix = IterativeImputer(estimator=LassoCV(), random_state=0, verbose=2, max_iter=3).fit_transform(X=data_imputed)
data_imputed = pd.DataFrame(data_imputed_matrix, columns=data_imputed.columns, index=data_imputed.index)

# 3. MODELLING ------------------------------------------------------------------
# Standardize the data by group of details_competition_type, gender, date and location
ols_selector = prep.DoubleSelection(target=target, treatment='treatment', alpha=0, controls=["total_wcp_nat_last"])
ols_selector.fit(X=data_imputed)
X, y = ols_selector.transform(X=data_imputed)

X.assign(**{target: y}).reset_index().to_csv("data/ols_data.csv", index=False)
# 3.1 OLS model -----------------------------------------------------------------
formula = f"{target} ~  {' + '.join(X.columns)}"
ols = smf.ols(formula=formula, data=X.assign(**{target: y})).fit()
print("OLS model: ")
print(ols.summary())

print("Print as latext version for overleaf:")
print(ols.summary().as_latex())

# 3,3 Panel data ------------------------------------------------------
data_panel = prep.FixedEffectsPreprocessor(target=target).fit_transform(X=data_imputed.reset_index())
panel_selector = prep.DoubleSelection(target=target, treatment='treatment', alpha=0, controls=["total_wcp_nat_last"])
panel_selector.fit(X=data_panel)
X_panel, y_panel = panel_selector.transform(X=data_panel)
X_panel.assign(**{target: y_panel}).reset_index().to_csv("data/panel_data.csv", index=False)
panel_ols = PanelOLS(
    y,
    sm.add_constant(X),
    entity_effects=True,
    drop_absorbed=True,
    singletons=True,
)
panel_ols.fit()
panel_ols.summary

formula = f"{target} ~  {' + '.join(X_panel.columns)}"
fe_ols = smf.ols(formula=formula, data=X_panel.assign(**{target: y_panel})).fit()
fe_ols.summary()
print("Panel model:")

print("Print as latext version for overleaf:")

print("Treatment effect: ")
prep.extract_treatment_effect(fe_ols_summary)

# 3.4 Partial linear regression -------------------------------------------
plr_features = [
    "z_score",
    # "run_time",  # Outcome
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
from category_encoders import TargetEncoder
data_plr = data_imputed
d = data_imputed[["treatment"]].astype(int)
X_plr = prep.FixedEffectsPreprocessor(target='z_score').fit_transform(X=data_plr.reset_index()).assign(treatment=d.treatment).reset_index(drop=True)
data_container = DoubleMLData(
    # X_panel.assign(**{target: y_panel}).reset_index(drop=True),
    X_plr.reset_index(drop=True),
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
rank_ols = prep.simulate_treatment_effect(data=data,
                                          treatment_effect=prep.extract_treatment_effect(ols).loc["beta"])
print(
    "Hypothetical changes in the ranking if the treatment would be applied to every athlete in a race"
)
rank_ols.rank_change.mean()
rank_ols.rank_change.hist(bins=30)
plt.show()

# Rank changes for panel
rank_panel = prep.simulate_treatment_effect(
    data, prep.extract_treatment_effect(panel_ols_summary).loc["beta"]
)
rank_panel.rank_change.mean()
rank_panel.rank_change.hist(bins=30)
plt.show()

# Rank changes for double ml
rank_double_ml = prep.simulate_treatment_effect(data, double_ml.coef[0])
rank_double_ml.rank_change.mean()
rank_double_ml.rank_change.hist(bins=30)
plt.show()


plot_data = pd.concat(
    [
        rank_ols.assign(model='ols'),
        rank_panel.assign(model='panel-ols'),
        rank_double_ml.assign(model='double-ml'),
    ],
    axis=0,
)
plot_data['model'] = plot_data.model.values + plot_data.treatment_effect.round(3).str.to_list()
# Plot the rank change for each model in one plot
plot = sns.boxplot(data=plot_data,
            x='model',
            y='rank_change',
            palette='Set3',
            linewidth=1.2,
            fliersize=2,
flierprops=dict(marker='o', markersize=4)
            )
plt.show()
plot.figure.savefig('plots/rank_change.pdf')
# Descriptive analysis -------------------------------------------------------
summary = data.groupby(['details_competition_type', 'gender']).describe().T
summary = data.describe().T

# Check missing at random
# plot_category_counts(data, "boots")
# plot_category_counts(data, "country")
# plot_category_counts(data, "location_country")
# plot_all_categories(
#     data, columns=cat_features, ncols=4, title_fontsize=8, label_fontsize=8
# )
