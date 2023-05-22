import pandas as pd
from doubleml import DoubleMLDID
from catboost import CatBoostRegressor, CatBoostClassifier
from doubleml import DoubleMLData, DoubleMLPLR
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

# 1. FEATURE ENGINEERING ------------------------------------------------------
# Prepare the treatment and run time
data["run_time"] = data.run_time.str.split(":|\.").apply(prep.to_miliseconds) / 1000
data.dropna(subset=["run_time"], inplace=True)
data["treatment"] = data.coursesetter_nat == data.country
data["date"] = pd.to_datetime(data["date"])

# 2. OUTLIER DETECTION ----------------------------------------------------------
data = data.query("vertical_drop > 0")
duplicated = data.duplicated(subset=["name", "date", "start_time"])
data = data[~duplicated]

# 1.1 Athlete specific features --------------------------------------------------
# Current form of athlete measured in world cup points
data = prep.convert_bib_to_start_list(data)
data = prep.assign_wcp_accumulated(data, "acc_wpc")  # wpc overall
data = prep.assign_wcp_accumulated(
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
    .groupby("name", group_keys=False)
    .apply(prep.rolling_mean_rank_last_month)
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
data = prep.retrieve_distance_to_tournament(data)

# 1.4 Course specifics ---------------------------------------------------------
data["gate_per_vertical_meter"] = data["number_of_gates"] / data["vertical_drop"]
data["start_time"] = data["start_time"].apply(prep.time_string_to_minutes)
# Rename the category 'CORTINA D' AMPEZZO' to 'CORTINA D'AMPEZZO'
data["location"] = data["location"].str.replace(
    "CORTINA D' AMPEZZO", "CORTINA D'AMPEZZO"
)


# 2. FEATURE SELECTION -------------------------------------------------------
# 2.1 Inspect multicollinearity
# select only numeric features
correlation_matrix = data.select_dtypes(include="number").corr()
print("Investigate correlation of numeric features: ")
correlation_matrix

correlation_matrix_performance = data.filter(
    regex="acc|wpc|treatment|run_time|total_rank|total_wcp_nat_last|approx|rank"
).corr()  # Finding: discipline wpc have higher correlation with run_time than wpc overall while having the similar correlation with treatment
correlation_matrix_performance
print(correlation_matrix_performance)

# Detect multicollinearity
multi_col = prep.detect_multicollinearity(df=data, threshold=0.8, target="run_time")
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
    # "details_competition_type",  # Important because indicates the discipline
    "date",
    # "start_altitude",  # Stays in favor of finish_altitude, high correlation
    # "vertical_drop",  # Very high correlation with run_time
    # "number_of_gates",  # Very high correlation, since it determines how fast one slits through the gates
    "start_time",  # Should be left, to incorporate time effects during one dqy
    "name",
    # "season",
    "age",  # Stays in favor of birthdate
    # "gender",  # Makes difference
    # "run",  # High correlation with treatment indicating that the 'cut/prefiltering' only allows the best athletes to run
    # "location",  # Accounts for general characteristics of the location
    "total_wcp_nat_last",  # High correlation with treatment
    "treatment",
    "approx_world_cup_start_list",
    "rolling_mean_rank",
    # "acc_discipline_wpc",
    # "acc_country_wpc_discipline",
    "distance_to_tournament",  # Accounts for proximity advantages
    # "day_since_season_start",  # In favor of acc_country_wpc
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

# Adjust the target variable
data["run_id"] = (
    data["date"].astype(str)
    + data["run"].astype(str)
    + data["details_competition_type"]
    + data["number_of_gates"].astype(str)
    + data["start_altitude"].astype(str)
)
std_target = data.groupby("run_id", as_index=False)[target].apply(
    lambda x: (x - x.mean()) / x.std()
)
# get original index
std_target = std_target.reset_index()
std_target.set_index("level_1", inplace=True)
data[target] = std_target[target]
data.pop("run_id")

X = data[data.columns.intersection(features)].copy()
X.set_index(["date", "name"], inplace=True)
y = X.pop(target)

# data["country"].value_counts()

# 3.1 OLS model -----------------------------------------------------------------
formula = f"target ~ " + " + ".join(X.columns)
ols = smf.ols(formula=formula, data=X.assign(target=y)).fit()
print("OLS model: ")
print(ols.summary())
print(ols.summary().as_latex())

print("Print as latext version for overleaf:")
print(ols.summary().as_latex())

print("Treatment effect: ")
prep.extract_treatment_effect(ols)

# 3.2 Gamma GLM with log ---------------------------------------------------------
glm = smf.glm(
    formula=formula,
    data=X.assign(target=y),
    # family=sm.families.Gamma(),
).fit()

print("GLM model: ")
print(glm.summary())

print("Print as latext version for overleaf:")
print(glm.summary().as_latex())

print("Treatment effect: ")
prep.extract_treatment_effect(glm)

# 3,3 Panel data ------------------------------------------------------
X_panel, y_panel = prep.prepare_X_y(
    data=data[data.columns.intersection(features)],
    target="run_time",
    preprocess_pipe=pipe,
    add_shadow_features=True,
    dummy_encode_and_constant=False,
    post_lasso=True,
)
y_panel = y.copy()
X_panel = X.copy()

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
prep.extract_treatment_effect(panel_ols_summary)

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
X_double, y_double = prep.prepare_X_y(
    data=data[data.columns.intersection(features)],
    target="run_time",
    add_shadow_features=True,
    dummy_encode_and_constant=True,
    post_lasso=False,
)
data_container = DoubleMLData(
    X.assign(target=y).reset_index(drop=True),
    y_col="target",
    d_cols="treatment",
    force_all_x_finite="allow-nan",
)
double_ml = DoubleMLPLR(
    data_container, CatBoostRegressor(), CatBoostClassifier(), n_folds=10, n_rep=5
)
double_ml.fit()
print(double_ml)

# ------------------ Fit summary       ------------------
#                coef   std err         t     P>|t|     2.5 %    97.5 %
# treatment -0.058373  0.030424 -1.918623  0.055032 -0.118003  0.001258


print(double_ml)

# Best out-of-sample Performance (as reference)
# Learner ml_l RMSE: [[1.41758144]]
# Learner ml_m RMSE: [[0.26943345]]

# 5. Interpretation ------------------------------------------------------
# Simulate treatment effect on the whole dataset
rank_ols = prep.simulate_treatment_effect(
    data=data, treatment_effect=prep.extract_treatment_effect(ols).loc["beta"]
)
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
        rank_ols.assign(model="ols"),
        rank_panel.assign(model="panel-ols"),
        rank_double_ml.assign(model="double-ml"),
    ],
    axis=0,
)
plot_data["model"] = (
    plot_data.model.values + plot_data.treatment_effect.round(3).str.to_list()
)
# Plot the rank change for each model in one plot
plot = sns.boxplot(
    data=plot_data,
    x="model",
    y="rank_change",
    palette="Set3",
    linewidth=1.2,
    fliersize=2,
    flierprops=dict(marker="o", markersize=4),
)
plt.show()
plot.figure.savefig("plots/rank_change.pdf")
# Descriptive analysis -------------------------------------------------------


# Check missing at random
# plot_category_counts(data, "boots")
# plot_category_counts(data, "country")
# plot_category_counts(data, "location_country")
# plot_all_categories(
#     data, columns=cat_features, ncols=4, title_fontsize=8, label_fontsize=8
# )
