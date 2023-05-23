from typing import Tuple
import datetime

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from microeconometrics import preprocessing as prep
from sklearn.linear_model import BayesianRidge

from typing import List, Union

import pandas as pd
import numpy as np
from sklearn.utils import check_X_y
from scipy.stats import t, f
from scipy.linalg import qr

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(13)


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
