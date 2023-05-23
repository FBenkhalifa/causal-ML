import pandas as pd

data_prepared = pd.read_csv("data/data_prepared.csv", index_col=0)

data_prepared.describe()

data_ols = pd.read_csv("data/data_ols.csv", index_col=0)
data_panel = pd.read_csv("data/data_panel.csv", index_col=0)
