#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Microeconometrics: Self Study.

Authors: 
    
    Maximilian Jakob Arrich
    Florian Benkhalifa
    Linda Fiorina Odermatt (17-946-310)
    
Spring Semester 2023.

University of St. Gallen.
"""
# =============================================================================
# Packages
# =============================================================================
import pandas as pd

from microeconometrics import descriptive as ss_fcts

# =============================================================================
# Load the data
# =============================================================================
# load in data using pandas
data = pd.read_csv('data/data_group_1.csv', sep=";")


# =============================================================================
# Descriptive statistics
# =============================================================================
# check data types
data.dtypes

print(data.loc[:, data.dtypes == "object"].columns)
print(data.select_dtypes(include=["int64", "float64"]).columns)


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


# descriptive statistics
# empty dictionaries
stats_num = {}
stats_string = {}
# loop through slalom, giant slalom, super G datasets
for data_set in data_sets:
    # numeric columns statistics report (dataframe) for each
    stats_num[f"stats_num_{data_set}"] = ss_fcts.our_stats_num(data_sets[data_set])
    # string/object columns
    stats_string[f"stats_str_{data_set}"] = ss_fcts.our_stats_string(
        data_sets[data_set]
    )


# check names
stats_num.keys()
stats_string.keys()


# check individual histograms of the covariates
ss_fcts.hists_individual(data_sets["data_Giant Slalom"])

ss_fcts.hists_individual(data_sets["data_Slalom"])

ss_fcts.hists_individual(data_sets["data_Super G"])

# check combined histograms of the numeric covariates for comparatives
ss_fcts.hists_combined(data_sets)


# correlation matrix numeric variables
# loop through the keys in the data_sets dictionary
for data_set_name in data_sets.keys():
    # extract the DataFrame for the current data set
    data_set = data_sets[data_set_name]

    # compute the correlation matrix for the numeric variables
    corr_matrix_num = ss_fcts.corr_matrix(
        data_set.select_dtypes(include=["float64", "int"])
    )

    # print the correlation matrix
    print(f"Correlation matrix for {data_set_name}:")
    print(corr_matrix_num)

import ss_fcts

# create an empty dictionary to store correlation matrices
corr_matrices = {}

# loop over each dataset and calculate the correlation matrix
for dataset_name, dataset in data_sets.items():
    corr_matrix = ss_fcts.corr_matrix(dataset)
    corr_matrices[dataset_name] = corr_matrix
