#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Microeconometrics: Self Study Functions.

Authors: 
    
    Maximilian Arrich ()
    Florian Benkhalifa ()
    Linda Fiorina Odermatt (17-946-310)
    
Spring Semester 2023.

University of St. Gallen.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Descriptive statistics
# =============================================================================

def our_stats_num(data):
    '''
    Calculate comparative statics on float64 and int64 variables.

    Parameters
    ----------
    data : TYPE pd.DataFrame
        DESCRIPTION. dataframe to get variables and perform comparative stats on
    '''
    
    # empty dictionary with Key-Value pairs for every column in data
    stats_dict = {}
    
    # additional col_stats dictionary to store statistics of each column
    # loop over the single columns
    for col in data.columns:
        if data[col].dtype in [np.float64, np.int64]:  # check if the column contains numeric values
            col_stats = {
                'mean': round(data[col].mean(),3),
                'var': round(data[col].var(),3),
                'sd': round(data[col].std(),3),
                'max': round(data[col].max(),3),
                'min': round(data[col].min(),3),
                'miss_vals': round(data[col].isna().sum(),3),
                'uni_vals': data[col].nunique(),
                'num_obs': data[col].count(),
            }
            # add key-value pair (column-statistics pair) to the dictionary
            stats_dict[col] = col_stats

    # create a pandas DataFrame from the statistics dictionary
    stats_df = pd.DataFrame.from_dict(stats_dict, orient='index')
    
    # transpose for names on horizon and stats vertically
    stats_df = stats_df.transpose()
    
    # return the statistics in table
    return stats_df


def our_stats_string(data):
    '''
    Calculate comparative statics on string variables.

    Parameters
    ----------
    data : TYPE pd.DataFrame
        DESCRIPTION. dataframe to get variables and perform comparative stats on
    '''
    
    # empty dictionary with Key-Value pairs for every column in data
    stats_dict = {}
    
    # additional col_stats dictionary to store statistics of each column
    # loop over the single columns
    for col in data.columns:
        if data[col].dtype not in [np.float64, np.int64]:  # check the non numeric values
            col_stats = {
                # unique values
                'uni_vals': data[col].nunique(),
                # dictionary of count for each unique value
                'val_counts': dict(data[col].value_counts()),
                # missing values
                'miss_vals': round(data[col].isna().sum(),3),
                # most common value
                'mode': data[col].mode().values[0],
            }
            # add key-value pair (column-statistics pair) to the dictionary
            stats_dict[col] = col_stats

    # create a pandas DataFrame from the statistics dictionary
    stats_df = pd.DataFrame.from_dict(stats_dict, orient='index')
    
    # transpose for names on horizon and stats vertically
    stats_df = stats_df.transpose()
    
    # return the statistics in table
    return stats_df


# =============================================================================
# Histograms of numeric data
# =============================================================================

def hists_individual(data):
    '''
    Plots histogram of continuous variables with more than two unique values.
    Size of bins determined by Scott's rule (based on the sd and sample size).
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataframe to get data from.

    '''
    # randomly select a color
    color = np.random.rand(3,)
    
    for col in data.columns:
        
        # check if the column is continuous and not a dummy
        if data[col].dtype in [float, int] and data[col].nunique() > 2:
            
            # calculate the bin width using Scott's rule
            std = data[col].dropna().std()
            bin_width = 3.5 * std / (len(data[col]) ** (1/3))
            
            # calculate the number of bins
            min_val = data[col].dropna().min()
            max_val = data[col].dropna().max()
            num_bins = int((max_val - min_val) / bin_width)
            
            # plot the histogram
            plt.hist(data[col].dropna(), facecolor=color, edgecolor='black', alpha=0.7, bins=num_bins)
           
            # set the title
            plt.title(col)
            
            # show the plot
            plt.show()


def hists_combined(data_sets):
    '''
    Plots histograms of continuous variables with more than two unique values 
    for multiple datasets.
    
    Parameters
    ----------
    data_sets : dict
                Dictionary of pd.DataFrame objects to get data from.
                Keys = dataset names, values = corresponding dataframes.

    '''
    # randomly select a color for each dataset
    colors = [np.random.rand(3,) for _ in range(len(data_sets))]
    
    for col in data_sets[list(data_sets.keys())[0]].columns:
        
        # check if the column is continuous and not a dummy
        if data_sets[list(data_sets.keys())[0]][col].dtype in [float, int] and data_sets[list(data_sets.keys())[0]][col].nunique() > 2:
            
            # create a new plot
            fig, ax = plt.subplots()
            
            # set the plot title
            ax.set_title(col)
            
            for i, (data_name, data) in enumerate(data_sets.items()):
                
                # plot the histogram for the current dataset
                ax.hist(data[col].dropna(), facecolor=colors[i], edgecolor='black', alpha=0.7, bins=30, label=data_name)
            
            # add a legend to the plot
            ax.legend()
            
            # show the plot
            plt.show()


# additional for plotting all histograms on one pane
# used for illustration in the paper, same results as previuos function

def hists_combi_one_pane(data_sets):
    '''
    Plots histograms of continuous variables with more than two unique values 
    for multiple datasets on one single pane.
    
    Parameters
    ----------
    data_sets : dict
                Dictionary of pd.DataFrame objects to get data from.
                Keys = dataset names, values = corresponding dataframes.

    '''
    # randomly select a color for each dataset
    colors = [np.random.rand(3,) for _ in range(len(data_sets))]
    
    # get the columns that satisfy the condition for plotting a histogram
    cols_to_plot = [col for col in data_sets[list(data_sets.keys())[0]].columns 
                    if data_sets[list(data_sets.keys())[0]][col].dtype in [float, int] 
                    and data_sets[list(data_sets.keys())[0]][col].nunique() > 2]
    
    # calculate the number of rows and columns for the subplots
    num_plots = len(cols_to_plot)
    num_rows = int(np.ceil(np.sqrt(num_plots)))
    num_cols = int(np.ceil(num_plots/num_rows))
    
    # create the subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols*5, num_rows*5))
    
    # flatten the axes array for easier indexing
    axes = axes.flatten()
    
    for i, col in enumerate(cols_to_plot):
        
        # set the plot title
        axes[i].set_title(col)
        
        for j, (data_name, data) in enumerate(data_sets.items()):
            
            # plot the histogram for the current dataset
            axes[i].hist(data[col].dropna(), facecolor=colors[j], edgecolor='black', alpha=0.7, bins=30, label=data_name)
        
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
    numeric_data = data.select_dtypes(include=['float64', 'int'])
    
    # create empty DataFrame with index and columns set to the columns of the numeric data
    corr_df = pd.DataFrame(index=numeric_data.columns, columns=numeric_data.columns)
    
    # loop over each column in the correlation DataFrame
    for col in corr_df.columns:
        
        # loop over each other column in the correlation DataFrame
        for other_col in corr_df.columns:
            
            # calculate the Pearson correlation between the two columns using the corr method of pandas Series
            corr_df[col][other_col] = numeric_data[col].corr(numeric_data[other_col], method='pearson')
    
    # return the completed correlation DataFrame
    return corr_df























