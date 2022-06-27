#!/usr/bin/env python3
"""
Ageas Reborn

author: jy, nkmtmsys
"""

import re
import gzip
from scipy.stats import pearsonr


def Get_Pearson(source = None, target = None, p_thread = 0.05):
    """
    Get pearson correlation value while p-value not lower than thread.

    Args:
        source = None

        target = None

        p_thread = 0.05

    """
    pearson = pearsonr(source, target)
    if pearson[1] >= p_thread:
        return 0
    else:
        return pearson[0]


def STD_Filter(df = None, std_value_thread = None, std_ratio_thread = None):
    """
    Standard Deviation (STD) Filter for data frame(df).

    Args:
        df = None

        std_value_thread = None

        std_ratio_thread = None

    """
    data = df.transpose()
    sd_list = data[data.columns].std().sort_values(ascending=False)

    # filter by stdev threshod value
    if std_value_thread is not None:
        for i in range(len(sd_list)):
            if sd_list[i] < std_value_thread: break
        sd_list = sd_list[:i]
    gene_list = list(sd_list.index)

    # filter by ratio thread
    if std_ratio_thread is not None:
        gene_list = gene_list[:int(len(gene_list) * std_ratio_thread)]

    # stratify data
    data = data[gene_list]
    return data.transpose()


def Z_Score_Standardize(df, col):
    """
    Standardize feature scores applying Z score
    """
    df[col] = (df[col] - df[col].mean()) / df[col].std(ddof=0)
    return df


def Check_Feature_Type(features):
    """
    Function to check whether features in GEM are Gene Symbols or Ensembl IDs.
    """
    for ele in features:
        if ele[:3] != 'ENS':
            return 'gene_symbol'
    return 'ens_id'


class Error(Exception):
    """
    File processing related error handling
    """
    pass



class Reader_Template:
    """
    Template for Reader object.
    """
    def __init__(self, filename:str = None):
        """
        Initialize a Reader Object

        Args:
            filename:str = None

        """
        self.filePath = filename
        self.file = None

    # Load in file
    def load(self, filename):
        """
        """
        self.filePath = filename
        # Open as .gz file
        if re.search(r'\.gz$', self.filePath):
            self.file = gzip.open(self.filePath, 'rt', encoding='utf-8')
        # Open directly
        else:
            self.file = open(self.filePath, 'r')

    # Close file reading
    def close(self):
        self.file.close()

    # For iteration
    def __iter__(self):
        return self

    # Need to be override based on need
    def __next__(self):
        return self
