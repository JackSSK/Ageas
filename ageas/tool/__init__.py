#!/usr/bin/env python3
"""
Ageas Reborn

author: jy, nkmtmsys
"""

import re
import gzip
from scipy.stats import pearsonr

# Get pearson correlation value while p-value not lower than thread
# Originally pearson p-value thread was 1, which should be inappropriate
def Get_Pearson(source, target, p_thread = 0.05):
    pearson = pearsonr(source, target)
    if pearson[1] >= p_thread: return 0
    else: return pearson[0]

# Standard Deviation (STD) Filter for data frame(df)
def STD_Filter(df, std_value_thread = None, std_ratio_thread = None):
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
    data.columns = data.columns.str.upper()
    return data.transpose()

# standardize feature scores applying Z score
def Z_Score_Standardize(df, col):
    df[col] = (df[col] - df[col].mean()) / df[col].std(ddof=0)
    return df



class Error(Exception):
    """
    File processing related error handling
    """
    pass



class Reader_Template:
    """
    Template for file reading class
    """
    def __init__(self, filename):
        super(Reader_Template, self).__init__()
        self.filePath = filename
        self.file = None

    # Load in file
    def load(self, filename):
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
