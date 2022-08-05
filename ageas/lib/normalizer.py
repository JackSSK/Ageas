#!/usr/bin/env python3
"""
Ageas Reborn
Normalization related stuffs here.

author: jy, nkmtmsys
"""

def CPM(df):
    """
    Counts Per Million (CPM) normalization
    """
    for sample in df.columns:
        df[sample] = (df[sample] * 1000000) / sum(df[sample])
    return df

def Min_Max_100(df):
    """
    Values multiplied by 100 after Min-Max Normalization
    """
    return 100 * (df - df.min())/ (df.max() - df.min())
