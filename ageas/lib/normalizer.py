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
