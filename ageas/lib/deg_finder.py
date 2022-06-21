#!/usr/bin/env python3
"""
Ageas Reborn

Note:
D3E seems interesting: https://github.com/hemberg-lab/D3E

author: jy, nkmtmsys
"""

import numpy as np
import statistics as sta
from scipy.stats import mannwhitneyu

# Mann-Whitney U test:  Return True if p-value lower than given threshold
def MWW_Test(list1, list2, threshold = 0.05):
    if threshold is None: return True
    u1, p = mannwhitneyu(
        list1,         # Gene Expression list 1
        list2,          # Gene Expression list 2
        use_continuity = True,
        alternative = 'two-sided',
        axis = 0,
        method = 'asymptotic'
    )
    if p <= threshold: return True
    else: return False

# Calculate absolute value of log2 fold-change between 2 lists
def Log2FC_Calculate(list1, list2):
    return abs(np.log2(sum(list1) + 1) - np.log2(sum(list2) + 1))

# estimate size factor for every sample in GEM(df)
def Estimate_Size_Factors(df):
    count_per_cell = np.squeeze(np.asarray(df.sum(axis=0)))
    size_factors=count_per_cell.astype(np.float64)/np.median(count_per_cell)

    # all add 1 if size factor could be 0
    if 0 in size_factors:
        size_factors = [x+1 for x in size_factors]
    return size_factors



class Find:
    """
    Find Differential Expression Genes(DEGs) based on gene expression counts
    """

    def __init__(self,
                 gem1,                 # Gene Expression Matrix 1 as dataframe
                 gem2,                 # Gene Expression Matrix 2 as dataframe
                 mww_thread = 0.05,    # p-value threshold for MWW test
                 log2fc_thread = None  # minimum log2fc for potential DEGs
                ):
        super(Find, self).__init__()
        # non-intersection genes are automatically set as DEG
        self.degs = {
            gene:None for gene in gem1.index.symmetric_difference(gem2.index)
        }
        g1_factors = Estimate_Size_Factors(gem1)
        g2_factors = Estimate_Size_Factors(gem2)

        for gene in gem1.index.intersection(gem2.index):
            if gene in self.degs: continue
            g1_exps = gem1.loc[[gene]].values[0]
            g2_exps = gem2.loc[[gene]].values[0]

            if log2fc_thread is not None:
                assert len(g1_factors) == len(g1_exps)
                assert len(g2_factors) == len(g2_exps)
                g1_exp = [g1_exps[i] / f for i, f in enumerate(g1_factors)]
                g2_exp = [g2_exps[i] / f for i, f in enumerate(g2_factors)]
                if Log2FC_Calculate(g1_exp, g2_exp) < log2fc_thread:
                    continue

            if mww_thread is not None and MWW_Test(g1_exps,g2_exps,mww_thread):
                self.degs[gene] = None
