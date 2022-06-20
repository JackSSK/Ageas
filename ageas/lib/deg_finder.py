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
            gene:'' for gene in gem1.index.symmetric_difference(gem2.index)
        }
        for gene in gem1.index.intersection(gem2.index):
            group1_gene_exps = gem1.loc[[gene]].values[0]
            group2_gene_exps = gem2.loc[[gene]].values[0]
            log2FC = Log2FC_Calculate(group1_gene_exps, group2_gene_exps)
            # log2FC filter
            if log2fc_thread is not None and log2FC < log2fc_thread: continue
            # MWW filter
            if MWW_Test(group1_gene_exps, group2_gene_exps, mww_thread):
                self.degs[gene] = ''
