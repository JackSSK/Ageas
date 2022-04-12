#!/usr/bin/env python3
"""
Ageas Reborn

author: jy, nkmtmsys
"""

import re
import gzip
from scipy.stats import pearsonr

# Update grn_guidance if given pathway exist in either class
# and be able to pass corelation filter
def Update_GRN_Guidance(grn_guidance,
                        source,
                        target,
                        gem1,
                        gem2,
                        correlation_thread):
    # Skip if processing self-regulating pathway
    if source == target: return
    grp_ID = Cast_GRP_ID(source, target)
    if grp_ID in grn_guidance:
        if not grn_guidance[grp_ID]['reversable']:
            grn_guidance[grp_ID]['reversable'] = True
        return
    # Test out global scale correlation
    cor_class1 = None
    cor_class2 = None
    passed = False
    # check cor_class1
    if source in gem1.index and target in gem1.index:
        cor_class1 = Get_Pearson(gem1.loc[[source]].values[0],
                                gem1.loc[[target]].values[0])
    # check cor_class2
    if source in gem2.index and target in gem2.index:
        cor_class2 = Get_Pearson(gem2.loc[[source]].values[0],
                                gem2.loc[[target]].values[0])
    # Go through abs(correlation) threshod check
    if cor_class1 is None and cor_class2 is None:
        return
    if cor_class1 is None and abs(cor_class2) > correlation_thread:
        passed = True
    elif cor_class2 is None and abs(cor_class1) > correlation_thread:
        passed = True
    elif cor_class1 is not None and cor_class2 is not None:
        if abs(cor_class1 - cor_class2) > correlation_thread:
            passed = True
    # If the testing pass survived till here, save it
    if passed:
        grn_guidance[grp_ID] = {'id': grp_ID,
                                'reversable': False,
                                'regulatory_source': source,
                                'regulatory_target': target,
                                'correlation_in_class1': cor_class1,
                                'correlation_in_class2': cor_class2}


# Get pearson correlation value while p-value not lower than thread
# Originally pearson p-value thread was 1, which should be inappropriate
def Get_Pearson(source, target, p_thread = 0.05):
    pearson = pearsonr(source, target)
    if pearson[1] >= p_thread: return 0
    else: return pearson[0]

# Cast pathway ID based on
def Cast_GRP_ID(source, target):
    if source > target: return source + '_' + target
    else: return target + '_' + source

# Extract all genes influenced among regulon/GRPs
def Get_GRP_Genes(grps):
    answer = {}
    for grp_id in grps:
        genes = grp_id.split('_')
        assert len(genes) == 2
        if genes[0] not in answer: answer[genes[0]] = 0
        if genes[1] not in answer: answer[genes[1]] = 0
    return answer

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

# update dict for counting
def Update_Counting_Dict(dict, ele):
    if ele not in dict: dict[ele] = 1
    else: dict[ele] += 1



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
