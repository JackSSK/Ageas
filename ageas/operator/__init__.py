#!/usr/bin/env python3
"""
Ageas Reborn

author: jy, nkmtmsys
"""

import ageas.tool as tool

# Update grn_guidance if given pathway exist in either class
# and be able to pass corelation filter
def update_grn_guidance(grn_guidance,
                        source,
                        target,
                        gem1,
                        gem2,
                        correlation_thread):
    # Skip if processing self-regulating pathway
    if source == target: return
    grp_ID = tool.Cast_GRP_ID(source, target)
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
        cor_class1 = tool.Get_Pearson(gem1.loc[[source]].values[0],
                                        gem1.loc[[target]].values[0])
    # check cor_class2
    if source in gem2.index and target in gem2.index:
        cor_class2 = tool.Get_Pearson(gem2.loc[[source]].values[0],
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


class Error(Exception):
    """
    Error handling
    """
    pass
