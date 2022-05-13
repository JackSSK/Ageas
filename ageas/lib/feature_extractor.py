#!/usr/bin/env python3
"""
Ageas Reborn

author: jy, nkmtmsys
"""

import ageas.tool as tool
import ageas.tool.json as json
from collections import OrderedDict


class Extract(object):
    """
    Extract key genes from the most important GRPs
    """

    def __init__(self,
                grp_importances = None,
                score_thread = None,
                outlier_grps = None,
                top_grp_amount = None):
        super(Extract, self).__init__()
        self.key_genes = None
        self.common_reg_source = None
        self.common_reg_target = None
        self.outlier_grps = outlier_grps
        self.grps = grp_importances.stratify(score_thread,
                                            top_grp_amount,
                                            len(outlier_grps))

    # extract common regulaoty sources or targets of given genes
    def extract_common(self,
                        grn_guidance,
                        type = 'regulatory_source',
                        occurrence_thread = 2):
        if type == 'regulatory_source': known = 'regulatory_target'
        elif type == 'regulatory_target': known = 'regulatory_source'
        self.key_genes = self.__extract_genes(self.grps, self.outlier_grps)
        genes = [x[0] for x in self.key_genes]
        dict = {}
        for grp in grn_guidance:
            record = grn_guidance[grp]
            if record[known] in genes:
                target = record[type]
                if target not in dict:
                    dict[target] = {
                        'relate': [{record[known]:self.__makeEle(record)}],
                        'influence': 1
                    }
                else:
                    dict[target]['relate'].append(
                                        {record[known]:self.__makeEle(record)})
                    dict[target]['influence'] += 1
        dict = {ele:dict[ele]
                for ele in dict
                    if dict[ele]['influence'] >= occurrence_thread}
        dict = OrderedDict(sorted(dict.items(),
                                    key = lambda x: x[1]['influence'],
                                    reverse = True))
        if type == 'regulatory_source':      self.common_reg_source = dict
        elif type == 'regulatory_target':    self.common_reg_target = dict

    # save files in json format
    def save(self, folder_path):
        json.encode(self.key_genes,         folder_path + 'repeated_factors.js')
        json.encode(self.common_reg_source, folder_path + 'common_source.js')
        json.encode(self.common_reg_target, folder_path + 'common_target.js')

    # extract genes based on whether occurence in important GRPs passing thread
    def __extract_genes(self, stratified_grps, outlier_grps):
        dict = {}
        for ele in stratified_grps.index.tolist():
            score = stratified_grps.loc[ele]['importance']
            ele = ele.strip().split('_')    # get source and target from GRP ID
            if ele[0] not in dict: dict[ele[0]] = score
            else: dict[ele[0]] += score
            if ele[1] not in dict: dict[ele[1]] = score
            else: dict[ele[1]] += score
        for ele in outlier_grps:
            score = ele[1]
            ele = ele[0].strip().split('_')
            if ele[0] not in dict: dict[ele[0]] = score
            else: dict[ele[0]] += score
            if ele[1] not in dict: dict[ele[1]] = score
            else: dict[ele[1]] += score
        # filter by top_grp_amount
        answer = [[e, dict[e]] for e in dict]
        answer.sort(key = lambda x:x[-1], reverse = True)
        return answer

    # Add correlation in class 1 and class 2 into regulon record
    def __makeEle(self, rec):
        return {k:rec[k] for k in rec if k not in ['id','regulatory_source',
                                                        'regulatory_target']}
