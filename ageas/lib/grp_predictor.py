#!/usr/bin/env python3
"""
Ageas Reborn
Predict potential Gene Regulatory Pathways(GRPs) using GRNBoost-Like algo

ToDo: Need to revice. I broke every thing and this part is not working now
Check performance here, transposed dataframe may not work properly
author: jy, nkmtmsys
"""

import ageas.tool as tool
from xgboost import XGBRegressor
from collections import OrderedDict
from ageas.operator import update_grn_guidance



class Predict:
    """
    Predict regulation targets for source TFs
    Essentially, this is stimulating how GRNBoost2 works
    But thread can be automatically set by checking confirmed GRPs
    """
    def __init__(self, gem_data, sample_grps = None, thread = None):
        assert sample_grps is not None or thread is not None
        self.class1_gem = gem_data.class1
        self.class2_gem = gem_data.class2
        if thread is not None and thread != 'auto':
            self.thread = float(thread)
        else:
            self.thread = self.__auto_set_thread(sample_grps)
        print('GRP prediction thread set to ', self.thread)

    # Expand GRN guide by applying GRNBoost2-like algo on source TFs without
    # documented targets
    def expand_guide(self, grn_guidance, genes, correlation_thread):
        for gene in genes:
            class1FeatImpts, class2FeatImpts = self._getFeatureImportences(gene)
            for i in range(len(self.class1_gem.index)):
                if class1FeatImpts is None: break
                tar = self.class1_gem.index[i]
                if class1FeatImpts[i] > self.thread:
                    update_grn_guidance(grn_guidance,
                                        gene,
                                        tar,
                                        self.class1_gem,
                                        self.class2_gem,
                                        correlation_thread)
            for i in range(len(self.class2_gem.index)):
                if class2FeatImpts is None: break
                tar = self.class2_gem.index[i]
                if class2FeatImpts[i] > self.thread:
                    if tool.Cast_GRP_ID(gene, tar) not in grn_guidance:
                        update_grn_guidance(grn_guidance,
                                            gene,
                                            tar,
                                            self.class1_gem,
                                            self.class2_gem,
                                            correlation_thread)
        return grn_guidance

    # Automatically set prediction thread by tuning with sample GRPs
    # Essentially this part is finding a regulatory source having most
    # regulatory targets, and then find a confirmed GRP with minimum
    # importance predicted by GBM based method
    def __auto_set_thread(self, sample_grps):
        regulatory_sources = {}
        for grp in sample_grps:
            source = sample_grps[grp]['regulatory_source']
            if source not in regulatory_sources:
                regulatory_sources[source] = [grp]
            else:
                regulatory_sources[source].append(grp)
        regulatory_sources = OrderedDict(sorted(regulatory_sources.items(),
                                                key = lambda x: x[1]))
        # Choose a key presenting in both classes
        for src in regulatory_sources:
            if src in self.class1_gem.index and src in self.class2_gem.index:
                break
        grps = regulatory_sources[src]
        class1FeatImpts, class2FeatImpts = self._getFeatureImportences(src,True)
        assert len(self.class1_gem.index) == len(class1FeatImpts)
        rec = {}
        answer = float('inf')
        for i in range(len(self.class1_gem.index)):
            gene = self.class1_gem.index[i]
            score = class1FeatImpts[i]
            if score > 0: rec[gene] = score
        for i in range(len(self.class2_gem.index)):
            gene = self.class2_gem.index[i]
            score = class2FeatImpts[i]
            if score > 0:
                if gene in rec: rec[gene] = (rec[gene] + score)/2
                else: rec[gene] = score
        for gene in rec:
            if src == gene: continue
            grp_ID = tool.Cast_GRP_ID(src, gene)
            if grp_ID in grps and rec[gene] < answer: answer = rec[gene]
        return answer

    # Basically, this part mimicing what GRNBoost2 does
    def _getFeatureImportences(self, key, checked_in_gem = False):
        if checked_in_gem or key in self.class1_gem.index:
            c1_result = self.__gbm_feature_importances_calculat(self.class1_gem,
                                                                key)
        else: c1_result = None
        if checked_in_gem or key in self.class2_gem.index:
            c2_result = self.__gbm_feature_importances_calculat(self.class2_gem,
                                                                key)
        else: c2_result = None
        return c1_result, c2_result

    # as named
    def __gbm_feature_importances_calculat(self, gem, key, random_state = 0):
        gbm = XGBRegressor(random_state = random_state)
        gbm.fit(gem.transpose(), list(gem.loc[[key]].values[0]))
        return gbm.feature_importances_
