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



class Predict:
    """
    Predict regulation targets for source TFs
    Essentially, this is stimulating how GRNBoost2 works
    But thread can be automatically set by checking confirmed GRPs
    """
    def __init__(self, gem_data, sampleGRPs = None, thread = None):
        assert sampleGRPs is not None or thread is not None
        self.class1_tGEM = gem_data.class1
        self.class2_tGEM = gem_data.class2
        if thread is not None and thread != 'auto': self.thread = float(thread)
        else: self.thread = self._setThread(sampleGRPs)
        print('GRP prediction thread set to ', self.thread)

    # Expand GRN guide by applying GRNBoost2-like algo on source TFs without
    # documented targets
    def expandGuide(self, grn_guidance, genes, correlation_thread):
        for gene in genes:
            class1FeatImpts, class2FeatImpts = self._getFeatureImportences(gene)
            for i in range(len(self.class1_tGEM.index)):
                if class1FeatImpts is None: break
                tar = self.class1_tGEM.index[i]
                if class1FeatImpts[i] > self.thread:
                    update_grn_guidance(grn_guidance,
                                        gene,
                                        tar,
                                        self.class1_tGEM,
                                        self.class2_tGEM,
                                        correlation_thread)
            for i in range(len(self.class2_tGEM.index)):
                if class2FeatImpts is None: break
                tar = self.class2_tGEM.index[i]
                if class2FeatImpts[i] > self.thread:
                    if tool.Cast_GRP_ID(gene, tar) not in grn_guidance:
                        update_grn_guidance(grn_guidance,
                                            gene,
                                            tar,
                                            self.class1_tGEM,
                                            self.class2_tGEM,
                                            correlation_thread)
        return grn_guidance

    # Automatically set prediction thread by tuning with sample GRPs
    def _setThread(self, sampleGRPs):
        commonSource = {}
        for grp in sampleGRPs:
            source = sampleGRPs[grp]['regulatory_source']
            if source not in commonSource: commonSource[source] = [grp]
            else: commonSource[source].append(grp)
        keyset = sorted([[x, len(commonSource[x])] for x in commonSource],
                            key = lambda x:x[1])
        # Choose a key presenting in both classes
        for i in range(-1, 0):
            key = keyset[i][0]
            degree = keyset[i][1]
            if key in self.class1_tGEM.index and key in self.class2_tGEM.index:
                break
        realGRPs = commonSource[key]
        class1FeatImpts, class2FeatImpts = self._getFeatureImportences(key,True)
        assert len(self.class1_tGEM.index) == len(class1FeatImpts)
        rec = {}
        answer = float('inf')
        for i in range(len(self.class1_tGEM.index)):
            gene = self.class1_tGEM.index[i]
            score = class1FeatImpts[i]
            if score > 0: rec[gene] = score
        for i in range(len(self.class2_tGEM.index)):
            gene = self.class2_tGEM.index[i]
            score = class2FeatImpts[i]
            if score > 0:
                if gene in rec: rec[gene] = (rec[gene] + score)/2
                else: rec[gene] = score
        for gene in rec:
            if key == gene: continue
            grp_ID = tool.Cast_GRP_ID(key, gene)
            if grp_ID in realGRPs and rec[gene] < answer: answer = rec[gene]
        return answer

    # Basically, this part mimicing what GRNBoost2 does
    def _getFeatureImportences(self, key, checked = False):
        if checked or key in self.class1_tGEM.index:
            class1FeatImpts = XGBRegressor(random_state = 0)
            class1FeatImpts.fit(self.class1_tGEM,
                                list(self.class1_tGEM.loc[[key]].values[0]))
            class1FeatImpts = class1FeatImpts.feature_importances_
        else: class1FeatImpts = None
        if checked or key in self.class2_tGEM.index:
            class2FeatImpts = XGBRegressor(random_state = 0)
            class2FeatImpts.fit(self.class2_tGEM,
                                list(self.class2_tGEM.loc[[key]].values[0]))
            class2FeatImpts = class2FeatImpts.feature_importances_
        else: class2FeatImpts = None
        return class1FeatImpts, class2FeatImpts
