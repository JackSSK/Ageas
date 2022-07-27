#!/usr/bin/env python3
"""
Ageas Reborn
Predict potential Gene Regulatory Pathways(GRPs) using GRNBoost-Like algo

author: jy, nkmtmsys
"""
import ageas.tool.grn as grn
from xgboost import XGBRegressor
from collections import OrderedDict
import ageas.lib as lib



class Predict:
    """
    Predict regulation targets for source TFs
    Essentially, this is stimulating how GRNBoost2 works
    But thread can be automatically set by checking confirmed GRPs
    """
    def __init__(self, gem_data, sample_grps = None, thread = None):
        super(Predict, self).__init__()
        assert sample_grps is not None or thread is not None
        self.group1_gem = gem_data.group1
        self.group2_gem = gem_data.group2
        # roughly set thread simply with amount of genes in GEMs
        self.thread = 1 / len(gem_data.genes)

        # Set Thread
        if thread is not None and thread != 'auto':
            self.thread = float(thread)
        elif thread == 'auto':
            self.thread = self.__auto_set_thread(sample_grps)
        else:
            raise lib.Error('Predictor thread setting is wrong')

    # Expand meta GRN by applying GRNBoost2-like algo on source TFs without
    # documented targets
    def expand_meta_grn(self, meta_grn, genes, correlation_thread):
        for gene in genes:
            group1_targets, group2_targets = self.__predict_key_targets(gene)
            self.__update_grps_to_meta_grn(
                meta_grn,
                correlation_thread,
                gene,
                self.group1_gem.data.index,
                group1_targets,
            )
            self.__update_grps_to_meta_grn(
                meta_grn,
                correlation_thread,
                gene,
                self.group2_gem.data.index,
                group2_targets,
            )
        return meta_grn

    # decide whether update GRPs associated with given gene into GRN guidance
    def __update_grps_to_meta_grn(self,
                                  meta_grn,
                                  correlation_thread,
                                  gene,
                                  gene_list,
                                  feature_importances,
                                 ):
        if feature_importances is None: return
        for i in range(len(gene_list)):
            if feature_importances[i] > self.thread:
                grp_id = grn.GRP(gene, gene_list[i]).id
                if grp_id not in meta_grn.grps:
                    meta_grn.update_grn(
						source = gene,
						target = gene_list[i],
						gem1 = self.group1_gem,
						gem2 = self.group2_gem,
						correlation_thread = correlation_thread
					)
        return

    # Automatically set prediction thread by tuning with sample GRPs
    # Essentially this part is finding a regulatory source having most
    # regulatory targets, and then find a confirmed GRP with minimum
    # importance predicted by GBM based method
    def __auto_set_thread(self, sample_grps):
        regulatory_sources = {}
        for grp in sample_grps:
            source = sample_grps[grp].regulatory_source
            if source not in regulatory_sources:
                regulatory_sources[source] = [grp]
            else:
                regulatory_sources[source].append(grp)
        regulatory_sources = OrderedDict(
            sorted(regulatory_sources.items(), key = lambda x: x[1])
        )

        # Choose a key presenting in both classes
        grps = None
        for i, src in enumerate(regulatory_sources):
            if (src in self.group1_gem.data.index and
                src in self.group2_gem.data.index):
                grps = regulatory_sources[src]
                group1_targets, group2_targets = self.__predict_key_targets(
                    src, True
                )
                assert len(self.group1_gem.data.index) == len(group1_targets)
                assert len(self.group2_gem.data.index) == len(group2_targets)
                break

        if grps is None:
            return self.thread

        # Extract all genes influenced among regulon/GRPs
        genes = {}
        for grp_id in grps:
            fators = grp_id.split('_')
            assert len(fators) == 2
            if fators[0] not in genes: genes[fators[0]] = 0
            if fators[1] not in genes: genes[fators[1]] = 0

        # assign feature importances to each gene in selected reg source's GRPs
        for i in range(len(self.group1_gem.data.index)):
            if self.group1_gem.data.index[i] in genes and group1_targets[i]>0:
                genes[self.group1_gem.data.index[i]] = group1_targets[i]
        for i in range(len(self.group2_gem.data.index)):
            gene = self.group2_gem.data.index[i]
            if gene in genes and group2_targets[i] > 0:
                if genes[gene] == 0:
                    genes[gene] = group2_targets[i]
                else:
                    genes[gene] = (genes[gene] + group2_targets[i]) / 2

        # take out genes with 0 importances and reorder the dict
        genes = {x:genes[x] for x in genes if genes[x] > 0}
        genes = OrderedDict(sorted(genes.items(), key = lambda x: x[1]))
        return min(genes[next(iter(genes))], self.thread)

    # Basically, this part mimicing what GRNBoost2 does
    def __predict_key_targets(self, key, checked_in_gem = False):
        if checked_in_gem or key in self.group1_gem.data.index:
            group1_result = self.__gbm_feature_importances_calculate(
                self.group1_gem.data,
                key
            )
        else:
            group1_result = None

        if checked_in_gem or key in self.group2_gem.data.index:
            group2_result = self.__gbm_feature_importances_calculate(
                self.group2_gem.data,
                key
            )
        else:
            group2_result = None

        return group1_result, group2_result

    # as named
    def __gbm_feature_importances_calculate(self, gem, key, random_state = 0):
        gbm = XGBRegressor(random_state = random_state)
        gbm.fit(gem.transpose(), list(gem.loc[[key]].values[0]))
        return gbm.feature_importances_
