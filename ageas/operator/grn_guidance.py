#!/usr/bin/env python3
"""
Ageas Reborn

ToDo: May need to revise Prediction part

author: jy, nkmtmsys
"""

import warnings
import ageas.tool as tool
import ageas.tool.json as json
import ageas.lib.grn_caster as grn
import ageas.lib.grp_predictor as grp
import ageas.operator as operator

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



class Cast:
    """
    Make GRN Guide based on GEMs
    """
    def __init__(self,
                gem_data = None,
                prediction_thread = None,
                correlation_thread = 0.2):
        # Initialization
        self.guide = {}
        self.tfs_no_interaction_rec = {}
        gene4Pred = None
        # proces guidance casting process based on avaliable information
        if gem_data.interactions is not None:
            self.__with_grtd(gem_data, correlation_thread)
        else:
            self.__no_interaction(gem_data, correlation_thread)
        self.tfs_no_interaction_rec = [x for x in self.tfs_no_interaction_rec]

        # print out stats
        print('Total length of guide:', len(self.guide))
        print(len(self.tfs_no_interaction_rec),
                'potential source TFs not coverd by interaction DB')

        # Start GRNBoost2-like process if thread is set
        if prediction_thread is not None:
            gBoost = grp.Predict(gem_data, self.guide, prediction_thread)
            if len(self.tfs_no_interaction_rec) == 0:   genes = gem_data.genes
            else:   genes = self.tfs_no_interaction_rec
            self.guide = gBoost.expandGuide(self.guide,
                                            genes,
                                            correlation_thread)
            print('With predictions, total length of guide:', len(self.guide))
        # else: raise operator.Error('Sorry, such mode is not supported yet!')
        """ ToDo:if more than 1 guide can be casted, let them make agreement """

    # Make GRN casting guide
    def __with_grtd(self, data, correlation_thread):
        # Iterate source TF candidates for GRP
        for source in data.genes:

            # Go through tf_list filter if avaliable
            if data.tf_list is not None and source not in data.tf_list:
                continue

            # Get Uniprot ID to use GRTD
            uniprot_ids = []
            try:
                for id in data.interactions.idmap[source].split(';'):
                    if id in data.interactions.data:
                         uniprot_ids.append(id)
            except:
                warnings.warn(source, 'not in Uniprot ID Map.')
                pass

            # pass this TF if no recorded interactions in GRTD
            if len(uniprot_ids) == 0:
                    self.tfs_no_interaction_rec[source] = ''
                    continue

            # get potential regulatory targets
            reg_target = {}
            for id in uniprot_ids:
                reg_target.update(data.interactions.data[id])

            # Handle source TFs with no record in target database
            if len(reg_target) == 0:
                if source not in self.tfs_no_interaction_rec:
                    self.tfs_no_interaction_rec[source] = ''
                else:
                    raise operator.Error('Duplicat source TF when __with_grtd')
                break

            # Iterate target gene candidates for GRP
            for target in data.genes:
                # Handle source TFs with record in target database
                if target in reg_target:
                    update_grn_guidance(self.guide,
                                        source,
                                        target,
                                        data.class1,
                                        data.class2,
                                        correlation_thread)

    # Kinda like GRTD version but only with correlation_thread and
    def __no_interaction(self, data, correlation_thread):
        # Iterate source TF candidates for GRP
        for source in data.genes:
            # Go through tf_list filter if avaliable
            if data.tf_list is not None and source not in data.tf_list:
                continue
            for target in data.genes:
                update_grn_guidance(self.guide,
                                    source,
                                    target,
                                    data.class1,
                                    data.class2,
                                    correlation_thread)

    # Save guide file to given path
    def save_guide(self, path):
        json.encode(self.guide, path)
