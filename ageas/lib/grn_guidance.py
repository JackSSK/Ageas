#!/usr/bin/env python3
"""
Ageas Reborn

author: jy, nkmtmsys
"""

import warnings
import ageas.tool as tool
import ageas.tool.json as json
import ageas.lib.pcgrn_caster as grn
import ageas.lib.grp_predictor as grp
import ageas.lib as lib



class Cast:
    """
    Make GRN Guide based on GEMs
    """
    def __init__(self,
                gem_data = None,
                prediction_thread = None,
                correlation_thread = 0.2,
                load_path = None):
        super(Cast, self).__init__()
        # Initialization
        self.guide = {}
        self.tfs_no_interaction_rec = {}
        # Choose process
        if load_path is not None: self.__load(load_path)
        else: self.__cast(gem_data, prediction_thread, correlation_thread)

    def __load(self, load_path):
        self.guide = json.decode(load_path)

    # Process to Cast out GRN construction guidance
    def __cast(self, gem_data, prediction_thread, correlation_thread):
        # proces guidance casting process based on avaliable information
        if gem_data.interactions is not None:
            if gem_data.database_info.interaction_db == 'grtd':
                self.__with_grtd(gem_data, correlation_thread)
            elif gem_data.database_info.interaction_db == 'biogrid':
                self.__with_biogrid(gem_data, correlation_thread)
        else:
            self.__no_interaction(gem_data, correlation_thread)
        self.tfs_no_interaction_rec = [x for x in self.tfs_no_interaction_rec]

        # print out amount of TFs not covered by selected interaction database
        print(len(self.tfs_no_interaction_rec),
                'potential source TFs not coverd by interaction DB')

        # Start GRNBoost2-like process if thread is set
        if prediction_thread is not None and len(self.tfs_no_interaction_rec)>0:
            gBoost = grp.Predict(gem_data, self.guide, prediction_thread)
            """ ToDo: this condition may need to revise """
            if len(self.tfs_no_interaction_rec) == 0:
                genes = gem_data.genes
            else:
                genes = self.tfs_no_interaction_rec
            self.guide = gBoost.expand_guide(self.guide,
                                            genes,
                                            correlation_thread)
        print('Total length of guide:', len(self.guide))
        # else: raise lib.Error('Sorry, such mode is not supported yet!')
        """ ToDo: if more than 1 guide can be casted, make agreement """

    # Make GRN casting guide with GRTD data
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
                    continue
                else:
                    raise lib.Error('Duplicat source TF when __with_grtd')

            # Iterate target gene candidates for GRP
            for target in data.genes:
                # Handle source TFs with record in target database
                if target in reg_target:
                    tool.Update_GRN_Guidance(self.guide,
                                                source,
                                                target,
                                                data.class1,
                                                data.class2,
                                                correlation_thread)

    # Make GRN casting guide with bioGRID data
    def __with_biogrid(self, data, correlation_thread):
        # Iterate source TF candidates for GRP
        for source in data.genes:

            # Go through tf_list filter if avaliable
            if data.tf_list is not None and source not in data.tf_list:
                continue

            reg_target = {}
            if source in data.interactions.data:
                reg_target = {tar:'' for tar in data.interactions.data[source]}
            elif source in data.interactions.alias:
                alias_list = data.interactions.alias[source]
                for ele in alias_list:
                    temp = {tar:'' for tar in data.interactions.data[ele]}
                    reg_target.update(temp)
            else:
                self.tfs_no_interaction_rec[source] = ''
                continue

            # Handle source TFs with no record in target database
            if len(reg_target) == 0:
                if source not in self.tfs_no_interaction_rec:
                    self.tfs_no_interaction_rec[source] = ''
                    continue
                else:
                    raise lib.Error('Duplicat source TF when __with_biogrid')

            for target in data.genes:
                passing = False
                # Handle source TFs with record in target database
                if target in reg_target:
                    passing = True

                elif target in data.interactions.alias:
                    for ele in data.interactions.alias[target]:
                        if ele in reg_target:
                            passing = True

                if passing:
                    tool.Update_GRN_Guidance(self.guide,
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
                tool.Update_GRN_Guidance(self.guide,
                                            source,
                                            target,
                                            data.class1,
                                            data.class2,
                                            correlation_thread)

    # Save guide file to given path
    def save_guide(self, path):
        json.encode(self.guide, path)
