#!/usr/bin/env python3
"""
Ageas Reborn

author: jy, nkmtmsys
"""

import re
import os
import time
import warnings
from pkg_resources import resource_filename
import ageas.tool.json as json
import ageas.lib.config_maker as config_maker
import ageas.lib.pcgrn_caster as grn
import ageas.lib.trainer as trainer
from ageas.lib.gem_db_loader import Load
import ageas.lib.grn_guidance as grn_guidance
import ageas.lib.model_interpreter as interpreter
import ageas.lib.feature_extractor as extractor
import ageas.database_setup.binary_class as binary_db



class Find:
    """
    Get candidate key factors and pathways
    and write report files into given folder
    """
    def __init__(self,
                database_path = None,
                database_type = 'gem_file',
                class1_path = None,
                class2_path = None,
                specie = 'mouse',
                sliding_window_size = 10,
                sliding_window_stride = None,
                factor_name_type = 'gene_name',
                interaction_db = 'grtd',
                model_config_path = None,
                std_value_thread = 100,
                std_ratio_thread = None,
                mww_thread = 0.05,
                log2fc_thread = 0.1,
                prediction_thread = 'auto',
                correlation_thread = 0.2,
                model_select_iteration = 2,
                feature_select_iteration = 2,
                patient = 3,
                key_gene_change_thread = 0.1,
                train_size = 0.7,
                clf_keep_ratio = 0.5,
                clf_accuracy_thread = 0.9,
                z_score_extract_thread = 2,
                z_score_keep_thread = -1,
                score_sum_thread = 5,
                stabilize_iteration = None,
                warning = False):
        super(Find, self).__init__()
        """ Initialization """
        start = time.time()
        if not warning: warnings.filterwarnings('ignore')
        self.patient = patient
        self.model_select_iteration = model_select_iteration
        self.feature_select_iteration = feature_select_iteration
        self.stabilize_iteration = stabilize_iteration
        self.key_gene_change_thread = key_gene_change_thread
        self.no_change_iteration_num = 0
        # Set up database path info
        self.database_info = binary_db.Setup(database_path,
                                            database_type,
                                            class1_path,
                                            class2_path,
                                            specie,
                                            factor_name_type,
                                            interaction_db,
                                            sliding_window_size,
                                            sliding_window_stride)
        # load standard config file if not further specified
        if model_config_path is None:
            path = resource_filename(__name__, 'data/config/list_config.js')
            self.model_config = config_maker.Default_Config(path)
        else:
            self.model_config = json.decode(model_config_path)
        print('Time to initialize : ', time.time() - start)



        """ Make or load pcGRNs and GRN construction guidance """
        start = time.time()
        self.circe, pcGRNs = self.get_pcGRNs(database_info = self.database_info,
                                        std_value_thread = std_value_thread,
                                        std_ratio_thread = std_ratio_thread,
                                        mww_thread = mww_thread,
                                        log2fc_thread = log2fc_thread,
                                        prediction_thread = prediction_thread,
                                        correlation_thread = correlation_thread,
                                        guide_load_path = None,)
        # pcGRNs.save('data/pcGRN_easy.js')
        # self.circe.save_guide(path = 'data/guide_grtd_easy.js')
        # pcGRNs = grn.Make(load_path = 'data/pcGRN_easy.js')
        # self.circe = grn_guidance.Cast(load_path = 'data/guide_grtd_easy.js')
        print('Time to cast pcGRNs : ', time.time() - start)


        """ Model Selection """
        print('Entering Model Selection')
        start = time.time()
        clfs = trainer.Train(pcGRNs = pcGRNs,
                            database_info = self.database_info,
                            model_config = self.model_config,)
        clfs.successive_halving_process(iteration = self.model_select_iteration,
                                        clf_keep_ratio = clf_keep_ratio,
                                        clf_accuracy_thread=clf_accuracy_thread,
                                        last_train_size = train_size)
        print('Finished Model Selection', time.time() - start)
        start = time.time()
        self.penelope = interpreter.Interpret(clfs)
        self.factors = extractor.Extract(self.penelope,
                                        z_score_extract_thread,
                                        score_sum_thread)
        print('Time to interpret 1st Gen classifiers : ', time.time() - start)



        """ Feature Selection """
        if self.feature_select_iteration is not None:
            print('Entering Feature Selection')
            for i in range(self.feature_select_iteration):
                start = time.time()
                prev_key_genes = self.factors.key_genes
                rm = self.__get_grp_remove_list(self.penelope.feature_score,
                                                z_score_keep_thread)
                pcGRNs.update_with_remove_list(rm)
                clfs.clear_data()
                clfs.grns = pcGRNs
                clfs.general_process(train_size = train_size,
                                    clf_keep_ratio = clf_keep_ratio,
                                    clf_accuracy_thread = clf_accuracy_thread)
                self.penelope = interpreter.Interpret(clfs)
                self.factors = extractor.Extract(self.penelope,
                                                z_score_extract_thread,
                                                score_sum_thread)
                print('Time to do a feature selection : ', time.time() - start)
                if self.__early_stop(prev_key_genes, self.factors.key_genes):
                    self.stabilize_iteration = None
                    break



        """ Stabilizing Output """
        if self.stabilize_iteration is not None:
            print('Stabilizing Output')



        """ final part, get common source and common targets """
        start = time.time()
        self.factors.extract_common(self.circe.guide, type='regulatory_source')
        self.factors.extract_common(self.circe.guide, type='regulatory_target')
        print('Time to do everything else : ', time.time() - start)


    # get pseudo-cGRNs from GEMs or GRNs
    def get_pcGRNs(self,
                    database_info = None,
                    std_value_thread = 100,
                    std_ratio_thread = None,
                    mww_thread = 0.05,
                    log2fc_thread = 0.1,
                    prediction_thread = 'auto',
                    correlation_thread = 0.2,
                    guide_load_path = None,):
        guide = None
        # if reading in GEMs, we need to construct pseudo-cGRNs first
        if re.search(r'gem' , database_info.type):
            gem_data = Load(database_info,
                            mww_thread,
                            log2fc_thread,
                            std_value_thread)
            start1 = time.time()
            # Let kirke casts GRN construction guidance first
            guide = grn_guidance.Cast(gem_data = gem_data,
                                        prediction_thread = prediction_thread,
                                        correlation_thread = correlation_thread,
                                        load_path = guide_load_path)
            print('Time to cast GRN Guidnace : ', time.time() - start1)
            pcGRNs = grn.Make(database_info = database_info,
                                std_value_thread = std_value_thread,
                                std_ratio_thread = std_ratio_thread,
                                correlation_thread = correlation_thread,
                                gem_data = gem_data,
                                grn_guidance = guide.guide)
        # if we are reading in GRNs directly, just process them
        elif re.search(r'grn' , database_info.type):
            pcGRNs = None
            print('trainer.py: mode grn need to be revised here')
        else:
            raise lib.Error('Unrecogonized database type: ', database_info.type)
        return guide, pcGRNs

    # take out some GRPs based on Z score keeping thread
    def __get_grp_remove_list(self, feature_score, z_score_keep_thread):
        remove_list = []
        for i in reversed(range(len(feature_score.index))):
            grp_id = feature_score.index[i]
            z_score = feature_score.iloc[i]['importance']
            if z_score < z_score_keep_thread:
                remove_list.append(grp_id)
            else: break
        return remove_list

    # Stop iteration if key genes are not really changing
    def __early_stop(self, prev_key_genes, cur_key_genes):
        # just keep going if patient not set
        if self.patient is None: return False
        prev_key_genes = [x[0] for x in prev_key_genes]
        cur_key_genes = [x[0] for x in cur_key_genes]
        common = list(set(prev_key_genes).intersection(cur_key_genes))
        change1 = (len(prev_key_genes) - len(common)) / len(prev_key_genes)
        change2 = (len(cur_key_genes) - len(common)) / len(cur_key_genes)
        change = (change1 + change2) / 2
        print('Average Key Genes Changing Portion:', change)
        if change <= self.key_gene_change_thread:
            self.no_change_iteration_num += 1
            if self.no_change_iteration_num == self.patient:
                print('Run out of patient! Early stopping!')
                return True
            else: return False
        else:
            self.no_change_iteration_num = 0
            return False


    # Write report files in give folder
    def writeReport(self, reportPath = 'report/'):
        if reportPath[-1] != '/': reportPath += '/'
        # Make path if not exist
        if not os.path.exists(reportPath): os.makedirs(reportPath)
        self.penelope.save(reportPath + 'feature_importances.txt')
        self.circe.save_guide(reportPath + 'grn_guide.js')
        self.factors.save(reportPath)
