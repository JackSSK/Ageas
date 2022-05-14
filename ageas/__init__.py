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



class Ageas:
    """
    Get candidate key factors and pathways
    and write report files into given folder
    """
    def __init__(self,
                class1_path = None,
                class2_path = None,
                clf_keep_ratio = 0.5,
                clf_accuracy_thread = 0.9,
                correlation_thread = 0.2,
                database_path = None,
                database_type = 'gem_file',
                factor_name_type = 'gene_name',
                feature_dropout_ratio = 0.2,
                feature_select_iteration = None,
                guide_load_path = None,
                interaction_database = 'grtd',
                top_grp_amount = 100,
                key_gene_change_thread = 0.1,
                log2fc_thread = None,
                model_config_path = None,
                model_select_iteration = None,
                mww_p_val_thread = 0.05,
                outlier_thread = None,
                patient = None,
                pcgrn_load_path = None,
                prediction_thread = 'auto',
                specie = 'mouse',
                sliding_window_size = 20,
                sliding_window_stride = None,
                std_value_thread = None,
                std_ratio_thread = None,
                stabilize_iteration = None,
                train_size = 0.95,
                warning = False,
                z_score_extract_thread = 0,):
        super(Ageas, self).__init__()

        """ Initialization """
        start = time.time()
        if not warning: warnings.filterwarnings('ignore')
        self.far_out_grps = []
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
                                            interaction_database,
                                            sliding_window_size,
                                            sliding_window_stride)
        # load standard config file if not further specified
        if model_config_path is None:
            path = resource_filename(__name__, 'data/config/list_config.js')
            self.model_config = config_maker.List_Config_Reader(path)
        else:
            self.model_config = json.decode(model_config_path)
        print('Time to initialize : ', time.time() - start)

        """ Make or load pcGRNs and GRN construction guidance """
        start = time.time()
        if guide_load_path is not None and pcgrn_load_path is not None:
            pcGRNs = grn.Make(load_path = pcgrn_load_path)
            self.circe = grn_guidance.Cast(load_path = guide_load_path)
        else:
            self.circe, pcGRNs=self.get_pcGRNs(database_info=self.database_info,
                                        std_value_thread = std_value_thread,
                                        std_ratio_thread = std_ratio_thread,
                                        mww_p_val_thread = mww_p_val_thread,
                                        log2fc_thread = log2fc_thread,
                                        prediction_thread = prediction_thread,
                                        correlation_thread = correlation_thread,
                                        guide_load_path = guide_load_path,)
        pcGRNs.save('data/pcGRN_easy.js')
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
        self.factor = extractor.Extract(self.penelope,
                                        z_score_extract_thread,
                                        self.far_out_grps,
                                        top_grp_amount)
        print('Time to interpret 1st Gen classifiers : ', time.time() - start)



        """ Feature Selection """
        if self.feature_select_iteration is not None:
            print('Entering Feature Selection')
            for i in range(self.feature_select_iteration):
                start = time.time()
                prev_grps = self.factor.grps.index
                rm = self.__get_grp_remove_list(self.penelope.feature_score,
                                                feature_dropout_ratio,
                                                outlier_thread)
                pcGRNs.update_with_remove_list(rm)
                clfs.clear_data()
                clfs.grns = pcGRNs
                clfs.general_process(train_size = train_size,
                                    clf_keep_ratio = clf_keep_ratio,
                                    clf_accuracy_thread = clf_accuracy_thread)
                self.penelope = interpreter.Interpret(clfs)
                self.factor = extractor.Extract(self.penelope,
                                                z_score_extract_thread,
                                                self.far_out_grps,
                                                top_grp_amount)
                print('Time to do a feature selection : ', time.time() - start)
                if self.__early_stop(prev_grps, self.factor.grps.index):
                    self.stabilize_iteration = None
                    break



        """ Stabilizing Output """
        if self.stabilize_iteration is not None:
            print('Stabilizing Output')
            for i in range(self.stabilize_iteration):
                prev_grps = self.factor.grps.index
                clfs.general_process(train_size = train_size,
                                    clf_keep_ratio = clf_keep_ratio,
                                    clf_accuracy_thread = clf_accuracy_thread)
                self.penelope.add(interpreter.Interpret(clfs).feature_score)
                self.factor = extractor.Extract(self.penelope,
                                                z_score_extract_thread,
                                                self.far_out_grps,
                                                top_grp_amount)
                if self.__early_stop(prev_grps, self.factor.grps.index):
                    break

    # get pseudo-cGRNs from GEMs or GRNs
    def get_pcGRNs(self,
                    database_info = None,
                    std_value_thread = 100,
                    std_ratio_thread = None,
                    mww_p_val_thread = 0.05,
                    log2fc_thread = 0.1,
                    prediction_thread = 'auto',
                    correlation_thread = 0.2,
                    guide_load_path = None,):
        guide = None
        # if reading in GEMs, we need to construct pseudo-cGRNs first
        if re.search(r'gem' , database_info.type):
            gem_data = Load(database_info,
                            mww_p_val_thread,
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

    # take out some GRPs based on feature dropout ratio
    def __get_grp_remove_list(self, feature_score = None,
                                    feature_dropout_ratio = 0.2,
                                    outlier_thread = 3):
        total_grp = len(feature_score.index)
        gate_index = int(total_grp * (1 - feature_dropout_ratio))
        remove_list = list(feature_score.index[gate_index:])
        for ele in self.__get_outliers(feature_score, outlier_thread):
            self.far_out_grps.append(ele)
            remove_list.append(ele[0])
        return remove_list

    # get outlier based on IQR value and outlier thread
    def __get_outliers(self, df, outlier_thread):
        # not using outlier filter if thread set to none
        if outlier_thread is None: return []
        result = []
        q3_value = df.iloc[int(len(df.index) * 0.25)]['importance']
        q1_value = df.iloc[int(len(df.index) * 0.75)]['importance']
        # set far out thread according to interquartile_range (IQR)
        far_out_thread = 3 * (q3_value - q1_value)
        # remove outliers as well
        prev_score = outlier_thread * 3
        for i in range(len(df.index)):
            score = df.iloc[i]['importance']
            if score >= max(far_out_thread, (prev_score / 3), outlier_thread):
                result.append([df.index[i], score])
                prev_score = score
            else: break
        return result

    # Stop iteration if key genes are not really changing
    def __early_stop(self, prev_grps = None, cur_grps = None):
        # just keep going if patient not set
        if self.patient is None: return False
        common = len(list(set(prev_grps).intersection(set(cur_grps))))
        change1 = (len(prev_grps) - common) / len(prev_grps)
        change2 = (len(cur_grps) - common) / len(cur_grps)
        change = (change1 + change2) / 2
        print('Average Key GRPs Changing Portion:', change)
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
    def write_reports(self, folder = 'report/'):
        if folder[-1] != '/': folder += '/'
        # Make path if not exist
        if not os.path.exists(folder): os.makedirs(folder)
        # GRN guide related
        self.circe.save_guide(folder + 'grn_guide.js')
        grn_guidance.Analysis(self.circe.guide).save(folder + 'grn_based.csv')
        # GRP importances
        self.penelope.save(folder + 'all_grps_importances.txt')
        self.factor.grps.to_csv(folder + 'key_grps_importances.txt', sep = '\t')
        if len(self.factor.outlier_grps) > 0:
            json.encode(self.factor.outlier_grps, folder + 'outlier_grps.js')
        # Common Regulatory sources and targets related
        self.factor.extract_common(self.circe.guide, type = 'regulatory_source')
        self.factor.extract_common(self.circe.guide, type = 'regulatory_target')
        self.factor.report().to_csv(folder + 'ageas_based.csv', index = False)
        # json.encode(self.factor.common_reg_source, folder + 'common_source.js')
        # json.encode(self.factor.common_reg_target, folder + 'common_target.js')
