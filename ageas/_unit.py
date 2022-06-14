#!/usr/bin/env python3
"""
Ageas Reborn

author: jy, nkmtmsys
"""

import time
import ageas.lib.clf_trainer as trainer
import ageas.lib.clf_interpreter as interpreter
import ageas.lib.atlas_extractor as extractor



class Unit:
    """
    Get candidate key factors and pathways
    and write report files into given folder
    """
    def __init__(self,
                 # Processed in Launch Initialization
                 database_info = None,
                 meta = None,
                 model_config = None,
                 pseudo_grns = None,
                 # Parameters
                 clf_keep_ratio:float = 0.5,
                 clf_accuracy_thread:float = 0.8,
                 correlation_thread:float = 0.0,
                 feature_dropout_ratio:float = 0.1,
                 feature_select_iteration:int = 1,
                 grp_changing_thread:float = 0.05,
                 impact_depth:int = 3,
                 link_step_allowrance:int = 0,
                 max_train_size:float = 0.95,
                 model_select_iteration:int = 2,
                 outlier_thread:float = 3.0,
                 patient:int = 3,
                 stabilize_iteration:int = 10,
                 top_grp_amount:int = 100,
                 z_score_extract_thread:float = 0.0,
                ):
        super(Unit, self).__init__()

        """ Initialization """
        self.far_out_grps = {}
        self.no_change_iteration_num = 0

        self.meta = meta
        self.pseudo_grns = pseudo_grns
        self.model_config = model_config
        self.database_info = database_info

        self.correlation_thread = correlation_thread
        self.top_grp_amount = top_grp_amount
        self.z_score_extract_thread = z_score_extract_thread

        self.max_train_size = max_train_size
        self.clf_keep_ratio = clf_keep_ratio
        self.clf_accuracy_thread = clf_accuracy_thread
        self.model_select_iteration = model_select_iteration

        self.outlier_thread = outlier_thread
        self.feature_dropout_ratio = feature_dropout_ratio
        self.feature_select_iteration = feature_select_iteration

        self.patient = patient
        self.grp_changing_thread = grp_changing_thread
        self.stabilize_iteration = stabilize_iteration

        self.impact_depth = impact_depth
        self.link_step_allowrance = link_step_allowrance

    # Select Classification models for later interpretations
    def select_models(self,):
        print('\nEntering Model Selection')
        start = time.time()
        # initialize trainer
        self.clf = trainer.Train(
            psGRNs = self.pseudo_grns,
            database_info = self.database_info,
            model_config = self.model_config,
        )
        # start model selection
        self.clf.successive_pruning(
            iteration = self.model_select_iteration,
            clf_keep_ratio = self.clf_keep_ratio,
            clf_accuracy_thread = self.clf_accuracy_thread,
            last_train_size = self.max_train_size
        )
        print('Finished Model Selection', time.time() - start)

    def launch(self,):
        start = time.time()
        self.grp_importances = interpreter.Interpret(self.clf)
        self.atlas = extractor.Extract(
            self.correlation_thread,
            self.grp_importances,
            self.z_score_extract_thread,
            self.far_out_grps,
            self.top_grp_amount
        )
        print('Time to interpret 1st Gen classifiers : ', time.time() - start)

        """ Feature Selection """
        if (self.feature_select_iteration is not None and
            self.feature_select_iteration > 0):
            print('\nEntering Feature Selection')
            for i in range(self.feature_select_iteration):
                start = time.time()
                prev_grps = self.atlas.top_grps.index
                rm = self.__get_grp_remove_list(
                            self.grp_importances.result,
                            self.feature_dropout_ratio,
                            self.outlier_thread
                        )
                self.pseudo_grns.update_with_remove_list(rm)
                self.clf.clear_data()
                self.clf.grns = self.pseudo_grns
                self.clf.general_process(
                    train_size = self.max_train_size,
                    clf_keep_ratio = self.clf_keep_ratio,
                    clf_accuracy_thread = self.clf_accuracy_thread
                )
                self.grp_importances = interpreter.Interpret(self.clf)
                self.atlas = extractor.Extract(
                    self.correlation_thread,
                    self.grp_importances,
                    self.z_score_extract_thread,
                    self.far_out_grps,
                    self.top_grp_amount
                )
                print('Time to do a feature selection : ', time.time() - start)
                if self.__early_stop(prev_grps, self.atlas.top_grps.index):
                    self.stabilize_iteration = None
                    break
        print('Total Length of Outlier GRPs is:', len(self.far_out_grps))

        """ Stabilizing Key GRPs """
        if (self.stabilize_iteration is not None and
            self.stabilize_iteration > 0):
            print('\nStabilizing Key GRPs')
            start = time.time()
            denominator = 1
            for i in range(self.stabilize_iteration):
                denominator += i
                prev_grps = self.atlas.top_grps.index
                self.clf.general_process(
                    train_size = self.max_train_size,
                    clf_keep_ratio = self.clf_keep_ratio,
                    clf_accuracy_thread = self.clf_accuracy_thread
                )
                self.grp_importances.add(interpreter.Interpret(self.clf).result)
                self.atlas = extractor.Extract(
                    self.correlation_thread,
                    self.grp_importances,
                    self.z_score_extract_thread,
                    self.far_out_grps,
                    self.top_grp_amount
                )
                if self.__early_stop(prev_grps, self.atlas.top_grps.index):
                    break
            self.grp_importances.divide(denominator)
            self.atlas = extractor.Extract(
                self.correlation_thread,
                self.grp_importances,
                self.z_score_extract_thread,
                self.far_out_grps,
                self.top_grp_amount
            )
            print('Time to stabilize key GRPs : ', time.time() - start)
        del self.grp_importances

    # Construct Regulons with Extracted GRPs and Access Them
    def generate_regulons(self,):
        print('\nBuilding Regulons with key GRPs')
        start = time.time()
        self.atlas.build_regulon(
            meta_grn = self.meta.grn,
            impact_depth = self.impact_depth
        )
        # Attempting to Connect Regulons if necessary
        if (self.link_step_allowrance is not None and
            self.link_step_allowrance > 0 and
            len(self.atlas.regulons) > 1):
            self.atlas.link_regulon(
                meta_grn = self.meta.grn,
                allowrance = self.link_step_allowrance
            )
        self.atlas.change_regulon_list_to_dict()
        print('Time to build key regulons : ', time.time() - start)

    # take out some GRPs based on feature dropout ratio
    def __get_grp_remove_list(self,
                              df = None,
                              feature_dropout_ratio = 0.2,
                              outlier_thread = 3
                             ):
        total_grp = len(df.index)
        gate_index = int(total_grp * (1 - feature_dropout_ratio))
        remove_list = list(df.index[gate_index:])
        for ele in self.__get_outliers(df, outlier_thread):
            self.far_out_grps[ele[0]] = ele[1]
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
        if change <= self.grp_changing_thread:
            self.no_change_iteration_num += 1
            if self.no_change_iteration_num == self.patient:
                print('Run out of patient! Early stopping!')
                return True
            else: return False
        else:
            self.no_change_iteration_num = 0
            return False
