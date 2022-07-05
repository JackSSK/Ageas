#!/usr/bin/env python3
"""
Ageas Reborn

author: jy, nkmtmsys
"""

import time
import ageas
import ageas.lib.clf_trainer as trainer
import ageas.lib.clf_interpreter as interpreter
import ageas.lib.atlas_extractor as extractor



class Unit(object):
    """
    Extractor Unit to get candidate key regulatory pathways
    and corresponding genes.

    Results are stored in attributes and can be saved as files.

    """
    def __init__(self,
                 # Need to be processed before initialize Unit
                 database_info = None,
                 meta = None,
                 model_config = None,
                 pseudo_grns = None,
                 # Parameters
                 clf_keep_ratio:float = 0.5,
                 clf_accuracy_thread:float = 0.8,
                 correlation_thread:float = 0.2,
                 cpu_mode:bool = False,
                 feature_dropout_ratio:float = 0.1,
                 feature_select_iteration:int = 1,
                 grp_changing_thread:float = 0.05,
                 max_train_size:float = 0.95,
                 model_select_iteration:int = 2,
                 outlier_thread:float = 3.0,
                 regulatory_trace_depth:int = 1,
                 stabilize_patient:int = 3,
                 stabilize_iteration:int = 10,
                 top_grp_amount:int = 100,
                 z_score_extract_thread:float = 0.0,
                ):
        """
        Start a new AGEAS Extractor Unit.

        Parameters:
            database_info: <object> Default = None
                Integrated database information returned by
                ageas.Get_Pseudo_Samples()

            meta: <object> Default = None
                Meta level processed GRN information returned by
                ageas.Get_Pseudo_Samples()

            model_config: <dict> Default = None
                Dictionary containing configs of all candidate classification
                models.

            pseudo_grns: <object> Default = None
                pseudo-sample GRNs returned by
                ageas.Get_Pseudo_Samples()

            clf_keep_ratio: <float> Default = 0.5
                Portion of classifier model to keep after each model selection
                iteration.

                When performing SHA based model selection, this value is
                set as lower bound to keep models

            clf_accuracy_thread: <float> Default = 0.8
                Filter thread of classifier's accuracy in local test performed
                at each model selection iteration.

                When performing SHA based model selection, this value is
                only used at last iteration

            correlation_thread: <float> Default = 0.2
                Gene expression correlation thread value of GRPs.

                Potential GRPs failed to reach this value will be dropped.

            cpu_mode: <bool> Default = False
                Whether force to use CPU only or not.
                By default, AGEAS will automatically select device favoring
                CUDA based GPUs.

            feature_dropout_ratio: <float> Default = 0.1
                Portion of features(GRPs) to be dropped out after each
                iteration of feature selection.

            feature_select_iteration: <int> Default = 1
                Number of iteration for feature(GRP) selection before
                key GRP extraction

            top_grp_amount: <int> Default = 100
                Amount of GRPs an AGEAS extractor unit would extract.

                If outlier_thread is set, since outlier GRPs are extracted
                during feature selection part and will also be considered as
                key GRPs, actual amount of key GRPs would be greater.

            grp_changing_thread: <float> Default = 0.05
                If changing portion of key GRPs extracted by AGEAS unit from two
                stabilize iterations lower than this thread, these two
                iterations will be considered as having consistent result.

            model_select_iteration: <int> Default = 2
                Number of iteration for classification model selection before
                the mandatory filter.

            outlier_thread: <float> Default = 3.0
                The lower bound of Z-score scaled importance value to consider
                a GRP as outlier need to be retain.

            regulatory_trace_depth: <int> Default = 1
                Trace regulatory upstream of regulatory sources included in key
                regulons extracted.

            stabilize_patient: <int> Default = 3
                If stabilize iterations continuously having consistent
                result for this value, an early stop on result stabilization
                will be executed.

            stabilize_iteration: <int> Default = 10
                Number of iteration for a AGEAS unit to repeat key GRP
                extraction after model and feature selections in order to find
                key GRPs consistently being important.

            max_train_size: <float> Default = 0.95
                The largest portion of avaliable data can be used to
                train models.

                At the mandatory model filter,
                this portion of data will be given to each model to train.

            z_score_extract_thread: <float> Default = 0.0
                The lower bound of Z-score scaled importance value to extract
                a GRP.
        """
        super(Unit, self).__init__()

        """ Initialization """
        self.far_out_grps = {}
        self.no_change_iteration_num = 0

        self.meta = meta
        self.pseudo_grns = pseudo_grns
        self.model_config = model_config
        self.database_info = database_info

        self.cpu_mode = cpu_mode
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

        self.stabilize_patient = stabilize_patient
        self.grp_changing_thread = grp_changing_thread
        self.stabilize_iteration = stabilize_iteration
        self.regulatory_trace_depth = regulatory_trace_depth

    # Select Classification models for later interpretations
    def select_models(self,):
        print('\nEntering Model Selection')
        start = time.time()

        # initialize trainer
        self.clf = trainer.Train(
            psGRNs = self.pseudo_grns,
            cpu_mode = self.cpu_mode,
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
        self.atlas = extractor.Atlas(
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
                self.atlas = extractor.Atlas(
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
                self.atlas = extractor.Atlas(
                    self.correlation_thread,
                    self.grp_importances,
                    self.z_score_extract_thread,
                    self.far_out_grps,
                    self.top_grp_amount
                )

                if self.__early_stop(prev_grps, self.atlas.top_grps.index):
                    break

            self.grp_importances.divide(denominator)
            self.atlas = extractor.Atlas(
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
        self.atlas.build_regulon(meta_grn = self.meta.grn,)

        # Trace regulatory sources of regulons in atlas extracted
        for i in range(self.regulatory_trace_depth):
            self.atlas.add_reg_sources(meta_grn = self.meta.grn,)

        self.atlas.find_bridges(meta_grn = self.meta.grn)
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
        # just keep going if stabilize_patient not set
        if self.stabilize_patient is None: return False
        common = len(list(set(prev_grps).intersection(set(cur_grps))))
        change1 = (len(prev_grps) - common) / len(prev_grps)
        change2 = (len(cur_grps) - common) / len(cur_grps)
        change = (change1 + change2) / 2

        print('Average Key GRPs Changing Portion:', change)
        if change <= self.grp_changing_thread:
            self.no_change_iteration_num += 1
            if self.no_change_iteration_num == self.stabilize_patient:
                print('Run out of stabilize_patient! Early stopping!')
                return True
            else: return False
        else:
            self.no_change_iteration_num = 0
            return False
