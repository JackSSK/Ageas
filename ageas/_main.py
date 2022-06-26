#!/usr/bin/env python3
"""
Ageas Reborn

author: jy, nkmtmsys
"""
import os
import sys
import copy
import time
import threading
import warnings
from pkg_resources import resource_filename
import ageas
import ageas.tool.json as json
import ageas.lib.meta_grn_caster as meta_grn
import ageas.lib.config_maker as config_maker
import ageas.lib.atlas_extractor as extractor



GRP_TYPES = ['Standard', 'Outer', 'Bridge', 'Mix']



class Launch:
    """
    Object containing basic pipeline to launch AGEAS.

    Results are stored in attributes and can be saved as files.
    """

    def __init__(self,
                 model_config_path:str = None,
                 mute_unit:bool = True,
                 protocol:str = 'solo',
                 unit_num:int = 2,
                 warning_filter:str = 'ignore',
                 correlation_thread:float = 0.2,
                 database_path:str = None,
                 database_type:str = 'gem_files',
                 factor_id_type:str = 'gene_symbol',
                 group1_path:str = None,
                 group2_path:str = None,
                 interaction_database:str = 'gtrd',
                 log2fc_thread:float = None,
                 meta_load_path:str = None,
                 mww_p_val_thread:float = 0.05,
                 prediction_thread = 'auto',
                 psgrn_load_path:str = None,
                 specie:str = 'mouse',
                 sliding_window_size:int = 10,
                 sliding_window_stride:int = None,
                 std_value_thread:float = 1.0,
                 std_ratio_thread:float = None,
                 clf_keep_ratio:float = 0.5,
                 clf_accuracy_thread:float = 0.8,
                 cpu_mode:bool = False,
                 feature_dropout_ratio:float = 0.1,
                 feature_select_iteration:int = 1,
                 impact_depth:int = 3,
                 top_grp_amount:int = 100,
                 grp_changing_thread:float = 0.05,
                 link_step_allowrance:int = 1,
                 model_select_iteration:int = 2,
                 outlier_thread:float = 3.0,
                 stabilize_patient:int = 3,
                 stabilize_iteration:int = 10,
                 max_train_size:float = 0.95,
                 z_score_extract_thread:float = 0.0,
                ):
        """
        Start a new pipeline to launch AGEAS.

        Parameters:
            model_config_path: <str> Default = None
                Path to load model config file which will be used to initialize
                classifiers

            mute_unit: <bool> Default = True
                Whether AGEAS unit print out log while running.
                It is not mandatory but encouraged to remain True
                especially when using 'multi' protocol

            protocol: <str> Default = 'solo'
                AGEAS unit launching protocol.

                Supporting:

                    'solo': All units will run separately.

                    'multi': All units will run parallelly by multithreading.

            unit_num: <int> Default = 2
                Number of AGEAS units to launch.

            warning_filter: <str> Default = 'ignore'
                How warnings should be filtered. For other options,
                please check 'The Warnings Filter' section in:
                https://docs.python.org/3/library/warnings.html#warning-filter

            Additional:
                All args in ageas.Get_Pseudo_Samples()

                All args in ageas.Unit() excluding database_info, meta,
                model_config, and pseudo_grns,
        """
        super(Launch, self).__init__()

        """ Initialization """
        print('Launching Ageas')
        warnings.filterwarnings(warning_filter)
        start = time.time()
        self.reports = list()
        self.protocol = protocol
        self.unit_num = unit_num
        self.silent = mute_unit
        self.impact_depth = impact_depth

        # Get model configs
        if model_config_path is None:
            path = resource_filename(__name__, 'data/config/list_config.js')
            self.model_config = config_maker.List_Config_Reader(path)
        else:
            self.model_config = json.decode(model_config_path)
        print('Time to Boot: ', time.time() - start)

        # integrate database info
        # and make meta GRN, pseudo samples if not loaded
        self.database_info,self.meta,self.pseudo_grns=ageas.Get_Pseudo_Samples(
            meta_load_path = meta_load_path,
            psgrn_load_path = psgrn_load_path,
            database_path = database_path,
            database_type = database_type,
            group1_path = group1_path,
            group2_path = group2_path,
            specie = specie,
            factor_id_type = factor_id_type,
            interaction_database = interaction_database,
            sliding_window_size = sliding_window_size,
            sliding_window_stride = sliding_window_stride,
            std_value_thread = std_value_thread,
            std_ratio_thread = std_ratio_thread,
            mww_p_val_thread = mww_p_val_thread,
            log2fc_thread = log2fc_thread,
            prediction_thread = prediction_thread,
            correlation_thread = correlation_thread,
        )

        # Meta GRN Analysis
        self.meta_report = meta_grn.Analysis(self.meta.grn)

        print('\nDeck Ready')

        start = time.time()
        # Initialize a basic unit
        self.basic_unit = ageas.Unit(
            meta = self.meta,
            pseudo_grns = self.pseudo_grns,
            model_config = self.model_config,
            database_info = self.database_info,

            cpu_mode = cpu_mode,
            correlation_thread = correlation_thread,
            top_grp_amount = top_grp_amount,
            z_score_extract_thread = z_score_extract_thread,

            max_train_size = max_train_size,
            clf_keep_ratio = clf_keep_ratio,
            clf_accuracy_thread = clf_accuracy_thread,
            model_select_iteration = model_select_iteration,

            outlier_thread = outlier_thread,
            feature_dropout_ratio = feature_dropout_ratio,
            feature_select_iteration = feature_select_iteration,

            stabilize_patient = stabilize_patient,
            grp_changing_thread = grp_changing_thread,
            stabilize_iteration = stabilize_iteration,

            impact_depth = impact_depth,
            link_step_allowrance = link_step_allowrance,
        )

        self.lockon = threading.Lock()
        print('Protocol:', self.protocol)
        print('Silent:', self.silent)

        # Do everything unit by unit
        if self.protocol == 'solo':
            self._proto_solo()

        # Multithreading protocol
        elif self.protocol == 'multi':
            self._proto_multi()

        self.atlas = self._combine_unit_reports()
        print('Operation Time: ', time.time() - start)
        print('\nComplete\n')


    def save_reports(self,
                     folder_path:str = None,
                     save_unit_reports:bool = False,
                    ):
        """
        Save meta processed GRN, pseudo-sample GRNs,
        meta-GRN based analysis report,
        AGEAS based analysis report, and key atlas extracted by AGEAS.

        Args:
            folder_path: <str> Default = None
                Path to create folder for saving AGEAS report files.

            save_unit_reports: <bool> Default = False
                Whether saving key GRPs extracted by each AGEAS Unit or not.
                If True, reports will be saved in folder_path under folders
                named 'no_{}'.format(unit_num) starting from 0.
        """
        # prepare folder path
        if folder_path[-1] != '/':
            folder_path += '/'
        if not os.path.exists(folder_path): os.makedirs(folder_path)

        self.meta_report.save(folder_path + 'meta_report.csv')
        self.pseudo_grns.save(folder_path + 'psGRNs.js')
        self.meta.grn.save_json(folder_path + 'metaGRN.js')

        if save_unit_reports:
            for index, atlas in enumerate(self.reports):
                report_path = folder_path + 'no_' + str(index) + '/'
                if not os.path.exists(report_path): os.makedirs(report_path)
                atlas.grps.save(report_path + 'grps_importances.txt')
                json.encode(atlas.outlier_grps, report_path + 'outlier_grps.js')

        # change class objects to dicts and save regulons in JSON format
        json.encode(
            {k:v.as_dict() for k,v in self.atlas.regulons.items()},
            folder_path + 'key_atlas.js'
        )

        self.atlas.report(self.meta.grn).to_csv(
            folder_path + 'report.csv',
            index = False
        )


    # Protocol SOLO
    def _proto_solo(self):
        for i in range(self.unit_num):
            id = 'RN_' + str(i)
            new_unit = copy.deepcopy(self.basic_unit)
            print('Unit', id, 'Ready')
            print('\nSending Unit', id, '\n')
            if self.silent: sys.stdout = open(os.devnull, 'w')
            new_unit.select_models()
            new_unit.launch()
            new_unit.generate_regulons()
            self.reports.append(new_unit.atlas)
            if self.silent: sys.stdout = sys.__stdout__
            print(id, 'RTB\n')

    # Protocol MULTI
    def _proto_multi(self):
        units = []
        for i in range(self.unit_num):
            id = 'RN_' + str(i)
            units.append(threading.Thread(target=self._multi_unit, name=id))
            print('Unit', id, 'Ready')

        # Time to work
        print('\nSending All Units\n')
        if self.silent: sys.stdout = open(os.devnull, 'w')

        # Start each unit
        for unit in units: unit.start()

        # Wait till all thread terminates
        for unit in units: unit.join()
        if self.silent: sys.stdout = sys.__stdout__
        print('Units RTB\n')

    # Model selection and regulon contruction part run parallel
    def _multi_unit(self,):
        new_unit = copy.deepcopy(self.basic_unit)
        new_unit.select_models()
        # lock here since SHAP would bring Error
        self.lockon.acquire()
        new_unit.launch()
        self.lockon.release()
        new_unit.generate_regulons()
        self.reports.append(new_unit.atlas)
        del new_unit

    # Combine information from reports returned by each unit
    def _combine_unit_reports(self):
        all_grps = dict()
        for index, atlas in enumerate(self.reports):
            for regulon in atlas.regulons.values():
                for id, record in regulon.grps.items():
                    if id not in all_grps:
                        all_grps[id] = record
                    elif id in all_grps:
                        all_grps[id] = self._combine_grp_records(
                            record_1 = all_grps[id],
                            record_2 = record
                        )

        # now we build regulons
        regulon = extractor.Extract()
        for id, grp in all_grps.items():
            regulon.update_regulon_with_grp(
                grp = grp,
                meta_grn = self.meta.grn
            )
        regulon.find_bridges(meta_grn = self.meta.grn)
        regulon.update_genes(impact_depth = self.impact_depth)
        regulon.change_regulon_list_to_dict()
        return regulon

    # combine information of same GRP form different reports
    def _combine_grp_records(self, record_1, record_2):
        answer = copy.deepcopy(record_1)
        if answer.type != record_2.type:

            if answer.type == GRP_TYPES[2]:
                assert answer.score == 0
                if record_2.type != GRP_TYPES[2]:
                    answer.type = record_2.type
                    answer.score = record_2.score

            else:
                if record_2.type != GRP_TYPES[2]:
                    answer.type = GRP_TYPES[3]
                    answer.score = max(answer.score, record_2.score)

        else:
            answer.score = max(answer.score, record_2.score)

        return answer
