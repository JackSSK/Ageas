#!/usr/bin/env python3
"""
Ageas Reborn

author: jy, nkmtmsys
"""
import re
import os
import sys
import copy
import time
import threading
import warnings
from pkg_resources import resource_filename
import ageas
import ageas.tool.json as json
import ageas.lib.psgrn_caster as psgrn
import ageas.lib.meta_grn_caster as meta_grn
import ageas.lib.config_maker as config_maker
import ageas.lib.atlas_extractor as extractor
import ageas.database_setup.binary_class as binary_db



GRP_TYPES = ['Standard', 'Outer', 'Bridge', 'Mix']



class Launch:
    """
    Main function to launch AGEAS

    Args:
        group1_path: <str> Default = None
            Path to file or folder being considered as sample group 1 data

        group2_path: <str> Default = None
            Path to file or folder being considered as sample group 2 data

        clf_keep_ratio: <float> Default = 0.5
            Portion of classifier model to keep after each model selection
            iteration.
            .. note::
                When performing SHA based model selection, this value is
                set as lower bound to keep models

        clf_accuracy_thread: <float> Default = 0.8
            Filter thread of classifier's accuracy in local test performed at
            each model selection iteration
            .. note::
                When performing SHA based model selection, this value is
                only used at last iteration

        correlation_thread: <float> Default = 0.2
            Gene expression correlation thread value of GRPs
            Potential GRPs failed to reach this value will be dropped

        cpu_mode: <bool> Default = False
            Whether force to use CPU only or not

        database_path: <str> Default = None
            Database header. If specified, group1_path and group2_path will be
            rooted here.

        database_type: <str> Default = 'gem_files'
            Type of data group1_path and group1_path are directing to
            Supporting:
                'gem_files': Each path is directing to a GEM file.
                    Pseudo samples will be generated with sliding window algo
                'gem_folders': Each path is directing to a GEM folder. Files in
                    each folder will be used to generate pseudo samples
                'mex_folders': Each path is directing to a folder consisting MEX
                    files(***matrix.mtx***, ***genes.tsv***, ***barcodes.tsv***)
                    Pseudo samples will be generated with sliding window tech

        factor_name_type: <str> Default = 'gene_name'
            What type of ID name to use for each gene.
            Supporting:
                'gene_name': Gene Symbols/Names
                'ens_id': Ensembl ID
                .. note::
                    If using BioGRID as interaction database, factor_name_type
                    must be set to 'gene_name' for now.
                    # TODO: Find a way to map gene names with Ensembl IDs

        feature_dropout_ratio: <float> Default = 0.1
            Portion of features(GRPs) to be dropped out after each iteration of
            feature selection.

        feature_select_iteration: <int> Default = 1
            Number of iteration for feature(GRP) selection before
            key GRP extraction

        interaction_database: <str> Default = 'gtrd'
            Which interaction database to use for confirming a GRP has a high
            possibility to exist.
            Supporting:
                None: No database will be used. As long as a GRP can pass all
                    related filters, it's good to go.
                'gtrd': Using GTRD as regulatory pathway reference
                    https://gtrd.biouml.org/
                'biogrid': Using BioGRID as regulatory pathway reference
                    https://thebiogrid.org/

        impact_depth: <int> Default = 3
            When assessing a TF's regulatory impact on other genes, how far the
            distance between TF and potential regulatory source can be.
            .. note::
                The correlation strength of stepped correlation strength of TF
                and gene still need to be greater than correlation_thread.

        top_grp_amount: <int> Default = 100
            Amount of GRPs an AGEAS unit would extract.
            .. note::
                If outlier_thread is set, since outlier GRPs are extracted
                during feature selection part and will also be considered as
                key GRPs, actual amount of key GRPs would be greater.

        grp_changing_thread: <float> Default = 0.05
            If changing portion of key GRPs extracted by AGEAS unit from two
            stabilize iterations lower than this thread, these two iterations
            will be considered as having consistent result.

        log2fc_thread: <float> Default = None
            Log2 fold change thread to filer non-differntial expressing genes.
            .. note::
                It's generally not encouraged to set up this filter since it can
                result in lossing key TFs not having great changes on overall
                expression volume but having changes on expression pattern.
                If local computational power is relatively limited, setting up
                this thread can help a lot to keep program runable.

        link_step_allowrance: <int> Default = 1
            During key atlas extraction, when finding bridge GRPs to link 2
            separate regulons, how many steps will be allowed.
            link_step_allowrance == 1 means, no intermediate gene can be used
            and portential regulatory source must be able to interact with gene
            from another regulon.

        meta_load_path: <str> Default = None
            Path to load meta_GRN

        meta_save_path: <str> Default = None
            Path to save meta_GRN

        model_config_path: <str> Default = None
            Path to load model config file which will be used to initialize
            classifiers

        model_select_iteration: <int> Default = 2
            Number of iteration for classification model selection before
            the mandatory filter.

        mute_unit: <bool> Default = True
            Whether AGEAS unit print out log while running.
            .. note::
                It's not mandatory but encouraged to remain True especially
                when using multi protocol

        mww_p_val_thread: <str> Default = 0.05
            Gene expression Mann–Whitney–Wilcoxon test p-value thread.
            To make sure one gene's expression profile is not constant among
            differnt classes.

        outlier_thread: <float> Default = 3.0
            The lower bound of Z-score scaled importance value to consider a GRP
            as outlier need to be retain.

        protocol: <str> Default = 'solo'
            AGEAS unit launching protocol.
            Supporting:
                'solo': All units will run separately
                'multi': All units will run parallelly by multithreading

        patient: <int> Default = 3
            If stabilize iterations continuously having consistent result for
            this value, an early stop on result stabilization will be executed.

        psgrn_load_path: <str> Default = None
            Path to load pseudo-sample GRNs.

        psgrn_save_path: <str> Default = None
            Path to save pseudo-sample GRNs.

        prediction_thread: <str> or <float> Default = 'auto'
            The importance thread for a GRP predicted with GRNBoost2-like algo
            to be included.
            Supporting:
                'auto': Automatically set up thread value by minimum imporatnace
                    value of a interaction database recorded GRP of TF having
                    most amount of GRPs. If not using interaction database, it
                    will be set by (1 / amount of genes)
                float type: Value will be set as thread directly

        report_folder_path: <str> Default = None
            Path to create folder for saving AGEAS report files.

        save_unit_reports: <bool> Default = False
            Whether saving key GRPs extracted by each AGEAS Unit or not.
            If True, reports will be saved in report_folder_path under folders
            named 'no_{}'.format(unit_num) starting from 0.

        specie: <str> Default = 'mouse'
            Specify which sepcie's interaction database shall be used.
            Supporting:
                'mouse'
                'human'

        sliding_window_size: <int> Default = 10
            Number of samples a pseudo-sample generated with
            sliding window technique contains.

        sliding_window_stride: <int> Default = None
            Stride of sliding window when generating pseudo-samples.

        std_value_thread: <float> Default = 1.0
            Set up gene expression standard deviation thread by value.
            To rule out genes having relatively constant expression in each type
            class.

        std_ratio_thread: <float> Default = None
            Set up gene expression standard deviation thread by portion.
            Only genes reaching top portion will be kept in each sample group.

        stabilize_iteration: <int> Default = 10
            Number of iteration for a AGEAS unit to repeat key GRP extraction
            after model and feature selections in order to find key GRPs
            consistently being important.

        max_train_size: <float> Default = 0.95
            The largest portion of avaliable data can be used to train models.
            At the mandatory model filter, this portion of data will be given to
            each model to train.

        unit_num: <int> Default = 2
            Number of AGEAS units to launch.

        warning_filter: <str> Default = 'ignore'
            How warnings should be filtered.
            For other options, please check 'The Warnings Filter' section in:
                https://docs.python.org/3/library/warnings.html#warning-filter

        z_score_extract_thread: <float> Default = 0.0
            The lower bound of Z-score scaled importance value to extract a GRP.

    Inputs: None
    Outputs: None
    Attributes:

    Examples::
        >>> easy = ageas.Launch(
            	group1_path = 'test/ips.csv',
            	group2_path = 'test/mef.csv',
            )
    """
    def __init__(self,
                 group1_path:str = None,
                 group2_path:str = None,
                 clf_keep_ratio:float = 0.5,
                 clf_accuracy_thread:float = 0.8,
                 correlation_thread:float = 0.2,
                 cpu_mode:bool = False,
                 database_path:str = None,
                 database_type:str = 'gem_files',
                 factor_name_type:str = 'gene_name',
                 feature_dropout_ratio:float = 0.1,
                 feature_select_iteration:int = 1,
                 interaction_database:str = 'gtrd',
                 impact_depth:int = 3,
                 top_grp_amount:int = 100,
                 grp_changing_thread:float = 0.05,
                 log2fc_thread:float = None,
                 link_step_allowrance:int = 1,
                 meta_load_path:str = None,
                 meta_save_path:str = None,
                 model_config_path:str= None,
                 model_select_iteration:int = 2,
                 mww_p_val_thread:str = 0.05,
                 outlier_thread:float = 3.0,
                 protocol:str = 'solo',
                 patient:int = 3,
                 psgrn_load_path:str = None,
                 psgrn_save_path:str = None,
                 prediction_thread = 'auto',
                 report_folder_path:str = None,
                 save_unit_reports:bool = False,
                 specie:str = 'mouse',
                 sliding_window_size:int = 10,
                 sliding_window_stride:int = None,
                 std_value_thread:float = 1.0,
                 std_ratio_thread:float = None,
                 stabilize_iteration:int = 10,
                 max_train_size:float = 0.95,
                 unit_num:int = 2,
                 unit_silent:bool = True,
                 warning_filter:str = 'ignore',
                 z_score_extract_thread:float = 0.0,
                ):
        super(Launch, self).__init__()

        """ Initialization """
        print('Launching Ageas')
        warnings.filterwarnings(warning_filter)
        start = time.time()
        self.reports = list()
        self.protocol = protocol
        self.unit_num = unit_num
        self.silent = unit_silent
        self.impact_depth = impact_depth
        
        # Get database information
        self.database_info = binary_db.Setup(
            database_path,
            database_type,
            group1_path,
            group2_path,
            specie,
            factor_name_type,
            interaction_database,
            sliding_window_size,
            sliding_window_stride
        )

        # Get model configs
        if model_config_path is None:
            path = resource_filename(__name__, 'data/config/list_config.js')
            self.model_config = config_maker.List_Config_Reader(path)
        else:
            self.model_config = json.decode(model_config_path)

        # Prepare report folder
        self.report_folder_path = report_folder_path
        if self.report_folder_path is not None:
            if self.report_folder_path[-1] != '/':
                self.report_folder_path += '/'
            if not os.path.exists(self.report_folder_path):
                os.makedirs(self.report_folder_path)
        self.save_unit_reports = save_unit_reports
        if self.save_unit_reports and self.report_folder_path is None:
            raise Exception('Report Path must be given to save unit reports!')
        print('Time to Boot: ', time.time() - start)

        # Make or load psGRNs and meta GRN
        start = time.time()
        if meta_load_path is not None and psgrn_load_path is not None:
            self.meta = meta_grn.Cast(load_path = meta_load_path)
            self.pseudo_grns = psgrn.Make(load_path = psgrn_load_path)
        else:
            self.meta, self.pseudo_grns = self.get_pseudo_grns(
                database_info = self.database_info,
                std_value_thread = std_value_thread,
                std_ratio_thread = std_ratio_thread,
                mww_p_val_thread = mww_p_val_thread,
                log2fc_thread = log2fc_thread,
                prediction_thread = prediction_thread,
                correlation_thread = correlation_thread,
                meta_load_path = meta_load_path,
            )

        # Meta GRN Analysis
        self.meta_report = meta_grn.Analysis(self.meta.grn)

        # Save docs if specified path
        if self.report_folder_path is not None:
            self.meta_report.save(self.report_folder_path + 'meta_report.csv')
        if psgrn_save_path is not None:
            self.pseudo_grns.save(psgrn_save_path)
        if meta_save_path is not None:
            self.meta.grn.save_json(meta_save_path)
        print('Time to cast or load Pseudo-Sample GRNs : ', time.time() - start)
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

            patient = patient,
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
            self.proto_solo()

        # Multithreading protocol
        elif self.protocol == 'multi':
            self.proto_multi()

        self.atlas = self.combine_unit_reports()
        print('Operation Time: ', time.time() - start)

        if self.report_folder_path is not None:
            print('Generating Report Files')
            self._save_atlas_as_json(
                self.atlas.regulons,
                self.report_folder_path + 'key_atlas.js'
            )

            self.atlas.report(self.meta.grn).to_csv(
                self.report_folder_path + 'report.csv',
                index = False
            )

        print('\nComplete\n')

    # Protocol SOLO
    def proto_solo(self):
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
    def proto_multi(self):
        units = []
        for i in range(self.unit_num):
            id = 'RN_' + str(i)
            units.append(threading.Thread(target=self.multi_unit, name=id))
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
    def multi_unit(self,):
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
    def combine_unit_reports(self):
        all_grps = dict()
        for index, atlas in enumerate(self.reports):

            # save unit report if asking
            if self.save_unit_reports:
                report_path = self.report_folder_path + 'no_' + str(index) + '/'
                if not os.path.exists(report_path): os.makedirs(report_path)
                atlas.grps.save(report_path + 'grps_importances.txt')
                json.encode(atlas.outlier_grps, report_path+'outlier_grps.js')

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

    # get pseudo-cGRNs from GEMs or GRNs
    def get_pseudo_grns(self,
                        database_info = None,
                        std_value_thread = 100,
                        std_ratio_thread = None,
                        mww_p_val_thread = 0.05,
                        log2fc_thread = 0.1,
                        prediction_thread = 'auto',
                        correlation_thread = 0.2,
                        meta_load_path = None
                       ):
        meta = None
        # if reading in GEMs, we need to construct pseudo-cGRNs first
        # or if we are reading in MEX, make GEM first and then mimic GEM mode
        if (re.search(r'gem' , database_info.type) or
            re.search(r'mex' , database_info.type)):
            gem_data = binary_db.Load_GEM(
                database_info,
                mww_p_val_thread,
                log2fc_thread,
                std_value_thread
            )
            start1 = time.time()

            # Let kirke casts GRN construction guidance first
            meta = meta_grn.Cast(
                gem_data = gem_data,
                prediction_thread = prediction_thread,
                correlation_thread = correlation_thread,
                load_path = meta_load_path
            )
            print('Time to cast Meta GRN : ', time.time() - start1)
            psGRNs = psgrn.Make(
                database_info = database_info,
                std_value_thread = std_value_thread,
                std_ratio_thread = std_ratio_thread,
                correlation_thread = correlation_thread,
                gem_data = gem_data,
                meta_grn = meta.grn
            )

        # if we are reading in GRNs directly, just process them
        elif re.search(r'grn' , database_info.type):
            psGRNs = None
            print('trainer.py: mode GRN need to be revised here')
        else:
            raise lib.Error('Unrecogonized database type: ', database_info.type)
        return meta, psGRNs

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

    # change class objects to dicts and save regulons in JSON format
    def _save_atlas_as_json(self, regulons, path):
        json.encode({k:v.as_dict() for k,v in regulons.items()}, path)
