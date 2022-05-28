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
import ageas.lib.config_maker as config_maker
import ageas.lib.psgrn_caster as grn
import ageas.lib.meta_grn_caster as meta_grn
import ageas.database_setup.binary_class as binary_db



class Launch:
    """
    Get candidate key factors and pathways
    and write report files into given folder
    """
    def __init__(self,
                block_blog:bool = True,
                class1_path:str = None,
                class2_path:str = None,
                clf_keep_ratio:float = 0.5,
                clf_accuracy_thread:float = 0.8,
                correlation_thread:float = 0.2,
                database_path:str = None,
                database_type:str = 'gem_file',
                factor_name_type:str = 'gene_name',
                feature_dropout_ratio:float = 0.1,
                feature_select_iteration:int = 1,
                interaction_database:str = 'gtrd',
                impact_depth:int = 3,
                top_grp_amount:int = 100,
                grp_changing_thread:float = 0.05,
                log2fc_thread:float = None,
                link_step_allowrance:int = 0,
                meta_load_path:str = None,
                meta_save_path:str = None,
                model_config_path :str= None,
                model_select_iteration:int = 2,
                mww_p_val_thread:str = 0.05,
                outlier_thread:float = 3.0,
                protocol:str = 'solo',
                patient:int = 3,
                pcgrn_load_path:str = None,
                pcgrn_save_path:str = None,
                prediction_thread:str = 'auto',
                report_folder_path:str = None,
                specie:str = 'mouse',
                sliding_window_size:int = 10,
                sliding_window_stride:int = None,
                std_value_thread:float = None,
                std_ratio_thread:float = None,
                stabilize_iteration:int = 10,
                max_train_size:float = 0.95,
                unit_num:int = 2,
                warning:bool = False,
                z_score_extract_thread:float = 0.0,):
        super(Launch, self).__init__()

        """ Initialization """
        print('Launching Ageas')
        start = time.time()
        if not warning: warnings.filterwarnings('ignore')
        self.results = []
        self.protocol = protocol
        self.unit_num = unit_num
        self.silent = block_blog
        # Get database information
        self.database_info = binary_db.Setup(
            database_path,
            database_type,
            class1_path,
            class2_path,
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
        print('Time to Boot: ', time.time() - start)

        # Make or load psGRNs and meta GRN
        start = time.time()
        if meta_load_path is not None and pcgrn_load_path is not None:
            self.meta = meta_grn.Cast(load_path = meta_load_path)
            self.pseudo_grns = grn.Make(load_path = pcgrn_load_path)
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
        if pcgrn_save_path is not None:
            self.pseudo_grns.save(pcgrn_save_path)
        if meta_save_path is not None:
            self.meta.save_guide(meta_save_path)
        print('Time to cast or load Pseudo-Sample GRNs : ', time.time() - start)
        print('\nDeck Ready')

        start = time.time()
        # Initialize a basic unit
        self.basic_unit = ageas.Unit(
            meta = self.meta,
            pseudo_grns = self.pseudo_grns,
            model_config = self.model_config,
            database_info = self.database_info,

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

        # Prepare Units
        print('Protocol:', self.protocol)
        self.lockon = threading.Lock()
        units = []
        for i in range(self.unit_num):
            id = 'RN_' + str(i)
            print('Preparing Unit', id)
            if self.protocol == 'multi':
                units.append(threading.Thread(target=self.proto_multi, name=id))
            elif self.protocol == 'solo':
                units.append(threading.Thread(target=self.proto_solo, name=id))

        # Time to work
        print('\nSending All Units')
        print('Silent:', self.silent)
        if self.silent: sys.stdout = open(os.devnull, 'w')
        # Start each unit
        for unit in units:
            unit.start()
        # Wait till all thread terminates
        for unit in units:
            unit.join()
        if self.silent: sys.stdout = sys.__stdout__
        print('\nUnits RTB')
        print('Operation Time: ', time.time() - start)
        del units

        self.regulon = self.results[0]
        if self.report_folder_path is not None:
            self.regulon.full_grps.save(
                self.report_folder_path + 'grps_importances.txt'
            )
            json.encode(
                self.regulon.regulons,
                self.report_folder_path + 'regulons.js'
            )
            self.regulon.report(self.meta.grn).to_csv(
                self.report_folder_path + 'report.csv',
                index = False
            )
        print('\nFin\n')

    # Model selection and regulon contruction part run parallel
    def proto_multi(self,):
        new_unit = copy.deepcopy(self.basic_unit)
        new_unit.select_models()
        # lock here since SHAP would bring Error
        self.lockon.acquire()
        new_unit.launch()
        self.lockon.release()
        new_unit.generate_regulons()
        self.results.append(new_unit.regulon)

    # Do everything unit by unit
    def proto_solo(self,):
        self.lockon.acquire()
        new_unit = copy.deepcopy(self.basic_unit)
        new_unit.select_models()
        new_unit.launch()
        new_unit.generate_regulons()
        self.results.append(new_unit.regulon)
        self.lockon.release()

    # get pseudo-cGRNs from GEMs or GRNs
    def get_pseudo_grns(self,
                        database_info = None,
                        std_value_thread = 100,
                        std_ratio_thread = None,
                        mww_p_val_thread = 0.05,
                        log2fc_thread = 0.1,
                        prediction_thread = 'auto',
                        correlation_thread = 0.2,
                        meta_load_path = None):
        meta = None
        # if reading in GEMs, we need to construct pseudo-cGRNs first
        if re.search(r'gem' , database_info.type):
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
            psGRNs = grn.Make(
                database_info = database_info,
                std_value_thread = std_value_thread,
                std_ratio_thread = std_ratio_thread,
                correlation_thread = correlation_thread,
                gem_data = gem_data,
                meta_grn = meta.grn
            )
        # if we are reading in MEX, make GEM first and then mimic GEM mode
        elif re.search(r'mex' , database_info.type):
            psGRNs = None
            print('trainer.py: mode MEX need to be revised here')
        # if we are reading in GRNs directly, just process them
        elif re.search(r'grn' , database_info.type):
            psGRNs = None
            print('trainer.py: mode GRN need to be revised here')
        else:
            raise lib.Error('Unrecogonized database type: ', database_info.type)
        return meta, psGRNs
