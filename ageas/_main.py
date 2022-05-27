#!/usr/bin/env python3
"""
Ageas Reborn

author: jy, nkmtmsys
"""

import re
import os
import copy
import time
import warnings
from pkg_resources import resource_filename
import ageas
import ageas.tool.json as json
import ageas.lib.config_maker as config_maker
import ageas.lib.pcgrn_caster as grn
import ageas.lib.meta_grn_caster as meta_grn
import ageas.database_setup.binary_class as binary_db



class Launch:
    """
    Get candidate key factors and pathways
    and write report files into given folder
    """
    def __init__(self,
                class1_path:str = None,
                class2_path:str = None,
                clf_keep_ratio:float = 0.5,
                clf_accuracy_thread:float = 0.8,
                correlation_thread:float = 0.0,
                database_path:str = None,
                database_type:str = 'gem_file',
                factor_name_type:str = 'gene_name',
                feature_dropout_ratio:float = 0.1,
                feature_select_iteration:int = 1,
                interaction_database:float = 'biogrid',
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
                patient:int = 3,
                pcgrn_load_path:str = None,
                pcgrn_save_path:str = None,
                prediction_thread:str = 'auto',
                report_folder_path:str = None,
                specie:str = 'mouse',
                sliding_window_size:int = 20,
                sliding_window_stride:int = None,
                std_value_thread:float = None,
                std_ratio_thread:float = None,
                stabilize_iteration:int = 10,
                max_train_size:float = 0.95,
                unit_num:int = 1,
                warning:bool = False,
                z_score_extract_thread:float = 0.0,):
        super(Launch, self).__init__()

        """ Initialization """
        print('Launching Ageas')
        start_first = time.time()
        if not warning: warnings.filterwarnings('ignore')

        self.unit_num = unit_num
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
        # Make or load pcGRNs and meta GRN
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
        print('Time to cast or load pcGRNs : ', time.time() - start)
        # Initialize a basic unit
        self.basic_unit = ageas.Unit(
            database_info = self.database_info,
            model_config = self.model_config,
            meta = self.meta,
            pseudo_grns = self.pseudo_grns,

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
        print('Time to initialize Launch Process: ', time.time() - start_first)

        # Work now
        new_unit = copy.deepcopy(self.basic_unit)
        new_unit.process()
        self.regulon = new_unit.regulon

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

    # get pseudo-cGRNs from GEMs or GRNs
    def get_pseudo_grns(self,
                    database_info = None,
                    std_value_thread = 100,
                    std_ratio_thread = None,
                    mww_p_val_thread = 0.05,
                    log2fc_thread = 0.1,
                    prediction_thread = 'auto',
                    correlation_thread = 0.2,
                    meta_load_path = None,):
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
            pcGRNs = grn.Make(
                database_info = database_info,
                std_value_thread = std_value_thread,
                std_ratio_thread = std_ratio_thread,
                correlation_thread = correlation_thread,
                gem_data = gem_data,
                meta_grn = meta.grn
            )
        # if we are reading in MEX, make GEM first and then mimic GEM mode
        elif re.search(r'mex' , database_info.type):
            pcGRNs = None
            print('trainer.py: mode MEX need to be revised here')
        # if we are reading in GRNs directly, just process them
        elif re.search(r'grn' , database_info.type):
            pcGRNs = None
            print('trainer.py: mode GRN need to be revised here')
        else:
            raise lib.Error('Unrecogonized database type: ', database_info.type)
        return meta, pcGRNs
