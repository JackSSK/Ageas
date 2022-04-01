#!/usr/bin/env python3
"""
Ageas Reborn

author: jy, nkmtmsys
"""

import os
import re
import pickle
import time
import warnings
import ageas.operator as operator
import ageas.tool.json as json
import ageas.lib.grn_caster as grn
import ageas.lib.model_caster as model
import ageas.database_setup.binary_class as database
from pkg_resources import resource_filename



class Train:
    """
    Train out well performing classification models
    """
    def __init__(self,
                database_path = None,
                database_type = 'gem_folder',
                class1_path = None,
                class2_path = None,
                model_config_path = None,
                # GRN casting params
                grn_guidance = None,
                stdevThread = 100,
                stdevKpRatio = None,
                correlation_thread = 0.2,
                distrThred = None,
                # Model casting params
                iteration = 1,
                testSize = 0.3,
                randomState = None,
                keepRatio = 1.0,
                keepThread = 0.9,):
        # load standard config file if not further specified
        if model_config_path is None:
            model_config_path = resource_filename(__name__,
                                            '../data/config/sample_config.js')
        model_config = json.decode(model_config_path)

        # Initialization
        self.grns = None
        self.database = database.Setup(database_path,
                                        class1_path,
                                        class2_path,
                                        database_type)

        # if reading in GEMs, we need to construct pseudo-cGRNs first
        if re.search(r'gem' , self.database.type):
            self.grns = grn.Make(database = self.database,
                                std_value_thread = stdevThread,
                                std_ratio_thread = stdevKpRatio,
                                correlation_thread = correlation_thread,
                                grn_guidance = grn_guidance)
            self.mode = 'gene_exp'
        # if we are reading in GRNs directly, just process them
        elif re.search(r'grn' , self.database.type):
            self.mode = 'grn'
        else:
            raise operator.Error('Unrecogonized database type: ',
                                self.database.type)
        # Train out models and find the best ones
        self.models = model.Cast(database = self.database,
                                modelsConfig = model_config,
                                mode = self.mode,
                                grnData = self.grns,
                                iteration = iteration,
                                testSize = testSize,
                                randomState = randomState,
                                keepRatio = keepRatio,
                                keepThread = keepThread)

    # Save GRN data in given path
    def save_GRNs(self, path):
        grn.save_GRN(self.grns, path)

    # Save result models in given path
    def save_models(self, path):
        with open(path, 'wb') as file: pickle.dump(self.models, file)
