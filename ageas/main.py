#!/usr/bin/env python3
"""
Ageas Reborn

author: jy, nkmtmsys
"""

import os
import time
import warnings
import ageas.tool.json as json
import ageas.lib.pcgrn_caster as grn
import ageas.operator.trainer as trainer
from ageas.operator.gem_db_loader import Load
import ageas.operator.grn_guidance as grn_guidance
import ageas.operator.model_interpreter as interpreter
import ageas.operator.feature_extractor as extractor
import ageas.database_setup.binary_class as binary_db



class Find:
    """
    Get candidate key factors and pathways
    and write report files into given folder
    """
    def __init__(self,
                # GEM data location related args
                database_path = None,
                database_type = 'gem_file',
                class1_path = None,
                class2_path = None,
                specie = 'mouse',
                # sliding window related args (for gem_file mode)
                sliding_window_size = 10,
                sliding_window_stride = None,
                # supportive data location related args
                facNameType = 'gn',
                model_config_path = None,
                # filter thread related args
                std_value_thread = 100,
                mww_thread = 0.05,
                log2fc_thread = 0.1,
                prediction_thread = 'auto',
                correlation_thread = 0.2,
                # training related args
                iteration = 1,
                patient = None,
                noChangeThread = 0.1,
                clf_keep_ratio = 1.0,
                clf_accuracy_thread = 0.9,
                topGRP = 100,
                warning = False):
        super(Find, self).__init__()
        # Initialization
        start = time.time()
        self.patient = patient
        self.noChangeThread = noChangeThread
        self.noChangeNum = 0
        if not warning: warnings.filterwarnings('ignore')
        # Set up database path info
        self.database_info = binary_db.Setup(database_path,
                                            database_type,
                                            class1_path,
                                            class2_path,
                                            specie,
                                            sliding_window_size,
                                            sliding_window_stride)
        gem_data = Load(self.database_info,
                        facNameType,
                        mww_thread,
                        log2fc_thread,
                        std_value_thread)
        # Let kirke casts GRN construction guidance first
        self.circe = grn_guidance.Guide(load_path = 'data/guide_2.js')
        # self.circe = grn_guidance.Guide(gem_data = gem_data,
        #                                 prediction_thread = prediction_thread,
        #                                 correlation_thread = correlation_thread)
        # self.circe.save_guide(path = 'data/guide_2.js')
        print('Time to cast GRN Guidnace : ', time.time() - start)
        start = time.time()
        # train classifiers
        # loaded_grns = grn.Make(load_path = 'data/grns_1.js')
        loaded_grns = None
        self.ulysses = trainer.Train(database_info = self.database_info,
                                    model_config_path = model_config_path,
                                    gem_data = gem_data,
                                    grn_guidance = self.circe.guide,
                                    std_value_thread = std_value_thread,
                                    correlation_thread = correlation_thread,
                                    iteration = iteration,
                                    clf_keep_ratio = clf_keep_ratio,
                                    clf_accuracy_thread = clf_accuracy_thread,
                                    grns = loaded_grns)
        self.ulysses.grns.save('data/grns_2.js')
        print('Time to train out classifiers : ', time.time() - start)
        # interpret classifiers
        start = time.time()
        self.penelope = interpreter.Find(self.ulysses.models)
        print('Time to interpret classifiers : ', time.time() - start)
        # final part: extract key factors
        start = time.time()
        self.factors = extractor.Extract(self.penelope,
                                        top_GRP_amount = topGRP)
        self.factors.extract_common(self.circe.guide,
                                    type = 'regulatory_source')
        self.factors.extract_common(self.circe.guide,
                                    type = 'regulatory_target')
        print('Time to do everything else : ', time.time() - start)
        start = time.time()

    # Stop iteration if abundace factors are not really changing
    def _earlyStopping(self, prevFactors, curFactors):
        prevFactors = [x[0] for x in prevFactors]
        curFactors = [x[0] for x in curFactors]
        common = list(set(prevFactors).intersection(curFactors))
        change = (len(prevFactors) - len(common)) / len(prevFactors)
        if change <= self.noChangeThread:
            self.noChangeNum += 1
            if self.noChangeNum == self.patient:
                print('Run out of patient! Early stopping!')
                return True
            else: return False
        else:
            self.noChangeNum = 0
            return False

    # Write report files in give folder
    def writeReport(self, reportPath = 'report/'):
        if reportPath[-1] != '/': reportPath += '/'
        # Make path if not exist
        if not os.path.exists(reportPath): os.makedirs(reportPath)
        self.penelope.save(reportPath + 'feature_importances.txt')
        self.circe.save_guide(reportPath + 'grn_guide.js')
        self.factors.save(reportPath)
