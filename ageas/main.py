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
                std_ratio_thread = None,
                mww_thread = 0.05,
                log2fc_thread = 0.1,
                prediction_thread = 'auto',
                correlation_thread = 0.2,
                # training related args
                iteration = 2,
                patient = None,
                noChangeThread = 0.1,
                clf_keep_ratio = None,
                clf_accuracy_thread = 0.9,
                topGRP = 100,
                warning = False):
        super(Find, self).__init__()
        """ Initialization """
        start = time.time()
        if not warning: warnings.filterwarnings('ignore')
        self.patient = patient
        self.noChangeThread = noChangeThread
        self.noChangeNum = 0
        # Set up database path info
        self.database_info = binary_db.Setup(database_path,
                                            database_type,
                                            class1_path,
                                            class2_path,
                                            specie,
                                            sliding_window_size,
                                            sliding_window_stride)
        # load standard config file if not further specified
        if model_config_path is None:
            model_config_path = resource_filename(__name__,
                                            'data/config/sample_config.js')
        self.model_config = json.decode(model_config_path)
        print('Time to initialize : ', time.time() - start)

        self.circe, pcGRNs = self.get_pcGRNs(facNameType = facNameType,
                                        std_value_thread = std_value_thread,
                                        std_ratio_thread = std_ratio_thread,
                                        mww_thread = mww_thread,
                                        log2fc_thread = log2fc_thread,
                                        prediction_thread = prediction_thread,
                                        correlation_thread = correlation_thread,
                                        guide_load_path = None,)
        # pcGRNs.save('data/grns_2.js')
        # self.circe.save_guide(path = 'data/guide_2.js')
        # pcGRNs = grn.Make(load_path = 'data/grns_1.js')
        # self.circe = grn_guidance.Cast(load_path = guide_load_path)

        """ train classifiers """
        start = time.time()
        self.ulysses = trainer.Train(pcGRNs = pcGRNs,
                                    database_info = self.database_info,
                                    model_config = self.model_config,)
        # self.ulysses.general_process(train_size = 0.7,
        #                             clf_keep_ratio = clf_keep_ratio,
        #                             clf_accuracy_thread = clf_accuracy_thread)
        self.ulysses.successive_halving_process(iteration = iteration,
                                    clf_accuracy_thread = clf_accuracy_thread,
                                    last_train_size = 0.9)
        print('Time to train out classifiers : ', time.time() - start)

        """ interpret classifiers """
        start = time.time()
        self.penelope = interpreter.Find(self.ulysses)
        print('Time to interpret classifiers : ', time.time() - start)

        """ final part: extract key factors """
        start = time.time()
        self.factors = extractor.Extract(self.penelope, top_GRP_amount = topGRP)
        self.factors.extract_common(self.circe.guide,type = 'regulatory_source')
        self.factors.extract_common(self.circe.guide,type = 'regulatory_target')
        print('Time to do everything else : ', time.time() - start)
        start = time.time()

    # get pseudo-cGRNs from GEMs or GRNs
    def get_pcGRNs(self,
                    facNameType = 'gn',
                    std_value_thread = 100,
                    std_ratio_thread = None,
                    mww_thread = 0.05,
                    log2fc_thread = 0.1,
                    prediction_thread = 'auto',
                    correlation_thread = 0.2,
                    guide_load_path = None,):
        start = time.time()
        guide = None
        # if reading in GEMs, we need to construct pseudo-cGRNs first
        if re.search(r'gem' , self.database_info.type):
            gem_data = Load(self.database_info,
                            facNameType,
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
            pcGRNs = grn.Make(database_info = self.database_info,
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
        print('Time to cast pcGRNs : ', time.time() - start)
        return guide, pcGRNs

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
