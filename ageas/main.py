#!/usr/bin/env python3
"""
Ageas Reborn

author: jy, nkmtmsys
"""

import os
import warnings
from ageas.operator.gem_db_loader import Load
import ageas.operator.grn_guidance as grn_guidance
import ageas.operator.trainer as trainer
import ageas.operator.model_interpreter as interpreter
import ageas.operator.feature_extractor as extractor
import ageas.tool.json as json



class Find:
    """
    Get candidate key factors and pathways
    and write report files into given folder
    """
    def __init__(self,
                database_path,
                database_type = 'gem_folder',
                class1_path = None,
                class2_path = None,
                specie = 'mouse',
                facNameType = 'gn',
                model_config_path = None,
                stdevThread = 100,
                mww_thread = 0.05,
                log2fc_thread = 0.1,
                prediction_thread = 'auto',
                correlation_thread = 0.2,
                iteration = 1,
                patient = None,
                noChangeThread = 0.1,
                keepRatio = 1.0,
                keepThread = 0.9,
                topGRP = 100,
                warning = False):
        # Initialization
        self.patient = patient
        self.noChangeThread = noChangeThread
        self.noChangeNum = 0
        if not warning: warnings.filterwarnings('ignore')

        # Let kirke casts GRN construction guidance first
        self.circe = grn_guidance.Cast(gem_data = Load(database_path,
                                                        database_type,
                                                        class1_path,
                                                        class2_path,
                                                        specie,
                                                        facNameType,
                                                        mww_thread,
                                                        log2fc_thread,
                                                        stdevThread),
                                        prediction_thread = prediction_thread,
                                        correlation_thread = correlation_thread)

        # train classifiers
        self.ulysses = trainer.Train(database_path =  database_path,
                                    database_type = database_type,
                                    class1_path = class1_path,
                                    class2_path = class2_path,
                                    model_config_path = model_config_path,
                                    grn_guidance = self.circe.guide,
                                    stdevThread = stdevThread,
                                    correlation_thread = correlation_thread,
                                    iteration = iteration,
                                    keepRatio = keepRatio,
                                    keepThread = keepThread)
        self.penelope = interpreter.Find(self.ulysses.models)
        self.factors = extractor.Extract(self.penelope,
                                        top_GRP_amount = topGRP)
        self.factors.extract_common(self.circe.guide, type = 'reg_source')
        self.factors.extract_common(self.circe.guide, type = 'reg_target')

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
