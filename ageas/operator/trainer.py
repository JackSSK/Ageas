#!/usr/bin/env python3
"""
Ageas Reborn
This file contains classes to find best models for further analysis

author: jy, nkmtmsys
"""

import re
import math
import torch
import pickle
import difflib
import numpy as np
import pandas as pd
from warnings import warn
import ageas.operator as operator
import ageas.tool.json as json
import ageas.lib.pcgrn_caster as grn
from pkg_resources import resource_filename
import ageas.classifier.svm as svm
import ageas.classifier.cnn as cnn
import ageas.classifier.rnn as rnn
import ageas.classifier.xgb as xgb
import ageas.classifier as classifier
from ageas.database_setup.binary_class import Process



class Train:
    """
    Train out well performing classification models
    """
    def __init__(self,
                database_info,
                model_config_path = None,
                # GRN casting params
                gem_data = None,
                grn_guidance = None,
                std_value_thread = 100,
                std_ratio_thread = None,
                correlation_thread = 0.2,
                distrThred = None,
                # Model casting params
                iteration = 1,
                testSetRatio = 0.3,
                random_state = None,
                clf_keep_ratio = 1.0,
                clf_accuracy_thread = 0.9,
                grns = None):
        # load standard config file if not further specified
        if model_config_path is None:
            model_config_path = resource_filename(__name__,
                                            '../data/config/sample_config.js')
        model_config = json.decode(model_config_path)

        # Initialization
        self.grns = grns
        self.database_info = database_info

        if self.grns is None:
            # if reading in GEMs, we need to construct pseudo-cGRNs first
            if re.search(r'gem' , self.database_info.type):
                self.grns = grn.Make(database_info = self.database_info,
                                    std_value_thread = std_value_thread,
                                    std_ratio_thread = std_ratio_thread,
                                    correlation_thread = correlation_thread,
                                    gem_data = gem_data,
                                    grn_guidance = grn_guidance)
            # if we are reading in GRNs directly, just process them
            elif re.search(r'grn' , self.database_info.type):
                self.grns = None
                print('trainer.py: mode grn need to be revised here')
            else:
                raise operator.Error('Unrecogonized database type: ',
                                        self.database_info.type)
        assert self.grns is not None
        # Train out models and find the best ones
        self.models = Cast_Models(database_info = self.database_info,
                                    model_config = model_config,
                                    grnData = self.grns,
                                    iteration = iteration,
                                    testSetRatio = testSetRatio,
                                    random_state = random_state,
                                    clf_keep_ratio = clf_keep_ratio,
                                    clf_accuracy_thread = clf_accuracy_thread)

    # Save result models in given path
    def save_models(self, path):
        with open(path, 'wb') as file: pickle.dump(self.models, file)



class Cast_Models(classifier.Make_Template):
    """
    Find best models in each type of models
    """

    def __init__(self, database_info,
                        model_config,
                        grnData = None,
                        iteration = None,
                        testSetRatio = None,
                        random_state = None,
                        clf_keep_ratio = None,
                        clf_accuracy_thread = None):
        # Initialize and perform iterative training
        self.models = None
        self.model_config = model_config
        self.all_grp_ids = {}
        self.allData = None
        self.allLabel = None
        self.testSizeRatio = testSetRatio
        self.__iterative_training(database_info,
                                    grnData,
                                    iteration,
                                    testSetRatio,
                                    random_state,
                                    clf_keep_ratio,
                                    clf_accuracy_thread)

        # Concat models together based on performace
        temp = []
        for models in self.models:
            for model in models.models:
                temp.append(model)
        temp.sort(key = lambda x:x[-1], reverse = True)
        self.models = temp

        # Keep best performancing models
        self._filterModels(clf_keep_ratio, clf_accuracy_thread)
        # Filter based on all data performace
        self._calculateAllDataAccuracy()
        self.models.sort(key = lambda x:x[-1], reverse = True)
        self._filterModels(clf_keep_ratio, clf_accuracy_thread)
        print('Keeping ', len(self.models), ' models')
        self.allData = pd.DataFrame(self.allData, columns = self.all_grp_ids)
        del self.all_grp_ids

    # Make model sets based on given config
    def __initialize_classifiers(self, config):
        list = []
        if 'GBM' in config: list.append(xgb.Make(config = config['GBM']))
        if 'SVM' in config: list.append(svm.Make(config = config['SVM']))
        if 'CNN' in config: list.append(cnn.Make(config = config['CNN']))
        if 'RNN' in config: list.append(rnn.Make(config = config['RNN']))
        return list

    # Generate training data and testing data iteratively
    # Then train out models in model sets
    # Only keep top performancing models in each set
    def __iterative_training(self,
                            database_info,
                            grnData,
                            iteration,
                            testSetRatio,
                            random_state,
                            clf_keep_ratio,
                            clf_accuracy_thread):
        for i in range(iteration):
            # Change random state for each iteration
            if random_state is not None: random_state = i * random_state
            dataSets = Process(database_info,
                                grnData,
                                testSetRatio,
                                random_state,
                                self.all_grp_ids,
                                self.allData,
                                self.allLabel)
            dataSets.auto_inject_fake_grps()

            # Update allGRP_IDs, allData, allLabel after first iteration
            # to try to avoid redundant calculation
            if i == 0 and self.allData is None and self.allLabel is None:
                print('Total GRP amount: ', len(dataSets.all_grp_ids))
                self.all_grp_ids = dataSets.all_grp_ids
                self.testSizeRatio = len(dataSets.dataTest)
                self.allData = dataSets.dataTrain + dataSets.dataTest
                self.allLabel = np.concatenate((dataSets.labelTrain,
                                            dataSets.labelTest))
                self.__check_input_matrix_size()
                self.models = self.__initialize_classifiers(self.model_config)
                # Clear redundant data
                database_info = None
                grnData = None
                if len(self.allData) !=  len(self.allLabel):
                    raise pre.Preprocess_Error('Full data extraction Error')
            elif i == 0:
                self.__check_input_matrix_size()
                self.models = self.__initialize_classifiers(self.model_config)
            # For testing
            # else:
            #     print('constant allIDs' ,
                        # dataSets.all_grp_ids == self.all_grp_ids )

            # Do trainings
            for modelSet in self.models:
                modelSet.train(dataSets, clf_keep_ratio, clf_accuracy_thread)
            print('Finished iterative trainig: ', i)

    # Check whether matrix sizes are reasonable or not
    def __check_input_matrix_size(self):
        matlen = int(math.sqrt(len(self.all_grp_ids)))
        # m is square shaped data dimmensions
        m = [matlen, matlen]
        if 'CNN' in self.model_config and 'Hybrid' in self.model_config['CNN']:
            for id in self.model_config['CNN']['Hybrid']:
                mat_size = self.model_config['CNN']['Hybrid'][id]['matrix_size']
                if mat_size is not None:
                    if mat_size[0] * mat_size[1] != len(self.all_grp_ids):
                        warn('Ignored illegal matrixsize config:'+str(mat_size))
                        self.model_config['CNN']['Hybrid'][id]['matrix_size']= m

                elif mat_size is None:
                    warn('No valid matrix size in config')
                    warn('Using 1:1 matrix size: ' + str(idealMatSize))
                    self.model_config['CNN']['Hybrid'][id]['matrix_size'] = m

                if len(mat_size) != 2:
                    warn('No valid matrix size in config')
                    warn('Using 1:1 matrix size: ' + str(idealMatSize))
                    self.model_config['CNN']['Hybrid'][id]['matrix_size'] = m

    # Re-assign accuracy based on all data performance
    def _calculateAllDataAccuracy(self,):
        i = 0
        for record in self.models:
            model = record[0]
            accuracy = record[-1]
            modType = str(type(model))
            i+=1
            # Handel SVM and XGB cases
            # Or maybe any sklearn-style case
            if re.search(r'SVM', modType) or re.search(r'XGB', modType):
                allSizeResult = model.clf.predict(self.allData)
                allSizeAccuracy = difflib.SequenceMatcher(None, allSizeResult,
                                                        self.allLabel).ratio()
            # Hybrid CNN cases and 1D CNN cases
            elif re.search(r'Hybrid', modType) or re.search(r'1D', modType):
                # Enter eval mode and turn off gradient calculation
                model.eval()
                with torch.no_grad():
                    allSizeResult = model(cnn.Make.reshapeData(self.allData))
                allSizeAccuracy, allSizeResult = self._evalNN(allSizeResult,
                                                                self.allLabel)
            # RNN type handling
            elif re.search(r'LSTM', modType) or re.search(r'GRU', modType):
                # Enter eval mode and turn off gradient calculation
                model.eval()
                with torch.no_grad():
                    allSizeResult = model(rnn.Make.reshapeData(self.allData))
                allSizeAccuracy, allSizeResult = self._evalNN(allSizeResult,
                                                                self.allLabel)
            else:
                raise operator.Error('All data test cannot handle: ', modType)
            record[-1] = allSizeResult
            record.append(allSizeAccuracy)
            # For debug purpose
            # print('Performined all data test on model', i,
            #         ' type:', modType, '\n',
            #         'test set accuracy:', round(accuracy, 2),
            #         ' all data accuracy: ', round(allSizeAccuracy, 2), '\n')
        self.models.sort(key = lambda x:x[-1], reverse = True)


    # Evaluate Neural Network based methods'accuracies
    def _evalNN(self, result, label):
        modifiedResult = []
        correct = 0
        for i in range(len(result)):
            if result[i][0] > result[i][1]: predict = 0
            else: predict = 1
            if predict == label[i]: correct += 1
            modifiedResult.append(predict)
        accuracy = correct / len(label)
        return accuracy, modifiedResult
