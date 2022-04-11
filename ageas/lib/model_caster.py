#!/usr/bin/env python3
"""
This file contains classes to find best models for further analysis

author: jy, nkmtmsys
"""


import gc
import re
import time
import math
import torch
import difflib
import numpy as np
import pandas as pd
from warnings import warn
import ageas.classifier.svm as svm
import ageas.classifier.cnn as cnn
import ageas.classifier.xgb as xgb
import ageas.classifier as classifier
from ageas.database_setup.binary_class import Process


class Mod_Cas_Error(Exception):
    """
    Models casting related error handling
    """
    pass



class Cast(classifier.Make_Template):
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
        self._iterativeTraining(database_info,
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
        print('Finished model concat')

        # Keep best performancing models
        self._filterModels(clf_keep_ratio, clf_accuracy_thread)
        # Filter based on all data performace
        self._calculateAllDataAccuracy()
        self.models.sort(key = lambda x:x[-1], reverse = True)
        self._filterModels(clf_keep_ratio, clf_accuracy_thread)
        print('Finished all data test on candidate models')
        print('Keeping ', len(self.models), ' models')
        self.allData = pd.DataFrame(self.allData, columns = self.all_grp_ids)
        print('Changed allData into pandas data frame format')
        del self.all_grp_ids
        gc.collect()

    # Make model sets based on given config
    def _initializeModelSets(self, config):
        models = []
        start = time.time()
        if 'XGB' in config: models.append(xgb.Make(config = config['XGB']))
        if 'SVM' in config: models.append(svm.Make(config = config['SVM']))
        if 'CNN' in config: models.append(cnn.Make(config = config['CNN']))
        print('Initialize models: ', time.time() - start)
        return models

    # Generate training data and testing data iteratively
    # Then train out models in model sets
    # Only keep top performancing models in each set
    def _iterativeTraining(self, database_info,
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
                print('allIDs length: ', len(dataSets.all_grp_ids))
                self.all_grp_ids = dataSets.all_grp_ids
                self.testSizeRatio = len(dataSets.dataTest)
                self.allData = dataSets.dataTrain + dataSets.dataTest
                self.allLabel = np.concatenate((dataSets.labelTrain,
                                            dataSets.labelTest))
                self._checkMatrixConfig()
                self.models = self._initializeModelSets(self.model_config)
                # Clear redundant data
                database_info = None
                grnData = None
                if len(self.allData) !=  len(self.allLabel):
                    raise pre.Preprocess_Error('Full data extraction Error')
            elif i == 0:
                self._checkMatrixConfig()
                self.models = self._initializeModelSets(self.model_config)
            # For testing
            # else:
            #     print('constant allIDs' ,
                        # dataSets.all_grp_ids == self.all_grp_ids )

            # Do trainings
            for modelSet in self.models:
                modelSet.train(dataSets, clf_keep_ratio, clf_accuracy_thread)
            print('Finished iterative trainig: ', i)
            gc.collect()

    # Check whether matrix sizes are reasonable or not
    def _checkMatrixConfig(self):
        matlen = int(math.sqrt(len(self.all_grp_ids)))
        idealMatSize = [matlen, matlen]
        if 'CNN' in self.model_config:
            if self.model_config['CNN']['matrixSizes'] is not None:
                remove = []

                for size in self.model_config['CNN']['matrixSizes']:
                    if size[0] * size[1] != len(self.all_grp_ids):
                        warn('Ignored illegal matrix size setting:' + str(size))
                        remove.append(size)

                for ele in remove:
                    self.model_config['CNN']['matrixSizes'].remove(ele)

            elif self.model_config['CNN']['matrixSizes'] is None:
                warn('No valid matrix size in config')
                warn('Using 1:1 matrix size: ' + str(idealMatSize))
                self.model_config['CNN']['matrixSizes'] = [idealMatSize]

            if len(self.model_config['CNN']['matrixSizes']) == 0:
                warn('No valid matrix size in config')
                warn('Using 1:1 matrix size: ' + str(idealMatSize))
                self.model_config['CNN']['matrixSizes'] = [idealMatSize]

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
            elif re.search(r'Hybrid', modType) or re.search(r'OneD', modType):
                # Enter eval mode and turn off gradient calculation
                model.eval()
                with torch.no_grad():
                    allSizeResult = model(cnn.Make.reshapeData(self.allData))
                allSizeAccuracy, allSizeResult = self._evalNN(allSizeResult,
                                                                self.allLabel)
            else:
                raise Mod_Cas_Error('All data test cannot handle: ', modType)
            record[-1] = allSizeResult
            record.append(allSizeAccuracy)
            # For debug purpose
            # print('Performined all data test on model', i,
            #         ' type:', modType, '\n',
            #         'test set accuracy:', round(accuracy, 2),
            #         ' all data accuracy: ', round(allSizeAccuracy, 2), '\n')
            gc.collect()
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
