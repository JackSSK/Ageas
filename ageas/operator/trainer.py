#!/usr/bin/env python3
"""
Ageas Reborn
This file contains classes to find best models for further analysis

author: jy, nkmtmsys
"""

import re
import torch
import pickle
import difflib
import numpy as np
import pandas as pd
import ageas.operator as operator
import ageas.lib.pcgrn_caster as grn
import ageas.classifier.svm as svm
import ageas.classifier.cnn_1d as cnn_1d
import ageas.classifier.cnn_hybrid as cnn_hybrid
import ageas.classifier.rnn as rnn
import ageas.classifier.lstm as lstm
import ageas.classifier.gru as gru
import ageas.classifier.xgb as xgb
import ageas.classifier as classifier
from ageas.database_setup.binary_class import Process



class Train:
    """
    Train out well performing classification models
    """
    def __init__(self,
                database_info,
                model_config = None,
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
        # Initialization
        self.grns = grns
        # Generate pcGRNs if not avaliable
        if self.grns is None:
            # if reading in GEMs, we need to construct pseudo-cGRNs first
            if re.search(r'gem' , database_info.type):
                self.grns = grn.Make(database_info = database_info,
                                    std_value_thread = std_value_thread,
                                    std_ratio_thread = std_ratio_thread,
                                    correlation_thread = correlation_thread,
                                    gem_data = gem_data,
                                    grn_guidance = grn_guidance)
            # if we are reading in GRNs directly, just process them
            elif re.search(r'grn' , database_info.type):
                self.grns = None
                print('trainer.py: mode grn need to be revised here')
            else:
                raise operator.Error('Unrecogonized database type: ',
                                        database_info.type)
        assert self.grns is not None
        # Train out models and find the best ones
        self.models = Cast_Models(database_info = database_info,
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
        self.models = self.get_clf_accuracy(clf_list = self.models,
                                                data = self.allData,
                                                label = self.allLabel)
        self.models.sort(key = lambda x:x[-1], reverse = True)
        self._filterModels(clf_keep_ratio, clf_accuracy_thread)
        print('Keeping ', len(self.models), ' models')
        self.allData = pd.DataFrame(self.allData, columns = self.all_grp_ids)
        del self.all_grp_ids

    # Make model sets based on given config
    def __initialize_classifiers(self, config):
        list = []
        if 'GBM' in config:
            list.append(xgb.Make(config = config['GBM']))
        if 'SVM' in config:
            list.append(svm.Make(config = config['SVM']))
        if 'CNN_1D' in config:
            list.append(cnn_1d.Make(config = config['CNN_1D']))
        if 'CNN_Hybrid' in config:
            list.append(cnn_hybrid.Make(config = config['CNN_Hybrid'],
                                        grp_amount = len(self.all_grp_ids)))
        if 'RNN' in config:
            list.append(rnn.Make(config = config['RNN']))
        if 'LSTM' in config:
            list.append(lstm.Make(config = config['LSTM']))
        if 'GRU' in config:
            list.append(gru.Make(config = config['GRU']))
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
                self.allData = dataSets.dataTrain + dataSets.dataTest
                self.allLabel = np.concatenate((dataSets.labelTrain,
                                            dataSets.labelTest))
                self.models = self.__initialize_classifiers(self.model_config)
                # Clear redundant data
                database_info = None
                grnData = None
                if len(self.allData) !=  len(self.allLabel):
                    raise operator.Error('Full data extraction Error')
            elif i == 0:
                self.models = self.__initialize_classifiers(self.model_config)

            # Do trainings
            for modelSet in self.models:
                modelSet.train(dataSets, clf_keep_ratio, clf_accuracy_thread)
            print('Finished iterative trainig: ', i)

    # Re-assign accuracy based on all data performance
    def get_clf_accuracy(self, clf_list, data, label):
        i = 0
        for record in clf_list:
            model = record[0]
            accuracy = record[-1]
            clf_type = str(type(model))
            i+=1
            # Handel SVM and XGB cases
            # Or maybe any sklearn-style case
            if re.search(r'svm', clf_type) or re.search(r'xgb', clf_type):
                pred_result = model.clf.predict(data)
                pred_accuracy = difflib.SequenceMatcher(None,
                                                        pred_result,
                                                        label).ratio()
            # CNN cases
            elif re.search(r'cnn_', clf_type):
                # Enter eval mode and turn off gradient calculation
                model.eval()
                with torch.no_grad():
                    pred_result = model(classifier.reshape_tensor(data))
                pred_accuracy, pred_result = self.__evaluate_NN(pred_result,
                                                                label)
            # RNN type handling
            elif (re.search(r'rnn', clf_type) or
                    re.search(r'lstm', clf_type) or
                    re.search(r'gru', clf_type)):
                # Enter eval mode and turn off gradient calculation
                model.eval()
                with torch.no_grad():
                    pred_result = model(classifier.reshape_tensor(data))
                pred_accuracy, pred_result = self.__evaluate_NN(pred_result,
                                                                label)
            else:
                raise operator.Error('Cannot handle classifier: ', clf_type)
            record[-1] = pred_result
            record.append(pred_accuracy)
            # For debug purpose
            # print('Performined all data test on model', i,
            #         ' type:', clf_type, '\n',
            #         'test set accuracy:', round(accuracy, 2),
            #         ' all data accuracy: ', round(pred_accuracy, 2), '\n')
        clf_list.sort(key = lambda x:x[-1], reverse = True)
        return clf_list


    # Evaluate Neural Network based methods'accuracies
    def __evaluate_NN(self, result, label):
        modifiedResult = []
        correct = 0
        for i in range(len(result)):
            if result[i][0] > result[i][1]: predict = 0
            else: predict = 1
            if predict == label[i]: correct += 1
            modifiedResult.append(predict)
        accuracy = correct / len(label)
        return accuracy, modifiedResult
