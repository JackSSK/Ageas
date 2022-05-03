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
import ageas.lib as lib
import ageas.classifier as clf
import ageas.classifier.xgb as xgb
import ageas.classifier.svm as svm
import ageas.classifier.rnn as rnn
import ageas.classifier.gru as gru
import ageas.classifier.lstm as lstm
import ageas.classifier.cnn_1d as cnn_1d
import ageas.classifier.cnn_hybrid as cnn_hybrid
import ageas.database_setup.binary_class as binary_class



class Train(clf.Make_Template):
    """
    Train out well performing classification models
    """
    def __init__(self,
                pcGRNs = None,
                database_info = None,
                model_config = None,
                test_set_ratio = 0.3,
                random_state = None,
                clf_keep_ratio = 1.0,
                clf_accuracy_thread = 0.9,):
        super(Train, self).__init__()
        # Initialization
        self.grns = pcGRNs
        self.models = None
        self.model_config = model_config
        self.all_grp_ids = {}
        self.allData = None
        self.allLabel = None

        # Train out models and find the best ones
        self.process(database_info,
                    test_set_ratio,
                    random_state,
                    clf_keep_ratio,
                    clf_accuracy_thread)

    # Generate training data and testing data iteratively
    # Then train out models in model sets
    # Only keep top performancing models in each set
    def process(self,
                database_info,
                test_set_ratio,
                random_state,
                clf_keep_ratio,
                clf_accuracy_thread):
        # # Change random state for each iteration
        # if random_state is not None: random_state = i * random_state
        data = binary_class.Process(database_info,
                                    self.grns,
                                    test_set_ratio,
                                    random_state,
                                    self.all_grp_ids,
                                    self.allData,
                                    self.allLabel)
        data.auto_inject_fake_grps()

        # Update allGRP_IDs, allData, allLabel after first iteration
        # to try to avoid redundant calculation
        if self.allData is None and self.allLabel is None:
            print('Total GRP amount: ', len(data.all_grp_ids))
            self.all_grp_ids = data.all_grp_ids
            self.allData = data.dataTrain + data.dataTest
            self.allLabel = np.concatenate((data.labelTrain, data.labelTest))
            assert len(self.allData) == len(self.allLabel)

        # Do trainings
        self.models = self.__initialize_classifiers(self.model_config)
        for modelSet in self.models:
            modelSet.train(data)
            modelSet._filter_models(clf_keep_ratio, clf_accuracy_thread)

        # Concat models together based on performace
        temp = []
        for models in self.models:
            for model in models.models:
                temp.append(model)
        self.models = temp
        # Keep best performancing models in local test
        self._filter_models(clf_keep_ratio, clf_accuracy_thread)
        # Filter based on global test performace
        self.models = self.get_clf_accuracy(clf_list = self.models,
                                            data = self.allData,
                                            label = self.allLabel)
        self._filter_models(clf_keep_ratio, clf_accuracy_thread)
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

    # Re-assign accuracy based on all data performance
    def get_clf_accuracy(self, clf_list, data, label):
        i = 0
        for record in clf_list:
            model = record[0]
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
                    pred_result = model(clf.reshape_tensor(data))
                pred_accuracy, pred_result = self.__evaluate_NN(pred_result,
                                                                label)
            # RNN type handling
            elif (re.search(r'rnn', clf_type) or
                    re.search(r'lstm', clf_type) or
                    re.search(r'gru', clf_type)):
                # Enter eval mode and turn off gradient calculation
                model.eval()
                with torch.no_grad():
                    pred_result = model(clf.reshape_tensor(data))
                pred_accuracy, pred_result = self.__evaluate_NN(pred_result,
                                                                label)
            else:
                raise lib.Error('Cannot handle classifier: ', clf_type)
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

    # Save result models in given path
    def save_models(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self.models, file)


class Successive_Halving(Train):
    """
    Train out models in Successive Halving manner

    Amount of training data is set as limited resouce
    While accuracy is set as evaluation standard
    """

    def __init__(self,
                pcGRNs = None,
                database_info = None,
                model_config = None,
                iteration = 4,
                random_state = None,
                clf_keep_ratio = 1.0,
                clf_accuracy_thread = 0.9,):
        super(Successive_Halving, self).__init__()
        self.grns = pcGRNs
        self.models = None
        self.model_config = model_config
        self.all_grp_ids = {}
        self.allData = None
        self.allLabel = None
        self.init_train_ratio = float(1 / pow(2, iteration))
        for i in range(iteration):
            test_set_ratio = float(1 - self.init_train_ratio * pow(2, i))
