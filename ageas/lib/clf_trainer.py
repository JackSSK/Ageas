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
from warnings import warn
import ageas.lib as lib
import ageas.classifier as clf
import ageas.classifier.gnb as gnb
import ageas.classifier.rfc as rfc
import ageas.classifier.xgb as xgb
import ageas.classifier.svm as svm
import ageas.classifier.rnn as rnn
import ageas.classifier.gru as gru
import ageas.classifier.lstm as lstm
import ageas.classifier.logit as logit
import ageas.classifier.transformer as transformer
import ageas.classifier.cnn_1d as cnn_1d
import ageas.classifier.cnn_hybrid as cnn_hybrid
import ageas.database_setup.binary_class as binary_class



class Train(clf.Make_Template):
    """
    Train out well performing classification models
    """
    def __init__(self,
                 psGRNs = None,
                 cpu_mode = False,
                 database_info = None,
                 model_config = None,
                 random_state = None,
                 test_split_set = False,
                ):
        super(Train, self).__init__()
        # Initialization
        self.grns = psGRNs
        self.models = None
        self.all_data = None
        self.all_labels = None
        self.cpu_mode = cpu_mode
        self.random_state = random_state
        self.model_config = model_config
        self.database_info = database_info
        self.test_split_set = test_split_set

    def general_process(self,
                        train_size:float = 0.3,
                        clf_keep_ratio:float = None,
                       ):
        """
        Generate training data and testing data iteratively
        Then train out models in model sets
        Only keep top performancing models in each set
        """
        data = binary_class.Process(
            self.database_info,
            self.grns,
            train_size,
            self.random_state,
            self.all_data,
            self.all_labels
        )
        data.auto_inject_fake_grps()

        # Update allGRP_IDs, allData, allLabel after first iteration
        # to try to avoid redundant calculation
        if self.all_data is None and self.all_labels is None:
            print('Total GRP amount: ', len(data.all_grp_ids))
            self.all_data = data.dataTrain + data.dataTest
            self.all_labels = np.concatenate((data.labelTrain, data.labelTest))
            assert len(self.all_data) == len(self.all_labels)
            self.all_data = pd.DataFrame(self.all_data,columns=data.all_grp_ids)

        # Do trainings
        temp = list()
        self.models = self.__initialize_classifiers(self.model_config)
        for model_set in self.models:
            model_set.train(data, self.test_split_set)
            if self.test_split_set:
                model_set._performance_filter(clf_keep_ratio)
            # Concat all models of different type to one list
            for model in model_set.models:
                temp.append(model)
        self.models = temp

        # Keep best performancing models in local test
        if self.test_split_set and clf_keep_ratio is not None:
            self._performance_filter(clf_keep_ratio)

        # Filter based on global test performace
        self.models = self.update_model_records(
            model_records = self.models,
            data = self.all_data.to_numpy(),
            label = self.all_labels
        )
        self._performance_filter(clf_keep_ratio)
        print('Keeping ', len(self.models), ' models')


    def successive_pruning(self,
                           iteration:int = 3,
                           clf_keep_ratio:float = 0.5,
                           max_train_size:float = 0.9,
                          ):
        """
        Train out models in Successive Halving manner
        Amount of training data is set as limited resouce
        While accuracy is set as evaluation standard
        """
        # set iteration to 0 if not doing SHA
        if iteration is None: iteration = 1

        # initialize training data set
        init_train_size = float(1 / pow(2, iteration - 1))
        train_size = 0
        for i in range(iteration - 1):
            train_size = float(init_train_size * pow(2, i))
            print('Iteration:', i + 1, ' with training size:', train_size)
            # remove more and more portion as more resource being avaliable
            self.general_process(
                train_size = train_size,
                clf_keep_ratio = max(1 - train_size, clf_keep_ratio),
            )
            self.__prune_clf_config(id_keep={x.model.id:''for x in self.models})

        print('Iteration:Last with training size:', max_train_size)
        self.general_process(
            train_size = max_train_size,
            clf_keep_ratio = clf_keep_ratio
        )

        self.__prune_clf_config(id_keep = {x.model.id:'' for x in self.models})

        total_model = 0
        for genra in self.model_config:
            total_model += len(self.model_config[genra])
        print('Selecting ', total_model, ' Models after Model Selection')

    # Re-assign accuracy based on all data performance
    def update_model_records(self, model_records, data, label):
        for i, record in enumerate(model_records):
            # Handel SVM and XGB cases
            # Or maybe any sklearn-style case
            if (record.model.model_type == 'SVM' or
                record.model.model_type == 'Logit' or
                record.model.model_type == 'GNB' or
                record.model.model_type == 'RFC' or
                record.model.model_type == 'XGB_GBM'):
                new_record = self._evaluate_sklearn(record.model, data, label)
            # RNN type handling + CNN cases
            elif (record.model.model_type == 'RNN' or
                  record.model.model_type == 'LSTM' or
                  record.model.model_type == 'GRU' or
                  record.model.model_type == 'Transformer' or
                  re.search(r'CNN', record.model.model_type)):
                # Enter eval mode and turn off gradient calculation
                new_record = self._evaluate_torch(record.model, data, label)
            else:
                raise lib.Error('Cannot handle: ', record.model.model_type)
            model_records[i] = new_record
        model_records.sort(key = lambda x:x.accuracy, reverse = True)
        return model_records

    # clear stored data
    def clear_data(self):
        self.grns = None
        self.models = None
        self.all_data = None
        self.all_labels = None

    # Save result models in given path
    def save_models(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self.models, file)

    # Make model sets based on given config
    def __initialize_classifiers(self, config):
        list = []
        if 'Logit' in config:
            list.append(logit.Make(
                    config = config['Logit'],
                    cpu_mode = self.cpu_mode,
                )
            )
        if 'Transformer' in config:
            list.append(transformer.Make(
                    config = config['Transformer'],
                    cpu_mode = self.cpu_mode,
                )
            )
        if 'RFC' in config:
            list.append(rfc.Make(
                    config = config['RFC'],
                    cpu_mode = self.cpu_mode,
                )
            )
        if 'GNB' in config:
            list.append(gnb.Make(
                    config = config['GNB'],
                    cpu_mode = self.cpu_mode,
                )
            )
        if 'GBM' in config:
            list.append(xgb.Make(
                    config = config['GBM'],
                    cpu_mode = self.cpu_mode,
                )
            )
        if 'SVM' in config:
            list.append(svm.Make(
                    config = config['SVM'],
                    cpu_mode = self.cpu_mode,
                )
            )
        if 'CNN_1D' in config:
            list.append(cnn_1d.Make(
                    config = config['CNN_1D'],
                    cpu_mode = self.cpu_mode,
                )
            )
        if 'CNN_Hybrid' in config:
            list.append(cnn_hybrid.Make(
                    config = config['CNN_Hybrid'],
                    cpu_mode = self.cpu_mode,
                    grp_amount = len(self.all_data.columns),
                )
            )
        if 'RNN' in config:
            list.append(rnn.Make(
                    config = config['RNN'],
                    cpu_mode = self.cpu_mode,
                )
            )
        if 'LSTM' in config:
            list.append(lstm.Make(
                    config = config['LSTM'],
                    cpu_mode = self.cpu_mode,
                )
            )
        if 'GRU' in config:
            list.append(gru.Make(
                    config = config['GRU'],
                    cpu_mode = self.cpu_mode,
                )
            )
        return list

    # delete model configs not on while list(dict)
    def __prune_clf_config(self, id_keep):
        result = {}
        for genra in self.model_config:
            temp = {}
            for id in self.model_config[genra]:
                if id in id_keep:
                    temp[id] = self.model_config[genra][id]
                else:
                    print('     Pruning:', id)
            if len(temp) > 0:
                result[genra] = temp
        self.model_config = result
