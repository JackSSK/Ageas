#!/usr/bin/env python3
"""
Ageas Reborn
"""



import os
import re
import gc
import math
import numpy as  np
import pandas as pd
from sklearn.model_selection import train_test_split
import ageas.tool.grn as grnTool
import ageas.tool.json as json



class Preprocess_Error(Exception):
    """
    Data Preprocessing related error handling
    """
    pass



class Process(object):
    """
    Seperate sample gene expression data into training sets and testing sets
    by given ratio
    Then prepare sample data to be ready for training and analysis process
    """
    def __init__(self, database_info = None,
                        grnData = None,
                        test_set_ratio = 0.3,
                        randomState = None,
                        allIDs = {},
                        fullData = None,
                        fullLabel = None,
                        test_set_size = None):
        # Initialization
        if test_set_size is None:
            self.test_set_ratio = test_set_ratio
        else:
            self.test_set_ratio = test_set_size
        self.randomState = randomState
        self.allIDs = allIDs
        # Go through database_info based protocol
        if fullData is None or fullLabel is None:
            self._dbProtocol(database_info, grnData)
        elif fullData is not None and fullLabel is not None:
            self._fullDataProtocol(fullData, fullLabel)
        else:
            raise Preprocess_Error('Preprocessor Error: case not catched')

    # Process in database mode
    def _dbProtocol(self, database_info, grnData):
        # class1Result is [dataTrainC1, dataTestC1, lableTrainC1, labelTestC1]
        class1Result = self._splitTrainTest(grnData.class1_pcGRNs,
                                            database_info.label1)
        # similar with class1
        class2Result = self._splitTrainTest(grnData.class1_pcGRNs,
                                            database_info.label2)
        self.labelTrain = np.array(class1Result[2] + class2Result[2])
        self.labelTest = np.array(class1Result[3] + class2Result[3])
        self.dataTrain = []
        self.dataTest = []
        # standardize feature data
        # to make sure all training and testing samples
        # will be in same dimmension
        self._updateTrainTest(grnData.class1_pcGRNs,
                                train_set = class1Result[0],
                                test_set = class1Result[1])
        self._updateTrainTest(grnData.class2_pcGRNs,
                                train_set = class2Result[0],
                                test_set = class2Result[1])
        # Add zeros for position holding
        self._addZeros(self.dataTrain)
        self._addZeros(self.dataTest)
        # self._allIDsCheck()
        # Clear unnecessary data
        del grnData
        del database_info
        gc.collect()

    # Update training and testing set based on given expression data
    def _updateTrainTest(self, grns, train_set, test_set):
        for sample in grns:
            grn = grns[sample]
            grn_copy = {ele:'' for ele in grn}
            features = ''
            for ele in self.allIDs:
                if ele in grn:
                    features += str(grn[ele]['correlation']) + ';'
                    # Update grn_cop if ele already in allIDs
                    del grn_copy[ele]
                else: features += '0.0;'
            for ele in grn_copy:
                self.allIDs[ele] = ''
                features += str(grn[ele]['correlation']) + ';'
            # Change every elements into float type
            features = list(map(float, features.split(';')[:-1]))
            if sample in train_set:
                self.dataTrain.append(features)
            elif sample in test_set:
                self.dataTest.append(features)

    # Makke training/testing data and lable arrays based on given full data
    def _fullDataProtocol(self, fullData, fullLabel):
            data = train_test_split(fullData,
                                    fullLabel,
                                    test_size = self.test_set_ratio,
                                    random_state = self.randomState)
            self.dataTrain = data[0]
            self.dataTest = data[1]
            self.labelTrain = data[2]
            self.labelTest = data[3]
            # check whether allIDs and test_set_size are avaliable or not
            if len(self.allIDs) == 0:
                raise Preprocess_Error('allIDs not provided in fullData mode')

    # seperate files in given path into training
    # and testing sets based on ratio
    def _splitTrainTest(self, grnData, label):
        data_sample = []
        label_sample = []
        for sample in grnData:
            data_sample.append(sample)
            label_sample.append(label)
        return train_test_split(data_sample,
                                label_sample,
                                test_size = self.test_set_ratio,
                                random_state = self.randomState)

    # add zeros to each preprocessed samples
    # to make sure all training and testing samples
    # will be in same dimmension
    def _addZeros(self, dataset):
        for sample in dataset:
            if len(sample) != len(self.allIDs):
                sample += [0] * (len(self.allIDs) - len(sample))
                sample = np.array(sample)
        dataset = np.array(dataset)

    # Check whether allIDs have dupicate IDs or not
    # ToDo: need optimization to get unique ids
    def _allIDsCheck(self):
        # check whether allIDs have duplicate elements
        unique = [x for i, x in enumerate([*self.allIDs])
            if i == [*self.allIDs].index(x)]
        if len(self.allIDs) != len(unique):
            raise Preprocess_Error(
                'preprocessor malfunction: duplicate ID in allIDs')
        del unique
        gc.collect()

    # Calculate a close-to-square matrix size based on allIDs
    # for using 2D-input based machinle learning models (e.g. hybrid_model)
    def autoCastMatrixSize(self):
        total = len(self.allIDs)
        tar = math.sqrt(total)
        if int(tar) == tar: return
        elif int(tar) < tar:
            aim = int(tar) + 1
            fakeID = 'FAKE'
            fakeID_Num = aim*aim - total
            for i in range(fakeID_Num):
                id = fakeID + str(i)
                self.allIDs[id] = ''
        self._addZeros(self.dataTrain)
        self._addZeros(self.dataTest)
