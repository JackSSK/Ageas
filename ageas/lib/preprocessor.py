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



class GRN:
    """
    Seperate sample grn files into training sets and testing sets by given ratio
    Then prepare sample data to be ready for training and analysis process
    """
    def __init__(self, databaseSetting = None,
                        grnData = None,
                        testSize = 0.3,
                        randomState = None,
                        allIDs = {},
                        fullData = None,
                        fullLabel = None,
                        testSizeInt = None):
        # Initialization
        if testSizeInt is None: self.testSize = testSize
        else:                   self.testSize = testSizeInt
        self.randomState = randomState
        self.allIDs = allIDs
        self.mode = self._modeSelection(databaseSetting, fullData, fullLabel)
        # Go through databaseSetting based protocol
        if self.mode == 'database':
            self._dbProtocol(databaseSetting, grnData)
        elif self.mode == 'fullData':
            self._fullDataProtocol(fullData, fullLabel)
        else:
            raise Preprocess_Error('How can you make it here???')

    # Select a protocol to do preprocess based on avaliable data
    # If both protocol is doable, will go with fullData method
    def _modeSelection(self, databaseSetting, fullData, fullLabel):
        if databaseSetting is not None:
            if fullData is not None or fullLabel is not None:
                mode = 'fullData'
            else:
                mode = 'database'
        else:
            if fullData is not None and fullLabel is not None:
                mode = 'fullData'
            else:
                raise Preprocess_Error('Please determine which protocol to use')
        return mode

    # Process in database mode
    # grnData is only necessary in Gene_Exp class actually
    def _dbProtocol(self,databaseSetting, grnData):
        self._makeTrainTestSets(databaseSetting)
        self.dataTrain = self._prepareFeatures(self.fileTrain)
        self.dataTest = self._prepareFeatures(self.fileTest)
        # self._allIDsCheck()
        # Clear unnecessary data
        del databaseSetting
        gc.collect()

    # Makke training/testing data and lable arrays based on given full data
    def _fullDataProtocol(self, fullData, fullLabel):
            data = train_test_split(fullData, fullLabel,
                    test_size = self.testSize, random_state = self.randomState)
            self.dataTrain = data[0]
            self.dataTest = data[1]
            self.labelTrain = data[2]
            self.labelTest = data[3]
            # check whether allIDs and testSizeInt are avaliable or not
            if len(self.allIDs) == 0:
                raise Preprocess_Error('allIDs not provided in fullData mode')

    # Makke training/testing data and lable arrays based on database settings
    def _makeTrainTestSets(self, databaseSetting):
        # class1Result is [dataTrainC1, dataTestC1, lableTrainC1, labelTestC1]
        class1Result = self._splitTrainTest(databaseSetting.class1_path,
                                            databaseSetting.label1)
        # similar with class1
        class2Result = self._splitTrainTest(databaseSetting.class2_path,
                                            databaseSetting.label2)
        self.fileTrain = class1Result[0] + class2Result[0]
        self.fileTest = class1Result[1] + class2Result[1]
        self.labelTrain = np.array(class1Result[2] + class2Result[2])
        self.labelTest = np.array(class1Result[3] + class2Result[3])

    # standardize feature data
    # to make sure all training and testing samples
    # will be in same dimmension
    def _prepareFeatures(self, files):
        result = []
        for filename in files:
            features = ''
            """ToDo: Due to lack of .grn data, this part needs a test later"""

            # handling .grn cases
            if re.search(r'\.grn', filename):
                grn = grnTool.Reader(filename)
                grn_copy = {ele:'' for ele in grn.entryCoords}
                for ele in self.allIDs:
                    if ele in grn.entryCoords:
                        features += str(grn.get(ele)['correlation']) + ';'
                        # Update grn_cop if ele already in allIDs
                        del grn_copy[ele]
                    else:
                        features += '0.0;'
                for ele in grn_copy:
                    self.allIDs[ele] = ''
                    features += str(grn.get(ele)['correlation']) + ';'
                grn.close()

            # handling .js cases
            elif re.search(r'\.js', filename):
                grn = json.decode(filename)
                grn_copy = {ele:'' for ele in grn}
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
            result.append(features)

        # Clear unecessary data
        del grn
        del features
        gc.collect()
        # Add zeros for position holding
        self._addZeros(result)
        return result

    # seperate files in given path into training
    # and testing sets based on ratio
    def _splitTrainTest(self, path, label):
        data_sample = []
        label_sample = []
        for filename in os.listdir(path):
            data_sample.append(path + '/' + filename)
            label_sample.append(label)
        return train_test_split(data_sample, label_sample,
            test_size = self.testSize, random_state = self.randomState)

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



class Gene_Exp(GRN):
    """
    Seperate sample gene expression data into training sets and testing sets
    by given ratio
    Then prepare sample data to be ready for training and analysis process
    """
    # Process in database mode
    def _dbProtocol(self,databaseSetting, grnData):
        self._makeTrainTestSets(databaseSetting)
        self.dataTrain, self.dataTest = self._prepareFeatures(grnData)
        # self._allIDsCheck()
        # Clear unnecessary data
        del grnData
        del databaseSetting
        gc.collect()

    # standardize feature data
    # to make sure all training and testing samples
    # will be in same dimmension
    def _prepareFeatures(self, grnData):
        dataTrain = []
        dataTest = []
        self._updateTrainTest(grnData.class1_pcGRNs, dataTrain, dataTest)
        self._updateTrainTest(grnData.class2_pcGRNs, dataTrain, dataTest)
        # Add zeros for position holding
        self._addZeros(dataTrain)
        self._addZeros(dataTest)
        return dataTrain, dataTest

    # Update training and testing set based on given expression data
    def _updateTrainTest(self, grns, dataTrain, dataTest):
        for file in grns:
            grn = grns[file]
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
            if file in self.fileTrain:
                dataTrain.append(features)
            elif file in self.fileTest:
                dataTest.append(features)
