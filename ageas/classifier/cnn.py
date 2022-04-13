#!/usr/bin/env python3
"""
This file contains classes to build CNN based classifiers

ToDo:
Implement _earlystopping

Note 1
Just in case if LazyLinear having problem
You may want to try torchlayers
# import torchlayers as tl

Note 2
If using SHAP deepLIFT based method or Cuptum,
repeated layers cannot be interpreted.
Thus, layer set loop-adding feature is removed in limited versions
which has layer set max fixed being 3, should be enough

author: jy, nkmtmsys
"""

import os
import gc
import torch
import itertools
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
import ageas.classifier as classifier



class OneD_Model_Limited(nn.Module):
    """
    Defining a CNN model treating input as 1D data
    with given hyperparameters
    Layer set number limited to max == 3
    """
    def __init__(self, param):
        # Initialization
        super().__init__()
        self.layerNum = param['layerNum']
        self.lossFunc = nn.CrossEntropyLoss()
        # Layer set 1
        self.pool = nn.MaxPool1d(param['maxPoolKernelSize'])
        self.conv = nn.Conv1d(1, param['convKernelNum'],
                                    param['convKernelSize'])

        # Layer set 2
        self.pool1 = nn.MaxPool1d(param['maxPoolKernelSize'])
        self.conv1 = nn.Conv1d(param['convKernelNum'],
                                param['convKernelNum'],
                                param['convKernelSize'])

        # Layer set 3
        self.pool2 = nn.MaxPool1d(param['maxPoolKernelSize'])
        self.conv2 = nn.Conv1d(param['convKernelNum'],
                                param['convKernelNum'],
                                param['convKernelSize'])

        ### Trying to avoid Lazy module since it's under development ###
        ### But so far it's working just fine, so still using lazy module ###
        # flattenLength = int(featureNum / pow(maxPoolKernelSize, layerNum))
        # self.dense = nn.Linear(flattenLength, densedSize)
        ### -------------------------------------------------------- ###

        self.dense = nn.LazyLinear(param['densedSize'])
        self.decision = nn.Linear(param['densedSize'], 2)
        self.optimizer = param['optFunc'](self.parameters(),
                                            param['learningRate'])

    # Overwrite the forward function in nn.Module
    def forward(self, input):
        input = self.pool(func.relu(self.conv(input)))
        if self.layerNum > 1:
            input = self.pool1(func.relu(self.conv1(input)))
        if self.layerNum > 2:
            input = self.pool2(func.relu(self.conv2(input)))
        if self.layerNum > 3:
            raise classifier.Error('CNN Model with more than 3 layer sets')
        input = torch.flatten(input, start_dim = 1)
        input = func.relu(self.dense(input))
        input = func.softmax(self.decision(input), dim = 1)
        return input



class Hybrid_Model_Limited(nn.Module):
    """
    Defining a CNN model treating input as 2D data
    with given hyperparameters
    then using 2 1D convolution kernels to generate layers
    Layer set number limited to max == 3
    """
    def __init__(self, param):
        super().__init__()

        # Initialization
        self.matrixSize = param['matrixSize']
        self.layerNum = param['layerNum']
        self.lossFunc = nn.CrossEntropyLoss()

        # Layer set 1
        self.poolVer = nn.MaxPool2d((1, param['maxPoolKernelSize']))
        self.convVer = nn.Conv2d(1, param['convKernelNum'],
                                    (1, self.matrixSize[1]))
        self.poolHor = nn.MaxPool2d((param['maxPoolKernelSize'], 1))
        self.convHor = nn.Conv2d(1, param['convKernelNum'],
                                    (self.matrixSize[0], 1))

        # Layer set 2
        self.poolVer1 = nn.MaxPool2d((1, param['maxPoolKernelSize']))
        self.convVer1 = nn.Conv2d(param['convKernelNum'],
                                    param['convKernelNum'],
                    (int(self.matrixSize[1] / pow(param['maxPoolKernelSize'],
                                    self.layerNum)), 1))
        self.poolHor1 = nn.MaxPool2d((param['maxPoolKernelSize'], 1))
        self.convHor1 = nn.Conv2d(param['convKernelNum'],
                                    param['convKernelNum'],
                (1, int(self.matrixSize[0] / pow(param['maxPoolKernelSize'],
                                    self.layerNum))))

        # Layer set 3
        self.poolVer2 = nn.MaxPool2d((1, param['maxPoolKernelSize']))
        self.convVer2 = nn.Conv2d(param['convKernelNum'],
                                    param['convKernelNum'],
                    (int(self.matrixSize[1] / pow(param['maxPoolKernelSize'],
                                    self.layerNum)), 1))
        self.poolHor2 = nn.MaxPool2d((param['maxPoolKernelSize'], 1))
        self.convHor2 = nn.Conv2d(param['convKernelNum'],
                                    param['convKernelNum'],
                (1, int(self.matrixSize[0] / pow(param['maxPoolKernelSize'],
                                    self.layerNum))))


        ### Same problem as 1D model ###
        # flattenLength = int(featureNum / pow(maxPoolKernelSize, layerNum))
        # self.dense = nn.Linear(flattenLength, densedSize)

        self.dense = nn.LazyLinear(param['densedSize'])
        self.decision = nn.Linear(param['densedSize'], 2)
        self.optimizer = param['optFunc'](self.parameters(),
                                            param['learningRate'])

    # Overwrite the forward function in nn.Module
    def forward(self, input):
        input = self.reshape(input)
        temp0 = self.poolVer(func.relu(self.convHor(input)))
        temp1 = self.poolHor(func.relu(self.convVer(input)))
        if self.layerNum > 1:
            temp0 = self.poolVer1(func.relu(self.convHor1(temp0)))
            temp1 = self.poolHor1(func.relu(self.convVer1(temp1)))
        if self.layerNum > 2:
            temp0 = self.poolVer2(func.relu(self.convHor2(temp0)))
            temp1 = self.poolHor2(func.relu(self.convVer2(temp1)))
        if self.layerNum > 3:
            raise classifier.Error('CNN Model with more than 3 layer sets')
        temp0 = torch.flatten(temp0, start_dim = 1)
        temp1 = torch.flatten(temp1, start_dim = 1)
        input = torch.cat((temp0, temp1), dim = 1)
        input = func.relu(self.dense(input))
        input = func.softmax(self.decision(input),dim = 1)
        return input

    # transform input(1D) into a 2D matrix
    def reshape(self, input):
        return torch.reshape(input, (input.shape[0], input.shape[1],
                                    self.matrixSize[0], self.matrixSize[1]))



class OneD_Model_Unlimited(nn.Module):
    """
    Defining a CNN model treating input as 1D data
    with given hyperparameters
    """
    def __init__(self, param):
        # Initialization
        super().__init__()
        self.layerNum = param['layerNum']
        self.lossFunc = nn.CrossEntropyLoss()
        self.conv = nn.Conv1d(1, param['convKernelNum'],
                                    param['convKernelSize'])
        self.convMore = nn.Conv1d(param['convKernelNum'],
                                    param['convKernelNum'],
                                    param['convKernelSize'])
        self.pool = nn.MaxPool1d(param['maxPoolKernelSize'])

        ### Trying to avoid Lazy module since it's under development ###
        ### But so far it's working just fine, so still using lazy module ###
        # flattenLength = int(featureNum / pow(maxPoolKernelSize, layerNum))
        # self.dense = nn.Linear(flattenLength, densedSize)
        ### -------------------------------------------------------- ###

        self.dense = nn.LazyLinear(param['densedSize'])
        self.decision = nn.Linear(param['densedSize'], 2)
        self.optimizer = param['optFunc'](self.parameters(),
                                            param['learningRate'])

    # Overwrite the forward function in nn.Module
    def forward(self, input):
        input = self.pool(func.relu(self.conv(input)))
        for i in range(self.layerNum - 1):
            input = self.pool(func.relu(self.convMore(input)))
        input = torch.flatten(input, start_dim = 1)
        input = func.relu(self.dense(input))
        input = func.softmax(self.decision(input), dim = 1)
        return input



class Hybrid_Model_Unlimited(nn.Module):
    """
    Defining a CNN model treating input as 2D data
    with given hyperparameters
    then using 2 1D convolution kernels to generate layers
    """
    def __init__(self, param):
        super().__init__()

        # Initialization
        self.matrixSize = param['matrixSize']
        self.layerNum = param['layerNum']
        self.lossFunc = nn.CrossEntropyLoss()
        self.pool0 = nn.MaxPool2d((1, param['maxPoolKernelSize']))
        self.pool1 = nn.MaxPool2d((param['maxPoolKernelSize'], 1))

        self.conv0 = nn.Conv2d(1, param['convKernelNum'],
                                    (self.matrixSize[0], 1))
        self.conv0More = nn.Conv2d(param['convKernelNum'],
                                    param['convKernelNum'],
                (1, int(self.matrixSize[0] / pow(param['maxPoolKernelSize'],
                                    self.layerNum))))


        self.conv1 = nn.Conv2d(1, param['convKernelNum'],
                                    (1, self.matrixSize[1]))
        self.conv1More = nn.Conv2d(param['convKernelNum'],
                                    param['convKernelNum'],
                    (int(self.matrixSize[1] / pow(param['maxPoolKernelSize'],
                                    self.layerNum)), 1))

        ### Same problem as 1D model ###
        # flattenLength = int(featureNum / pow(maxPoolKernelSize, layerNum))
        # self.dense = nn.Linear(flattenLength, densedSize)

        self.dense = nn.LazyLinear(param['densedSize'])
        self.decision = nn.Linear(param['densedSize'], 2)
        self.optimizer = param['optFunc'](self.parameters(),
                                            param['learningRate'])

    # Overwrite the forward function in nn.Module
    def forward(self, input):
        input = self.reshape(input)
        temp0 = self.pool0(func.relu(self.conv0(input)))
        for i in range(self.layerNum - 1):
            temp0 = self.pool0(func.relu(self.conv0More(temp0)))

        temp0 = torch.flatten(temp0, start_dim = 1)

        temp1 = self.pool1(func.relu(self.conv1(input)))
        for i in range(self.layerNum - 1):
            temp1 = self.pool1(func.relu(self.conv1More(temp1)))
        temp1 = torch.flatten(temp1, start_dim = 1)
        input = torch.cat((temp0, temp1), dim = 1)
        input = func.relu(self.dense(input))
        input = func.softmax(self.decision(input),dim = 1)
        return input

    # transform input(1D) into a 2D matrix
    def reshape(self, input):
        return torch.reshape(input, (input.shape[0], input.shape[1],
                                    self.matrixSize[0], self.matrixSize[1]))



class Make(classifier.Make_Template):
    """
    Analysis the performances of CNN based approaches
    with different hyperparameters
    Find the top settings to build CNN
    """

    # Perform classifier training process for given times
    # and keep given ratio of top performing classifiers
    def train(self, dataSets, keepRatio, keepThread):
        testData = self.reshapeData(dataSets.dataTest)
        testLabel = dataSets.labelTest
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Generate blank models for training
        vanillaModels = []
        for setting in self.combs:
            if setting['modelType'] == '1d':
                if setting['layerSetLimit']:
                    model = OneD_Model_Limited(setting)
                else:
                    model = OneD_Model_Unlimited(setting)
            elif setting['modelType'] == 'hybrid':
                if setting['layerSetLimit']:
                    model = Hybrid_Model_Limited(setting)
                else:
                    model = Hybrid_Model_Unlimited(setting)
            vanillaModels.append(model)

        # Try out each batch size setting
        for batchSize in self.batchSizes:
            tempModels = vanillaModels
            for ep in range(self.epochs):
                currentPos = 0
                for i in range(len(dataSets.dataTrain) - batchSize):
                    if currentPos is None: break
                    if currentPos == len(dataSets.dataTrain): break
                    elif currentPos + batchSize <= len(dataSets.dataTrain):
                        batchData = self.reshapeData(dataSets.dataTrain
                                        [currentPos : currentPos + batchSize])
                        batchLabel = torch.tensor(dataSets.labelTrain
                                        [currentPos : currentPos + batchSize])
                        currentPos += batchSize

                    elif currentPos is not None:
                        batchData = self.reshapeData(dataSets.dataTrain
                                                                [currentPos:])
                        batchLabel = torch.tensor(dataSets.labelTrain
                                                                [currentPos:])
                        currentPos = None

                    batchData = batchData.to(device)
                    batchLabel = batchLabel.to(device)
                    for model in tempModels:
                        model.to(device)
                        model.train()
                        model.optimizer.zero_grad()
                        outputs = model(batchData)
                        loss = model.lossFunc(outputs, batchLabel)
                        loss.backward()
                        model.optimizer.step()
                        # gc.collect()

            for model in tempModels:
                accuracy = self._eval(model, testData, testLabel)
                self.models.append([model, batchSize, accuracy])
                # gc.collect()

            self.models.sort(key = lambda x:x[2], reverse = True)
            self._filterModels(keepRatio, keepThread)

        # Clear data
        del tempModels
        del vanillaModels
        del testData
        del testLabel
        del dataSets
        gc.collect()

    # Generate all possible hyperparameter combination
    def _getHyperParaSets(self, params):
        # Long initialize here
        self.epochs = params[0]
        self.batchSizes = params[1]
        layerSetLimit = params[2]
        matrixSizes = params[3]
        convKernelSizes = params[4]
        # optFuncs = [optim.SGD, optim.Adam],
        optFuncs = [optim.SGD]
        params.append(optFuncs)
        result = []
        combs = list(itertools.product(*params[5:]))
        for ele in combs:
            param = {
                'layerSetLimit': layerSetLimit,
                'convKernelNum': ele[0],
                'modelType': ele[1],
                'maxPoolKernelSize': ele[2],
                'densedSize': ele[3],
                'layerNum': ele[4],
                'learningRate': ele[5],
                'optFunc': ele[6],
                'matrixSize': 0,
                'convKernelSize': 0
            }
            if param['modelType'] == 'hybrid':
                try:
                    for size in matrixSizes:
                        param['matrixSize'] = size
                        result.append(param)
                except:
                    raise classifier.Error('CNN hybrid model: No matrix setup')
            elif param['modelType'] == '1d':
                for size in convKernelSizes:
                    param['convKernelSize'] = size
                    result.append(param)
        return result

    # Evaluate  the accuracy of given model with testing data
    def _eval(self, model, testData, testLabel):
        model.eval()
        with torch.no_grad():
            outputs = model(testData)
            correct = 0
            for i in range(len(outputs)):
                if outputs[i][0] > outputs[i][1]: predict = 0
                else: predict = 1
                if predict == testLabel[i]: correct += 1
            accuracy = correct / len(testLabel)
        return accuracy

    # stop epoch when no improvement on loss
    def _earlystopping(self,):
        print('under construction')

    # Cast input data into tensor format
    # Then reshape the data in format of [#data, 1(only 1 chgannel), len(data)]
    @staticmethod
    def reshapeData(data):
        return torch.reshape(torch.tensor(data, dtype = torch.float),
                                            (len(data), 1, len(data[0])))
