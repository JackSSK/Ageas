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
import torch
import itertools
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
from torch.utils.data import DataLoader
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
        self.num_layers = param['num_layers']
        self.lossFunc = nn.CrossEntropyLoss()
        # Layer set 1
        self.pool = nn.MaxPool1d(param['maxpool_kernel_size'])
        self.conv = nn.Conv1d(1, param['conv_kernel_num'],
                                    param['conv_kernel_size'])

        # Layer set 2
        self.pool1 = nn.MaxPool1d(param['maxpool_kernel_size'])
        self.conv1 = nn.Conv1d(param['conv_kernel_num'],
                                param['conv_kernel_num'],
                                param['conv_kernel_size'])

        # Layer set 3
        self.pool2 = nn.MaxPool1d(param['maxpool_kernel_size'])
        self.conv2 = nn.Conv1d(param['conv_kernel_num'],
                                param['conv_kernel_num'],
                                param['conv_kernel_size'])

        ### Trying to avoid Lazy module since it's under development ###
        ### But so far it's working just fine, so still using lazy module ###
        # flattenLength = int(featureNum / pow(maxpool_kernel_size, num_layers))
        # self.dense = nn.Linear(flattenLength, densed_size)
        ### -------------------------------------------------------- ###

        self.dense = nn.LazyLinear(param['densed_size'])
        self.decision = nn.Linear(param['densed_size'], 2)
        self.optimizer = optim.SGD(self.parameters(), param['learning_rate'])

    # Overwrite the forward function in nn.Module
    def forward(self, input):
        input = self.pool(func.relu(self.conv(input)))
        if self.num_layers > 1:
            input = self.pool1(func.relu(self.conv1(input)))
        if self.num_layers > 2:
            input = self.pool2(func.relu(self.conv2(input)))
        if self.num_layers > 3:
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
        self.matrixSize = param['matrix_size']
        self.num_layers = param['num_layers']
        self.lossFunc = nn.CrossEntropyLoss()

        # Layer set 1
        self.poolVer = nn.MaxPool2d((1, param['maxpool_kernel_size']))
        self.convVer = nn.Conv2d(1, param['conv_kernel_num'],
                                    (1, self.matrixSize[1]))
        self.poolHor = nn.MaxPool2d((param['maxpool_kernel_size'], 1))
        self.convHor = nn.Conv2d(1, param['conv_kernel_num'],
                                    (self.matrixSize[0], 1))
        self.poolVer1 = nn.MaxPool2d((1, param['maxpool_kernel_size']))
        self.convVer1 = nn.Conv2d(param['conv_kernel_num'],
                                    param['conv_kernel_num'],
                    (int(self.matrixSize[1] / pow(param['maxpool_kernel_size'],
                                    self.num_layers)), 1))
        self.poolHor1 = nn.MaxPool2d((param['maxpool_kernel_size'], 1))
        self.convHor1 = nn.Conv2d(param['conv_kernel_num'],
                                    param['conv_kernel_num'],
                (1, int(self.matrixSize[0] / pow(param['maxpool_kernel_size'],
                                    self.num_layers))))

        # Layer set 3
        self.poolVer2 = nn.MaxPool2d((1, param['maxpool_kernel_size']))
        self.convVer2 = nn.Conv2d(param['conv_kernel_num'],
                                    param['conv_kernel_num'],
                    (int(self.matrixSize[1] / pow(param['maxpool_kernel_size'],
                                    self.num_layers)), 1))
        self.poolHor2 = nn.MaxPool2d((param['maxpool_kernel_size'], 1))
        self.convHor2 = nn.Conv2d(param['conv_kernel_num'],
                                    param['conv_kernel_num'],
                (1, int(self.matrixSize[0] / pow(param['maxpool_kernel_size'],
                                    self.num_layers))))


        ### Same problem as 1D model ###
        # flattenLength = int(featureNum / pow(maxpool_kernel_size, num_layers))
        # self.dense = nn.Linear(flattenLength, densed_size)
        self.dense = nn.LazyLinear(param['densed_size'])
        self.decision = nn.Linear(param['densed_size'], 2)
        self.optimizer = optim.SGD(self.parameters(), param['learning_rate'])

    # Overwrite the forward function in nn.Module
    def forward(self, input):
        input = self.reshape(input)
        temp0 = self.poolVer(func.relu(self.convHor(input)))
        temp1 = self.poolHor(func.relu(self.convVer(input)))
        if self.num_layers > 1:
            temp0 = self.poolVer1(func.relu(self.convHor1(temp0)))
            temp1 = self.poolHor1(func.relu(self.convVer1(temp1)))
        if self.num_layers > 2:
            temp0 = self.poolVer2(func.relu(self.convHor2(temp0)))
            temp1 = self.poolHor2(func.relu(self.convVer2(temp1)))
        if self.num_layers > 3:
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
        self.num_layers = param['num_layers']
        self.lossFunc = nn.CrossEntropyLoss()
        self.conv = nn.Conv1d(1, param['conv_kernel_num'],
                                    param['conv_kernel_size'])
        self.convMore = nn.Conv1d(param['conv_kernel_num'],
                                    param['conv_kernel_num'],
                                    param['conv_kernel_size'])
        self.pool = nn.MaxPool1d(param['maxpool_kernel_size'])

        ### Trying to avoid Lazy module since it's under development ###
        ### But so far it's working just fine, so still using lazy module ###
        # flattenLength = int(featureNum / pow(maxpool_kernel_size, num_layers))
        # self.dense = nn.Linear(flattenLength, densed_size)
        ### -------------------------------------------------------- ###

        self.dense = nn.LazyLinear(param['densed_size'])
        self.decision = nn.Linear(param['densed_size'], 2)
        self.optimizer = optim.SGD(self.parameters(), param['learning_rate'])

    # Overwrite the forward function in nn.Module
    def forward(self, input):
        input = self.pool(func.relu(self.conv(input)))
        for i in range(self.num_layers - 1):
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
        self.matrixSize = param['matrix_size']
        self.num_layers = param['num_layers']
        self.lossFunc = nn.CrossEntropyLoss()
        self.pool0 = nn.MaxPool2d((1, param['maxpool_kernel_size']))
        self.pool1 = nn.MaxPool2d((param['maxpool_kernel_size'], 1))

        self.conv0 = nn.Conv2d(1, param['conv_kernel_num'],
                                    (self.matrixSize[0], 1))
        self.conv0More = nn.Conv2d(param['conv_kernel_num'],
                                    param['conv_kernel_num'],
                (1, int(self.matrixSize[0] / pow(param['maxpool_kernel_size'],
                                    self.num_layers))))


        self.conv1 = nn.Conv2d(1, param['conv_kernel_num'],
                                    (1, self.matrixSize[1]))
        self.conv1More = nn.Conv2d(param['conv_kernel_num'],
                                    param['conv_kernel_num'],
                    (int(self.matrixSize[1] / pow(param['maxpool_kernel_size'],
                                    self.num_layers)), 1))

        ### Same problem as 1D model ###
        # flattenLength = int(featureNum / pow(maxpool_kernel_size, num_layers))
        # self.dense = nn.Linear(flattenLength, densed_size)

        self.dense = nn.LazyLinear(param['densed_size'])
        self.decision = nn.Linear(param['densed_size'], 2)
        self.optimizer = optim.SGD(self.parameters(), param['learning_rate'])

    # Overwrite the forward function in nn.Module
    def forward(self, input):
        input = self.reshape(input)
        temp0 = self.pool0(func.relu(self.conv0(input)))
        for i in range(self.num_layers - 1):
            temp0 = self.pool0(func.relu(self.conv0More(temp0)))

        temp0 = torch.flatten(temp0, start_dim = 1)

        temp1 = self.pool1(func.relu(self.conv1(input)))
        for i in range(self.num_layers - 1):
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
        vanilla_models = self.__set_vanilla_models()

        # Try out each batch size setting
        for batchSize in self.configs['Batch_Size']:
            tempModels = vanilla_models
            for ep in range(self.configs['Epoch']):
                index_set = DataLoader(dataset = range(len(dataSets.dataTrain)),
                                        batch_size = batchSize,
                                        shuffle = True)
                for index in index_set:
                    index = index.tolist()
                    data = [dataSets.dataTrain[i] for i in index]
                    label = [dataSets.labelTrain[i] for i in index]
                    batchData = self.reshapeData(data).to(device)
                    batchLabel = torch.tensor(label).to(device)
                    for model in tempModels:
                        model.to(device)
                        model.train()
                        model.optimizer.zero_grad()
                        outputs = model(batchData)
                        loss = model.lossFunc(outputs, batchLabel)
                        loss.backward()
                        model.optimizer.step()

            for model in tempModels:
                accuracy = self._eval(model, testData, testLabel)
                self.models.append([model, batchSize, accuracy])

            self.models.sort(key = lambda x:x[2], reverse = True)
            self._filterModels(keepRatio, keepThread)

        # Clear data
        del tempModels
        del vanilla_models
        del testData
        del testLabel
        del dataSets

    # Generate blank models for training
    def __set_vanilla_models(self,):
        result = []
        if '1D' in self.configs:
            for id in self.configs['1D']:
                if self.configs['Num_Layers_Limit']:
                    model = OneD_Model_Limited(self.configs['1D'][id])
                else:
                    model = OneD_Model_Unlimited(self.configs['1D'][id])
                result.append(model)
        if 'Hybrid' in self.configs:
            for id in self.configs['Hybrid']:
                if self.configs['Num_Layers_Limit']:
                    model = Hybrid_Model_Limited(self.configs['Hybrid'][id])
                else:
                    model = Hybrid_Model_Unlimited(self.configs['Hybrid'][id])
                result.append(model)
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
