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



class Limited(nn.Module):
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


class Unlimited(nn.Module):
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



class Make(classifier.Make_Template):
    """
    Analysis the performances of CNN based approaches
    with different hyperparameters
    Find the top settings to build CNN
    """

    # Perform classifier training process for given times
    # and keep given ratio of top performing classifiers
    def train(self, dataSets, keepRatio, keepThread):
        vanilla_models = self.__set_vanilla_models()
        self.__train_process(dataSets, keepRatio, keepThread, vanilla_models)

    # generalized training process
    def __train_process(self, dataSets, keepRatio, keepThread, vanilla_models):
        testData = classifier.reshape_tensor(dataSets.dataTest)
        testLabel = dataSets.labelTest
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
                    batchData = classifier.reshape_tensor(data).to(device)
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
        for id in self.configs['Config']:
            if self.configs['Config'][id]['num_layers'] < 3:
                model = Limited(self.configs['Config'][id])
            else:
                model = Unlimited(self.configs['Config'][id])
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
