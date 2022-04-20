#!/usr/bin/env python3
"""
This file contains classes to build RNN based classifiers

ToDo:
Implement _earlystopping

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



import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Hyper-parameters
# input_size = 784 # 28x28
sequence_length = 28
input_size = 28
batch_size = 100
num_epochs = 2

param = {
    'hidden_size' = 128,
    'num_layers' = 2,
    'learning_rate' = 0.01
}

class RNN(nn.Module):
    """
    Recurrent neural network (many-to-one)
    """
    def __init__(self, input_size, param):
        super(RNN, self).__init__()
        self.hidden_size = param['hidden_size']
        self.num_layers = param['num_layers']
        self.lstm = nn.LSTM(input_size,
                            self.hidden_size,
                            self.num_layers,
                            batch_first = True)
        self.fc = nn.Linear(self.hidden_size, 2)
        self.optimizer = torch.optim.Adam(model.parameters(),
                                            lr = param['learning_rate'])
        self.lossFunc = nn.CrossEntropyLoss()

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out



class Make(classifier.cnn.Make):
    """
    Analysis the performances of CNN based approaches
    with different hyperparameters
    Find the top settings to build CNN
    """

    # Perform classifier training process for given times
    # and keep given ratio of top performing classifiers
    def train(self, dataSets, keepRatio, keepThread):
        num_features = len(dataSets.dataTest[0])
        testData = self.reshapeData(dataSets.dataTest)
        testLabel = dataSets.labelTest
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        vanilla_models = self.__set_vanilla_models()

        # Try out each batch size setting
        for batchSize in self.batchSizes:
            tempModels = vanilla_models
            for ep in range(self.epochs):
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
        for setting in self.combs:
            if setting['modelType'] == '1d':
                if setting['layerSetLimit']:
                    model = OneD_Model_Limited(setting)
                else:
                    model = Type_1D(setting)
            elif setting['modelType'] == 'hybrid':
                if setting['layerSetLimit']:
                    model = Hybrid_Model_Limited(setting)
                else:
                    model = Type_Hybrid(setting)
            result.append(model)
        return result

    # Generate all possible hyperparameter combination
    def _getHyperParaSets(self, params):
        # Long initialize here
        self.epochs = params[0]
        self.batchSizes = params[1]
        layerSetLimit = params[2]
        matrix_size = params[3]
        conv_kernel_size = params[4]
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
                'num_layers': ele[4],
                'learning_rate': ele[5],
                'optFunc': ele[6],
                'matrixSize': 0,
                'convKernelSize': 0
            }
            if param['modelType'] == 'hybrid':
                try:
                    for size in matrix_size:
                        param['matrixSize'] = size
                        result.append(param)
                except:
                    raise classifier.Error('CNN hybrid model: No matrix setup')
            elif param['modelType'] == '1d':
                for size in conv_kernel_size:
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
                # _, predicted = torch.max(outputs.data, 1)
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()
