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


# Hyper-parameters
# input_size = 784 # 28x28

# Fully connected neural network with one hidden layer
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # -> x needs to be: (batch_size, seq, input_size)

        # or:
        #self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        #self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        #c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # x: (n, 28, 28), h0: (2, n, 128)

        # Forward propagate RNN
        out, _ = self.rnn(x, h0)
        # or:
        #out, _ = self.lstm(x, (h0,c0))

        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (n, 28, 128)

        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        # out: (n, 128)

        out = self.fc(out)
        # out: (n, 10)
        return out



class LSTM(nn.Module):
    """
    Recurrent neural network (many-to-one)
    """
    def __init__(self, input_size, param):
        super(LSTM, self).__init__()
        self.hidden_size = param['hidden_size']
        self.num_layers = param['num_layers']
        self.lstm = nn.LSTM(input_size,
                            self.hidden_size,
                            self.num_layers,
                            batch_first = True)
        self.fc = nn.Linear(self.hidden_size, 2)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                            lr = param['learning_rate'])
        self.lossFunc = nn.CrossEntropyLoss()

    def forward(self, x):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(device)
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
        vanilla_models = self.__set_vanilla_models(num_features)

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
    def __set_vanilla_models(self, num_features):
        result = []
        if 'LSTM' in self.configs:
            for id in self.configs['LSTM']:
                model = LSTM(num_features, self.configs['LSTM'][id])
                result.append(model)
        if 'GRU' in self.configs:
            for id in self.configs['GRU']:
                model = None
                result.append(model)
        return result
