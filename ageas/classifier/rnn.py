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



class RNN(nn.Module):
    def __init__(self, device, input_size, param):
        super(RNN, self).__init__()
        self.device = device
        self.num_layers = param['num_layers']
        self.hidden_size = param['hidden_size']
        self.rnn = nn.RNN(input_size,
                            self.hidden_size,
                            self.num_layers,
                            batch_first=True)
        self.fc = nn.Linear(self.hidden_size, 2)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                            lr = param['learning_rate'])
        self.lossFunc = nn.CrossEntropyLoss()

    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        # -> x needs to be: (batch_size, seq, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out



class LSTM(nn.Module):
    """
    Recurrent neural network (many-to-one)
    """
    def __init__(self, device, input_size, param):
        super(LSTM, self).__init__()
        self.device = device
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
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(self.device)
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out



class GRU(nn.Module):
    def __init__(self, device, input_size, param):
        super(GRU, self).__init__()
        self.device = device
        self.num_layers = param['num_layers']
        self.hidden_size = param['hidden_size']
        self.gru = nn.GRU(input_size,
                            self.hidden_size,
                            self.num_layers,
                            batch_first=True)
        self.fc = nn.Linear(self.hidden_size, 2)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                            lr = param['learning_rate'])
        self.lossFunc = nn.CrossEntropyLoss()

    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        # -> x needs to be: (batch_size, seq, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        # Forward propagate RNN
        out, _ = self.gru(x, h0)
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
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if 'RNN' in self.configs:
            for id in self.configs['RNN']:
                model = RNN(device, num_features, self.configs['RNN'][id])
                result.append(model)
        if 'LSTM' in self.configs:
            for id in self.configs['LSTM']:
                model = LSTM(device, num_features, self.configs['LSTM'][id])
                result.append(model)
        if 'GRU' in self.configs:
            for id in self.configs['GRU']:
                model = GRU(device, num_features, self.configs['GRU'][id])
                result.append(model)
        return result
