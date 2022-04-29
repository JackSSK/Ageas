#!/usr/bin/env python3
"""
This file contains classes to build RNN based classifiers

ToDo:
Implement _earlystopping

author: jy, nkmtmsys
"""

import torch
import torch.nn as nn
import torch.optim as optim
import ageas.classifier as classifier



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
        h0 = torch.zeros(self.num_layers,
                        x.size(0),
                        self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers,
                        x.size(0),
                        self.hidden_size).to(self.device)
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out



class Make(classifier.Make_Template):
    """
    Analysis the performances of CNN based approaches
    with different hyperparameters
    Find the top settings to build CNN
    """

    # Perform classifier training process for given times
    # and keep given ratio of top performing classifiers
    def train(self, dataSets, keepRatio, keepThread):
        num_features = len(dataSets.dataTest[0])
        vanilla_models = self.__set_vanilla_models(configs = self.configs,
                                                    num_features = num_features)
        self._train_torch(dataSets, keepRatio, keepThread, vanilla_models)

    # Generate blank models for training
    def __set_vanilla_models(self, configs, num_features):
        result = []
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        for id in configs['Config']:
            model = LSTM(device, num_features, configs['Config'][id])
            result.append(model)
        return result
