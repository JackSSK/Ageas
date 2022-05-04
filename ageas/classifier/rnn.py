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
        h0 = torch.zeros(self.num_layers,
                        x.size(0),
                        self.hidden_size).to(self.device)
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)
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
    def train(self, dataSets, test_split_set):
        testData = classifier.reshape_tensor(dataSets.dataTest)
        testLabel = dataSets.labelTest
        num_features = len(dataSets.dataTest[0])
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        for id in self.configs:
            model = RNN(device, num_features, self.configs[id]['config'])
            epoch = self.configs[id]['epoch']
            batch_size = self.configs[id]['batch_size']
            self._train_torch(device, epoch, batch_size, model, dataSets)
            # local test
            accuracy = self._evaluate_torch(model,
                                            testData,
                                            testLabel,
                                            test_split_set)
            self.models.append([model, id, accuracy])
