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



class GRU(nn.Module):
    def __init__(self,
                id,
                input_size,
                num_layers,
                hidden_size,
                dropout,
                learning_rate,
                n_class = 2):
        super(GRU, self).__init__()
        self.id = id
        self.model_type = 'GRU'
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p = dropout)
        self.gru = nn.GRU(input_size,
                            self.hidden_size,
                            self.num_layers,
                            batch_first=True)
        self.fc = nn.Linear(self.hidden_size, n_class)
        self.optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, input):
        input = self.dropout(input)
        # Set initial hidden states (and cell states for LSTM)
        # input needs to be: (batch_size, seq, input_size)
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size)
        out, _ = self.gru(input, h0)
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
        for id in self.configs:
            model = GRU(id, num_features, **self.configs[id]['config'])
            epoch = self.configs[id]['epoch']
            batch_size = self.configs[id]['batch_size']
            self._train_torch(epoch, batch_size, model, dataSets)
            accuracy = self._evaluate_torch(model,
                                            testData,
                                            testLabel,
                                            test_split_set)
            self.models.append([model, accuracy])
