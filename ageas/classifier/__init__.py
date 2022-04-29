#!/usr/bin/env python3
"""
Ageas Reborn

author: jy, nkmtmsys
"""

import torch
import difflib
import itertools
from torch.utils.data import DataLoader
from sklearn.naive_bayes import GaussianNB


# Cast input data into tensor format
# Then reshape the data in format of [#data, 1(only 1 chgannel), len(data)]
def reshape_tensor(data):
    return torch.reshape(torch.tensor(data, dtype = torch.float),
                                        (len(data), 1, len(data[0])))


class Error(Exception):
    """
    Classifier related error handling
    """
    pass



class Sklearn_Template:
    """
    Build up sklearn-style general classifier based on given parameters
    Gaussian Naive Bayes is used as example here
    """
    def __init__(self, param):
        super(Sklearn_Template, self).__init__()
        self.param = param
        self.initial()

    def train(self,  dataTrain = None, labelTrain = None):
        self.clf.fit(dataTrain, labelTrain)

    # Set clf to default level
    def initial(self): self.clf = GaussianNB(**param)



class Make_Template:
    """
    Analysis the performances of models with different hyperparameters
    Find the top settings to build models
    """
    def __init__(self, config):
        super(Make_Template, self).__init__()
        self.configs = config
        self.models = []

    # Perform classifier training process for given times
    # and keep given ratio of top performing classifiers
    def train(self, dataSets, keepRatio, keepThread):
        return self

    # Filter models based on checking accuracy (or ranking)
    def _filter_models(self, keepRatio, keepThread):
        if len(self.models) > 1:
            if keepRatio is not None:
                self.models = self.models[:int(len(self.models) * keepRatio)]
            if keepThread is not None:
                lowBound = len(self.models)
                for i in reversed(range(lowBound)):
                    accuracy = self.models[i][-1]
                    if accuracy >= keepThread:break
                    lowBound -= 1
                self.models = self.models[:lowBound]

    # generalized pytorch model training process
    def _train_torch(self, dataSets, keepRatio, keepThread, vanilla_models):
        testData = reshape_tensor(dataSets.dataTest)
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
                    batchData = reshape_tensor(data).to(device)
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
                accuracy = self.__evaluate_torch(model, testData, testLabel)
                self.models.append([model, id, batchSize, accuracy])

            self.models.sort(key = lambda x:x[-1], reverse = True)
            self._filter_models(keepRatio, keepThread)

        # Clear data
        del tempModels
        del vanilla_models
        del testData
        del testLabel
        del dataSets

    # Evaluate  the accuracy of given model with testing data
    def __evaluate_torch(self, model, testData, testLabel):
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
    def _early_stopping(self,):
        print('under construction')
