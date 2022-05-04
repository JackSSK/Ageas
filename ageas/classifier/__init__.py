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
    def __init__(self, config = None):
        super(Make_Template, self).__init__()
        self.configs = config
        self.models = []

    # Perform classifier training process for given times
    # and keep given ratio of top performing classifiers
    def train(self, dataSets, clf_keep_ratio, clf_accuracy_thread):
        return self

    # Filter models based on checking accuracy (or ranking)
    def _filter_models(self, clf_keep_ratio, clf_accuracy_thread):
        # nothing to do
        if ((clf_keep_ratio is None and clf_accuracy_thread is None)
            or (len(self.models) <= 1)): return
        # Or we do the job
        accuracy_thread = None
        self.models.sort(key = lambda x:x[-1], reverse = True)
        if clf_keep_ratio is not None:
            index = int(len(self.models) * clf_keep_ratio) - 1
            accuracy_thread = self.models[index][-1]
        if clf_accuracy_thread is not None:
            if accuracy_thread is None:
                accuracy_thread = clf_accuracy_thread
            else:
                accuracy_thread = max(accuracy_thread, clf_accuracy_thread)
        # now we partition
        low_bound = len(self.models)
        for i in reversed(range(low_bound)):
            accuracy = self.models[i][-1]
            if accuracy >= accuracy_thread:break
            low_bound -= 1
        self.models = self.models[:low_bound]
        return

    # generalized pytorch model training process
    def _train_torch(self, device, epoch, batch_size, model, dataSets):
        for ep in range(epoch):
            index_set = DataLoader(dataset = range(len(dataSets.dataTrain)),
                                    batch_size = batch_size,
                                    shuffle = True)
            for index in index_set:
                index = index.tolist()
                data = [dataSets.dataTrain[i] for i in index]
                label = [dataSets.labelTrain[i] for i in index]
                batch_data = reshape_tensor(data).to(device)
                batch_label = torch.tensor(label).to(device)
            model.to(device)
            model.train()
            model.optimizer.zero_grad()
            outputs = model(batch_data)
            loss = model.lossFunc(outputs, batch_label)
            loss.backward()
            model.optimizer.step()

    # Evaluate the accuracy of given model with testing data
    def _evaluate_torch(self, model, testData, testLabel, do_test):
        accuracy = None
        if do_test:
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

    # Evaluate the accuracy of given sklearn style model with testing data
    def _evaluate_sklearn(self, model, testData, testLabel, do_test):
        accuracy = None
        if do_test:
            accuracy = difflib.SequenceMatcher(None,
                                                model.clf.predict(testData),
                                                testLabel).ratio()
        return accuracy

    # stop epoch when no improvement on loss
    def _early_stopping(self,):
        print('under construction')
