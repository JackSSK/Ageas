#!/usr/bin/env python3
"""
Ageas Reborn

author: jy, nkmtmsys
"""

import difflib
import itertools
from sklearn.naive_bayes import GaussianNB



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
        self.combs = self._getHyperParaSets(list(config.values()))
        self.models = []

    # Perform classifier training process for given times
    # and keep given ratio of top performing classifiers
    def train(self, dataSets, keepRatio, keepThread):
        return self

    # Generate all possible hyperparameter combination
    def _getHyperParaSets(self, params):
        result = []
        combs = list(itertools.product(*params))
        for ele in combs:
            param = {
                'ele':ele,
            }
            result.append(param)
        return result

    # Filter models based on checking accuracy (or ranking)
    def _filterModels(self, keepRatio, keepThread):
        if len(self.models) > 1:
            self.models = self.models[:int(len(self.models) * keepRatio)]
            if keepThread is not None:
                lowBound = len(self.models)
                for i in reversed(range(lowBound)):
                    accuracy = self.models[i][-1]
                    if accuracy >= keepThread:break
                    lowBound -= 1
                self.models = self.models[:lowBound]
