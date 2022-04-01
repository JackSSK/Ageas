#!/usr/bin/env python3
"""
Ageas Reborn
Scikit-Learn Support Vector Machine based classifier
related classes and functions

author: jy, nkmtmsys
"""


import difflib
import itertools
import ageas.classifier as classifier
from sklearn import svm



class SVM(classifier.Sklearn_Template):
    """
    Build up SVM classifier based on given parameters
    Note:
    kerenl = {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    """
    # Set clf to default level
    def initial(self): self.clf = svm.SVC(**self.param)



class Make(classifier.Make_Template):
    """
    Analysis the performances of SVM based approaches
    with different hyperparameters
    Find the top settings to build SVM
    """

    # Perform classifier training process for given times
    # and keep given ratio of top performing classifiers
    def train(self, dataSets, keepRatio, keepThread):
        for setting in self.combs:
            # Initialize SVM model
            candidate = SVM(setting)
            candidate.train(dataSets.dataTrain, dataSets.labelTrain)
            # Check performance
            testResult = candidate.clf.predict(dataSets.dataTest)
            accuracy = difflib.SequenceMatcher(None,
                testResult, dataSets.labelTest).ratio()
            self.models.append([candidate, accuracy])
        self.models.sort(key = lambda x:x[1], reverse = True)
        self._filterModels(keepRatio, keepThread)

    # Generate all possible hyperparameter combination
    # Check model config file for orders of parameters
    def _getHyperParaSets(self, params):
        result = []
        combs = list(itertools.product(*params[:3]))
        for ele in combs:
            param = {
                'kernel': ele[0],
                'gamma': ele[1],
                'C': ele[2],
                'degree': 0,
                'cache_size': 500,
                'probability': True
            }
            if ele[0] == 'poly':
                for value in params[3]:
                    param['degree'] = value
                    result.append(param)
            else: result.append(param)
        return result
