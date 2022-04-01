#!/usr/bin/env python3
"""
Ageas Reborn
XGBoost Gradient Boosting based classifier related classes and functions

author: jy, nkmtmsys
"""

import difflib
import itertools
import ageas.classifier as classifier
from xgboost import XGBClassifier



class XGB(classifier.Sklearn_Template):
    """
    Build up XGB classifier based on given parameters
    """
    # Set clf to default level
    # Turned off label encoder as official doc recommended
    def initial(self):
        self.clf = XGBClassifier(**self.param, use_label_encoder=False)



class Make(classifier.Make_Template):
    """
    Analysis the performances of XGB based approaches
    with different hyperparameters
    Find the top settings to build XGB
    """
    # Perform classifier training process for given times
    # and keep given ratio of top performing classifiers
    def train(self, dataSets, keepRatio, keepThread):
        for setting in self.combs:
            # Initialize XGB model
            candidate = XGB(setting)
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
        combs = list(itertools.product(*params))
        for ele in combs:
            param = {
                'booster':ele[0],
                'objective':ele[1],
                'eval_metric':ele[2],
                'eta':ele[3],
                'gamma':ele[4],
                'max_depth':ele[5],
                'min_child_weight':ele[6],
                'alpha':ele[7],
            }
            if ele[1] == 'multi:softmax':
                param['num_class'] = 2
                result.append(param)
            else: result.append(param)
        return result
