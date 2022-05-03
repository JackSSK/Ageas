#!/usr/bin/env python3
"""
Ageas Reborn
XGBoost Gradient Boosting based classifier related classes and functions

author: jy, nkmtmsys
"""

import difflib
import itertools
from xgboost import XGBClassifier
import ageas.classifier as classifier


class XGB(classifier.Sklearn_Template):
    """
    Build up XGB classifier based on given parameters
    """
    # Set clf to default level
    # Turned off label encoder as official doc recommended
    def initial(self):
        self.clf = XGBClassifier(**self.param, use_label_encoder = False)



class Make(classifier.Make_Template):
    """
    Analysis the performances of XGB based approaches
    with different hyperparameters
    Find the top settings to build XGB
    """
    # Perform classifier training process for given times
    def train(self, dataSets):
        for id in self.configs:
            # Initialize XGB model
            candidate = XGB(self.configs[id]['config'])
            candidate.train(dataSets.dataTrain, dataSets.labelTrain)
            # Check performance
            testResult = candidate.clf.predict(dataSets.dataTest)
            accuracy = difflib.SequenceMatcher(None,
                                                testResult,
                                                dataSets.labelTest).ratio()
            self.models.append([candidate, id, accuracy])
