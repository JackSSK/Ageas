#!/usr/bin/env python3
"""
Ageas Reborn
"""


import re
import shap
import numpy as np
import pandas as pd
from warnings import warn
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import ageas.classifier.cnn as cnn
import ageas.operator as operator



class Find:
    """
    Apply various methods to interpret correct predictions made by models
    Then, assign an importance score to every feature
    """
    def __init__(self, odysseusModels):
        super(Find, self).__init__()
        # make background example based on mean of every sample
        # ToDo:
        # Background generation may need to be revised
        # We may just use grn generated based on universal exp matrix
        bases = pd.DataFrame().append(odysseusModels.allData.mean(axis = 0),
                                                            ignore_index = True)
        self.featureImpts = self._findPaths(odysseusModels, bases)
        # Delete redundant data
        del bases
        del odysseusModels

    # Calculate importances of each feature
    def _findPaths(self, odysseusModels, bases):
        sumFeatureImpts = None
        # sumFIs = None
        for records in odysseusModels.models:
            model = records[0]
            accuracy = records[-1]
            usefullData = ''
            # Get sample index when test result consist with ground truth
            for i in range(len(records[-2])):
                if odysseusModels.allLabel[i] == records[-2][i]:
                    usefullData += str(i) + ';'
            usefullData =  list(map(int, usefullData.split(';')[:-1]))
            # usefullLabel = odysseusModels.allLabel[usefullData]
            usefullData = odysseusModels.allData.iloc[usefullData,:]
            # Switch method based on model type
            modType = str(type(model))
            # Handling SVM cases
            if re.search(r'SVM', modType):
                # Handle linear kernel SVC here
                if model.param['kernel'] == 'linear':
                    featureImpts = softmax(abs(model.clf.coef_[0]))
                # Handle other cases here
                else:
                    warn('SVM with kernel other than linear ' +
                            'could result in unacceptable running time now.')
                    warn('Pleaase consider other methods instead!')
                    warn('Skipping this SVM now')
                    continue
                    explainer = shap.KernelExplainer(model.clf.predict_proba,
                                                        data = bases,)
                    shapVals = explainer.shap_values(usefullData)
                    # Get feature importances based on shapley value
                    featureImpts = softmax(sum(np.abs(shapVals).mean(0)))
            # Hybrid CNN cases and 1D CNN cases
            elif re.search(r'Hybrid', modType) or re.search(r'1D', modType):
                # Use DeepExplainer when in limited mode
                if re.search(r'Limited', modType):
                    explainer = shap.DeepExplainer(model,
                            data = cnn.Make.reshapeData(bases.values.tolist()))
                # Use GradientExplainer when in unlimited mode
                elif re.search(r'Unlimited', modType):
                    explainer = shap.GradientExplainer(model,
                            data = cnn.Make.reshapeData(bases.values.tolist()))
                else:
                    raise operator.Error('Unrecogonized CNN model: ', modType)
                # Calculate shapley values
                shapVals = explainer.shap_values(
                            cnn.Make.reshapeData(usefullData.values.tolist()))
                # Get feature importances based on shapley value
                featureImpts = softmax(sum(np.abs(shapVals).mean(0))[0])
            # XGB's GBM cases
            elif re.search(r'XGB', modType):
                # explainer = shap.TreeExplainer(model.clf,
                #                     feature_perturbation = 'interventional',
                #                     check_additivity = False,
                #                     data = bases,)
                # shapVals = explainer.shap_values(usefullData,)
                # featureImpts = softmax(sum(np.abs(shapVals).mean(0)))
                featureImpts = softmax(model.clf.feature_importances_)
            else:
                raise operator.Error('Unrecogonized model type: ', modType)

            # Update sumFeatureImpts
            if sumFeatureImpts is None:
                sumFeatureImpts = pd.array((featureImpts * accuracy),
                                                                dtype=float)
            else: sumFeatureImpts += (featureImpts * accuracy)

        # Make feature importnace matrix
        featureImpts = pd.DataFrame()
        featureImpts.index = bases.columns
        featureImpts['importance'] = sumFeatureImpts
        featureImpts = featureImpts.sort_values('importance', ascending = False)
        return featureImpts

    # Update feature importance matrix with newer matrix
    def add(self, df):
        self.featureImpts = self.featureImpts.add(df, axis = 0, fill_value = 0
                                ).sort_values('importance', ascending = False)

    # Just to stratify feature importances to top n scale
    # need to revise this part to support stratify by value
    def stratify(self, top_GRP_amount, importance_thread):
        return self.featureImpts[:top_GRP_amount]

    # Save feature importances to given path
    def save(self, path): self.featureImpts.to_csv(path, sep='\t')
