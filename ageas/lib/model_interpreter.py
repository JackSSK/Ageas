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
from ageas.classifier import reshape_tensor
import ageas.lib as lib


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
        shap_explainer = SHAP_Explainer(bases)
        sumFeatureImpts = None
        # sumFIs = None
        for records in odysseusModels.models:
            # get model and data to explain
            model = records[0]
            accuracy = records[-1]
            print('     Interpreting:',model.id,' which reached ACC:',accuracy)
            usefullData = ''
            # Get sample index when test result consist with ground truth
            for i in range(len(records[-2])):
                if odysseusModels.allLabel[i] == records[-2][i]:
                    usefullData += str(i) + ';'
            usefullData =  list(map(int, usefullData.split(';')[:-1]))
            # usefullLabel = odysseusModels.allLabel[usefullData]
            usefullData = odysseusModels.allData.iloc[usefullData,:]

            # Handling RFC cases
            if model.model_type == 'RFC':
                featureImpts = shap_explainer.get_tree_explain(
                                                        model.clf,
                                                        usefullData)

            # Handling GNB cases
            elif model.model_type == 'GNB':
                featureImpts = shap_explainer.get_kernel_explain(
                                                        model.clf.predict_proba,
                                                        usefullData)

            # Handling SVM cases
            elif model.model_type == 'SVM':
                # Handle linear kernel SVC here
                if model.param['kernel'] == 'linear':
                    featureImpts = softmax(abs(model.clf.coef_[0]))
                # Handle other cases here
                else:
                    featureImpts = shap_explainer.get_kernel_explain(
                                                        model.clf.predict_proba,
                                                        usefullData)
            # Hybrid CNN cases and 1D CNN cases
            elif re.search(r'CNN', model.model_type):
                # Use DeepExplainer when in limited mode
                if re.search(r'Limited', model.model_type):
                    featureImpts = shap_explainer.get_deep_explain(model,
                                                                    usefullData)
                # Use GradientExplainer when in unlimited mode
                elif re.search(r'Unlimited', model.model_type):
                    featureImpts = shap_explainer.get_gradient_explain(model,
                                                                    usefullData)
                else:
                    raise lib.Error('Unrecogonized CNN model:',model.model_type)
            # XGB's GBM cases
            elif model.model_type == 'XGB_GBM':
                featureImpts = softmax(model.clf.feature_importances_)
            # RNN_base model cases
            elif (model.model_type == 'RNN' or
                    model.model_type == 'LSTM' or
                    model.model_type == 'GRU'):
                featureImpts = shap_explainer.get_gradient_explain(model,
                                                                    usefullData)
            else:
                raise lib.Error('Unrecogonized model type: ', model.model_type)

            # Update sumFeatureImpts
            if sumFeatureImpts is None and featureImpts is not None:
                sumFeatureImpts = pd.array((featureImpts*accuracy) ,dtype=float)
            elif featureImpts is not None:
                sumFeatureImpts += (featureImpts * accuracy)

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



class SHAP_Explainer(object):
    """docstring for SHAP_Explainer."""

    def __init__(self, basement_data = None):
        super(SHAP_Explainer, self).__init__()
        self.basement_data = basement_data

    # Use KernelExplainer
    def get_kernel_explain(self, model, data: pd.DataFrame):
        print('Kernel Explainer is too slow! Skipping now!')
        return None
        # explainer = shap.KernelExplainer(model, data = self.basement_data,)
        # shap_vals = explainer.shap_values(data)
        # return softmax(sum(np.abs(shap_vals).mean(0)))

    # Use GradientExplainer
    def get_gradient_explain(self, model, data: pd.DataFrame):
        # reshape basement data to tensor type
        base = reshape_tensor(self.basement_data.values.tolist())
        explainer = shap.GradientExplainer(model, data = base)
        # Calculate shapley values
        shap_vals = explainer.shap_values(reshape_tensor(data.values.tolist()))
        # Get feature importances based on shapley value
        return softmax(sum(np.abs(shap_vals).mean(0))[0])

    # Use DeepExplainer
    def get_deep_explain(self, model, data: pd.DataFrame):
        # reshape basement data to tensor type
        base = reshape_tensor(self.basement_data.values.tolist())
        explainer = shap.DeepExplainer(model, data = base)
        # Calculate shapley values
        shap_vals = explainer.shap_values(reshape_tensor(data.values.tolist()))
        # Get feature importances based on shapley value
        return softmax(sum(np.abs(shap_vals).mean(0))[0])

    # Use TreeExplainer
    def get_tree_explain(self, model, data: pd.DataFrame):
        explainer = shap.TreeExplainer(model,
                                        feature_perturbation = 'interventional',
                                        check_additivity = False,
                                        data = self.basement_data,)
        shap_vals = explainer.shap_values(data)
        return softmax(sum(np.abs(shap_vals).mean(0)))
