#!/usr/bin/env python3
"""
This file contains classes to build model config file

author: jy, nkmtmsys
"""


import itertools
from collections import deque
import ageas.operator as operator


class Sklearn_SVM(operator.Config_Maker_Template):
    """
    config maker for sklearn based SVMs
    """
    def __init__(self, header = None, config = None):
        self.header = header
        deq = deque()
        combs = list(itertools.product(*config.values()))
        for ele in combs:
            assert len(config.keys()) == len(ele)
            temp = {}
            for i in range(len(ele)):
                key = list(config.keys())[i]
                value = ele[i]
                temp[key] = value
            temp = self.__verify_config(temp)
            if temp is not None and temp not in deq:
                deq.appendleft(temp)
        self.configs = {self.header + str(i) : deq[i] for i in range(len(deq))}

    # verify a SVC config before adding it to set
    def __verify_config(self, query):
        if 'kernel' in query and query['kernel'] != 'poly':
            if 'degree' in query:
                query['degree'] = 0
        return query



class XGBoost_GBM(operator.Config_Maker_Template):
    """
    config maker for XGBoost based GBMs
    """
    def __init__(self, header = None, config = None):
        self.header = header
        deq = deque()
        combs = list(itertools.product(*config.values()))
        for ele in combs:
            assert len(config.keys()) == len(ele)
            temp = {}
            for i in range(len(ele)):
                key = list(config.keys())[i]
                value = ele[i]
                temp[key] = value
            temp = self.__verify_config(temp)
            if temp is not None and temp not in deq:
                deq.appendleft(temp)
        self.configs = {self.header + str(i) : deq[i] for i in range(len(deq))}

    # verify a GBM config before adding it to set
    def __verify_config(self, query):
        if 'objective' in query and query['objective'] == 'multi:softmax':
            query['num_class'] = 2
        return query



class Pytorch_CNN_Hybrid(operator.Config_Maker_Template):
    """
    config maker for XGBoost based GBMs
    """
    def __init__(self, header = None, config = None):
        self.epoch = 1
        self.batch_size = None
        self.header = header
        config = self.__resize_config(config)
        self.configs = self.__get_configs(config)

    # Generate all possible hyperparameter combination
    # Check model config file for orders of parameters
    def __get_configs(self, config):
        deq = deque()
        combs = list(itertools.product(*config.values()))
        for ele in combs:
            assert len(config.keys()) == len(ele)
            temp = {}
            for i in range(len(ele)):
                key = list(config.keys())[i]
                value = ele[i]
                temp[key] = value
            temp = self.__verify_config(temp)
            if temp is not None and temp not in deq:
                deq.appendleft(temp)
        return {self.header + str(i) : deq[i] for i in range(len(deq))}

    # this should vary with different classes
    def __resize_config(self, query):
        try:
            self.epoch = query['epoch']
            self.batch_size = query['batch_size']
            del query['epoch']
            del query['batch_size']
        except Exception as CNN_Config_Maker_Error:
            raise
        return query

    # verify a hybrid CNN config before adding it to set
    def __verify_config(self, query):
        return query


class Pytorch_CNN_1D(Pytorch_CNN_Hybrid):
    """
    config maker for XGBoost based GBMs
    """
    # verify a 1D CNN config before adding it to set
    def __verify_config(self, query):
        return query

""" For test """
if __name__ == "__main__":
    import ageas.tool.json as json
    a = json.decode("../data/config/list_config.js")
    svm = Sklearn_SVM(header = 'sklearn_svc_', config = a['SVM'])
    gbm = XGBoost_GBM(header = 'xgboost_gbm_', config = a['GBM'])
    hybrid = Pytorch_CNN_Hybrid(header = 'pytorch_cnn_hybrid_',
                            config = a['CNN_Hybrid'])
    d1 = Pytorch_CNN_1D(header = 'pytorch_cnn_1d_', config = a['CNN_1D'])
    assert d1.epoch == hybrid.epoch
    assert d1.batch_size == hybrid.batch_size
    result = {
        'SVM':svm.configs,
        'GBM':gbm.configs,
        'CNN':{
            'Epoch':d1.epoch,
            'Batch_Size':d1.batch_size,
            '1D': d1.configs,
            'Hybrid':hybrid.configs
        }
    }
    json.encode(result, 'sample_config.js')
