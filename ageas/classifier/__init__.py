#!/usr/bin/env python3
"""
Ageas Reborn

author: jy, nkmtmsys
"""

import numpy as np
import torch
import difflib
import itertools
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB


# Cast input data into tensor format
# Then reshape the data in format of [#data, 1(only 1 chgannel), len(data)]
def reshape_tensor(data):
	return torch.reshape(
		torch.tensor(data, dtype = torch.float),
		(len(data), 1, len(data[0]))
	)


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
	def __init__(self, id, param):
		super(Sklearn_Template, self).__init__()
		self.id = id
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
	def __init__(self, config = None, cpu_mode = False):
		super(Make_Template, self).__init__()
		self.models = list()
		self.configs = config
		self.cpu_mode = cpu_mode

	# Perform classifier training process for given times
	# and keep given ratio of top performing classifiers
	def train(self,):
		return self

	# Filter models based on checking accuracy (or ranking)
	def _performance_filter(self, clf_keep_ratio:float = None,):
		# nothing to do
		if clf_keep_ratio is None or len(self.models) <= 1:
			return

		pos = int(np.ceil(len(self.models) * clf_keep_ratio))
		# Set up thread values
		acc_thread = sorted([x.accuracy for x in self.models],reverse=True)[pos]
		auroc_thread = sorted([x.auroc for x in self.models],reverse=True)[pos]
		loss_thread = sorted([x.loss for x in self.models])[pos]
		self.models = [
			x for x in self.models if (
				x.accuracy >= acc_thread and
				x.auroc >= auroc_thread and
				x.loss <= loss_thread
			)
		]
		return

	# generalized pytorch model training process
	def _train_torch(self, epoch, batch_size, model, dataSets):
		if self.cpu_mode or not torch.cuda.is_available():
			device = torch.device('cpu')
		else:
			device = torch.device('cuda')
		for ep in range(epoch):
			index_set = DataLoader(
				dataset = range(len(dataSets.dataTrain)),
				batch_size = batch_size,
				shuffle = True
			)
			for index in index_set:
				index = index.tolist()
				data = [dataSets.dataTrain[i] for i in index]
				label = [dataSets.labelTrain[i] for i in index]
				batch_data = reshape_tensor(data).to(device)
				batch_label = torch.tensor(label).to(device)
			# pass model to device
			if torch.cuda.device_count() > 1:
				model = torch.nn.DataParallel(model)
			else:
				model.to(device)
			# set model to training model
			model.train()
			if model.optimizer is not None: model.optimizer.zero_grad()
			output = model(batch_data)
			loss = model.loss_func(output, batch_label)
			loss.backward()
			if model.optimizer is not None:
				model.optimizer.step()

	# common func to evaluate both torch based and sklearn based API models
	def _evaluate(self, output, label):
		assert len(output) == len(label)
		correct_predictions = [
			i for i in range(len(output)) if list(
				output[i]).index(max(output[i])) == label[i]
		]
		auroc = roc_auc_score(label, output[:, 1], average = None)
		loss = torch.nn.CrossEntropyLoss()(
			torch.tensor(output, dtype = torch.float),
			torch.tensor(label)
		)
		return correct_predictions, float(loss), auroc

	# Evaluate the accuracy of given model with testing data
	def _evaluate_torch(self, model, data, label, do_test = True):
		if do_test:
			model.eval()
			with torch.no_grad(): output = model(reshape_tensor(data))
			correct_predictions, loss, auroc = self._evaluate(output, label)
			return Model_Record(
				model = model,
				correct_predictions = correct_predictions,
				accuracy = len(correct_predictions) / len(label),
				auroc = auroc,
				loss = loss
			)
		else:
			return Model_Record(model = model,)

	# Evaluate the accuracy of given sklearn style model with testing data
	def _evaluate_sklearn(self, model, data, label, do_test = True):
		if do_test:
			output = model.clf.predict_proba(data)
			correct_predictions, loss, auroc = self._evaluate(output, label)
			return Model_Record(
				model = model,
				correct_predictions = correct_predictions,
				accuracy = len(correct_predictions) / len(label),
				auroc = auroc,
				loss = loss
			)
		else:
			return Model_Record(model = model,)

	# stop epoch when no improvement on loss
	def _early_stopping(self,):
		print('under construction')



class Model_Record(object):
	"""docstring for Model_Record."""

	def __init__(self,
				model = None,
				correct_predictions:list = None,
				accuracy:float = 0.0,
				auroc:float = 0.0,
				loss:float = 1.0,
				**kwargs):
		super(Model_Record, self).__init__()
		self.model = model
		self.correct_predictions = correct_predictions
		self.accuracy = accuracy
		self.auroc = auroc
		self.loss = loss
		for key in kwargs:
			setattr(self, key, kwargs[key])
