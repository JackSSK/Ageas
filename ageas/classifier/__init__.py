#!/usr/bin/env python3
"""
Ageas Reborn

author: jy, nkmtmsys
"""

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
	def train(self, dataSets, clf_keep_ratio, clf_accuracy_thread):
		return self

	# Filter models based on checking accuracy (or ranking)
	def _filter_models(self,
					  auroc_thread:float = 0.8,
					  clf_keep_ratio:float = None,
					  clf_accuracy_thread:float = None):
		# filter models with low AUROC values
		self.models = [x for x in self.models if x.auroc >= auroc_thread]

		# nothing to do
		if ((clf_keep_ratio is None and clf_accuracy_thread is None)
			or (len(self.models) <= 1)): return

		# Or we do the job
		accuracy_thread = None
		self.models.sort(key = lambda x:x.accuracy, reverse = True)
		if clf_keep_ratio is not None:
			index = int(len(self.models) * clf_keep_ratio) - 1
			accuracy_thread = self.models[index].accuracy
		if clf_accuracy_thread is not None:
			if accuracy_thread is None:
				accuracy_thread = clf_accuracy_thread
			else:
				accuracy_thread = min(accuracy_thread, clf_accuracy_thread)
		# now we partition
		if accuracy_thread > self.models[0].accuracy:
			print('accuracy_thread is too high! Returning the best we can get')
			accuracy_thread = self.models[0].accuracy
		print('accuracy_thread is set to:', accuracy_thread)
		low_bound = len(self.models)
		for i in reversed(range(low_bound)):
			accuracy = self.models[i].accuracy
			if accuracy >= accuracy_thread:break
			low_bound -= 1
		self.models = self.models[:low_bound]
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
			if model.optimizer is not None: model.optimizer.step()

	# common func to evaluate both torch based and sklearn based API models
	def _evaluate(self, output, label):
		assert len(output) == len(label)
		correct_predictions = []
		for i in range(len(output)):
			if output[i][0] > output[i][1] and label[i] == 0:
				correct_predictions.append(i)
			elif output[i][0] < output[i][1] and label[i] == 1:
				correct_predictions.append(i)
		auroc = roc_auc_score(label, output[:, 1], average = None)
		accuracy = len(correct_predictions) / len(label)
		return correct_predictions, accuracy, auroc

	# Evaluate the accuracy of given model with testing data
	def _evaluate_torch(self, model, data, label, do_test = True):
		if do_test:
			model.eval()
			with torch.no_grad(): output = model(reshape_tensor(data))
			correct_predictions, accuracy, auroc = self._evaluate(output, label)
			return Model_Record(
				model = model,
				correct_predictions = correct_predictions,
				accuracy = accuracy,
				auroc = auroc
			)
		else:
			return Model_Record(model = model,)

	# Evaluate the accuracy of given sklearn style model with testing data
	def _evaluate_sklearn(self, model, data, label, do_test = True):
		if do_test:
			output = model.clf.predict_proba(data)
			correct_predictions, accuracy, auroc = self._evaluate(output, label)
			return Model_Record(
				model = model,
				correct_predictions = correct_predictions,
				accuracy = accuracy,
				auroc = auroc
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
				**kwargs):
		super(Model_Record, self).__init__()
		self.model = model
		self.correct_predictions = correct_predictions
		self.accuracy = accuracy
		self.auroc = auroc
		for key in kwargs:
			setattr(self, key, kwargs[key])
