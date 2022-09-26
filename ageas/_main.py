#!/usr/bin/env python3
"""
Ageas Reborn

author: jy, nkmtmsys
"""

import os
import sys
import copy
import time
import threading
import warnings
from pkg_resources import resource_filename
import ageas
import ageas.tool.json as json
import ageas.lib.meta_grn_caster as meta_grn
import ageas.lib.config_maker as config_maker
import ageas.lib.atlas_extractor as extractor



GRP_TYPES = ['Standard', 'Outer', 'Bridge', 'Mix']



class Launch:
	"""
	Object containing basic pipeline to launch AGEAS.

	Results are stored in attributes and can be saved as files.
	"""

	def __init__(self,
				 model_config_path:str = None,
				 mute_unit:bool = True,
				 protocol:str = 'solo',
				 unit_num:int = 1,
				 warning_filter:str = 'ignore',
				 correlation_thread:float = 0.2,
				 database_path:str = None,
				 database_type:str = 'gem_files',
				 class1_path:str = None,
				 class2_path:str = None,
				 interaction_database:str = 'gtrd',
				 log2fc_thread:float = None,
				 meta_load_path:str = None,
				 mww_p_val_thread:float = 0.05,
				 normalize:str = None,
				 prediction_thread = 'auto',
				 psgrn_load_path:str = None,
				 specie:str = 'mouse',
				 sliding_window_size:int = 100,
				 sliding_window_stride:int = None,
				 std_value_thread:float = 1.0,
				 std_ratio_thread:float = None,
				 clf_keep_ratio:float = 0.5,
				 cpu_mode:bool = False,
				 feature_dropout_ratio:float = 0.1,
				 feature_select_iteration:int = 3,
				 top_grp_amount:int = 100,
				 grp_changing_thread:float = 0.05,
				 model_select_iteration:int = 3,
				 outlier_thread:float = 3.0,
				 regulatory_trace_depth:int = 1,
				 stabilize_patient:int = 3,
				 stabilize_iteration:int = 10,
				 max_train_size:float = 0.95,
				 z_score_extract_thread:float = 0.0,
				):
		"""
		Pipeline to launch AGEAS.

		:param model_config_path: <str Default = None>
			Path to load model config file which will be used to initialize
			classifiers.

			By default, AGEAS will use internalized model config file which
			contians following model types:

				Transformer

				Random Forest(RFC)

				Support Vector Machine(SVM)

				Gradient Boosting Machine(GBM)

				Convolutional Neural Network(CNN_1D, CNN_Hybrid)

				Recurrent Neural Network(RNN)

				Long Short Term Memory(LSTM)

				Gated Recurrent Unit(GRU)

		:param mute_unit: <bool Default = True>
			Whether AGEAS unit print out log while running.
			It is not mandatory but encouraged to remain True especially when
			using 'multi' protocol.

		:param protocol: <str Default = 'solo'>
			AGEAS unit launching protocol.

			Supporting:

				'solo': All units will run separately.

				'multi': All units will run parallelly by multithreading.

		:param unit_num: <int Default = 1>
			Amount of AGEAS extractor units to launch.

		:param warning_filter: <str Default = 'ignore'>
			How warnings should be filtered. For other options,
			please check 'The Warnings Filter' section in:
			https://docs.python.org/3/library/warnings.html#warning-filter

		Additional Parameters:
			All args in **ageas.Data_Preprocess()**

			All args in **ageas.Unit()** excluding database_info, meta,
			model_config, and pseudo_grns,


		Attributes:
			self.atlas

			self.meta

			self.pseudo_grns
		"""
		super(Launch, self).__init__()

		""" Initialization """
		print('Launching Ageas')
		warnings.filterwarnings(warning_filter)
		start = time.time()
		self._reports = list()
		self._unit_num = unit_num
		self._silent = mute_unit

		# Get model configs
		if model_config_path is None:
			path = resource_filename(__name__, 'data/config/list_config.js')
			self._model_config = config_maker.List_Config_Reader(path)
		else:
			self._model_config = json.decode(model_config_path)
		print('Time to Boot: ', time.time() - start)

		# integrate database info
		# and make meta GRN, pseudo samples if not loaded
		self.database_info, self.meta, self.pseudo_grns = ageas.Data_Preprocess(
			correlation_thread = correlation_thread,
			database_path = database_path,
			database_type = database_type,
			class1_path = class1_path,
			class2_path = class2_path,
			interaction_database = interaction_database,
			log2fc_thread = log2fc_thread,
			meta_load_path = meta_load_path,
			mww_p_val_thread = mww_p_val_thread,
			normalize = normalize,
			prediction_thread = prediction_thread,
			psgrn_load_path = psgrn_load_path,
			specie = specie,
			sliding_window_size = sliding_window_size,
			sliding_window_stride = sliding_window_stride,
			std_value_thread = std_value_thread,
			std_ratio_thread = std_ratio_thread,
		)

		print('\nDeck Ready')

		start = time.time()
		# Initialize a basic unit
		self._basic_unit = ageas.Unit(
			# Args already processed
			database_info = self.database_info,
			meta = self.meta,
			model_config = self._model_config,
			pseudo_grns = self.pseudo_grns,

			# raw parameters
			clf_keep_ratio = clf_keep_ratio,
			correlation_thread = correlation_thread,
			cpu_mode = cpu_mode,
			feature_dropout_ratio = feature_dropout_ratio,
			feature_select_iteration = feature_select_iteration,
			grp_changing_thread = grp_changing_thread,
			max_train_size = max_train_size,
			model_select_iteration = model_select_iteration,
			outlier_thread = outlier_thread,
			regulatory_trace_depth = regulatory_trace_depth,
			stabilize_patient = stabilize_patient,
			stabilize_iteration = stabilize_iteration,
			top_grp_amount = top_grp_amount,
			z_score_extract_thread = z_score_extract_thread,
		)

		self._lockon = threading.Lock()
		print('Protocol:', protocol)
		print('Silent:', self._silent)

		# Do everything unit by unit
		if protocol == 'solo':
			self._proto_solo()

		# Multithreading protocol
		elif protocol == 'multi':
			self._proto_multi()

		self.atlas = self._combine_unit_reports()
		print('Operation Time: ', time.time() - start)
		print('\nComplete\n')


	def save_reports(self,
					 folder_path:str = None,
					 network_header:str = 'network_',
					 save_unit_reports:bool = False,
					):
		"""
		Save meta processed GRN, pseudo-sample GRNs,
		meta-GRN based analysis report,
		AGEAS based analysis report, and key atlas extracted by AGEAS.

		:param folder_path: <str Default = None>
			Path to create folder for saving AGEAS report files.

		:param network_header: <str Default = 'network_'>
			The name header for each network in atlas.

		:param save_unit_reports: <bool Default = False>
			Whether saving key GRPs extracted by each AGEAS Unit or not.
			If True, reports will be saved in folder_path under folders
			named 'no_{}'.format(unit_num) starting from 0.

		"""
		# prepare folder path
		if folder_path[-1] != '/':
			folder_path += '/'
		if not os.path.exists(folder_path):
			os.makedirs(folder_path)

		# Meta GRN Analysis
		meta_grn.Analysis(self.meta.grn).save(folder_path + 'meta_report.csv')

		self.pseudo_grns.save(folder_path + 'pseudo_sample_GRNs.js')
		self.meta.grn.save_json(folder_path + 'meta_GRN.js')

		if save_unit_reports:
			for index, atlas in enumerate(self._reports):
				report_path = folder_path + 'no_' + str(index) + '/'
				if not os.path.exists(report_path):
					os.makedirs(report_path)
				atlas.grps.save(report_path + 'grps_importances.csv')
				json.encode(atlas.outlier_grps, report_path + 'outlier_grps.js')

		self.atlas.report(self.meta.grn, header = network_header).to_csv(
			folder_path + 'report.csv',
			index = False
		)

		self.atlas.list_grp_as_df().to_csv(
			folder_path + 'grp_scores.csv',
			index = False
		)

		json.encode(
			{network_header + str(k):v.as_dict() for k,v in enumerate(
				self.atlas.nets)},
			folder_path + 'full_atlas.js'
		)

		json.encode(self.atlas.key_atlas.as_dict(), folder_path+'key_atlas.js')


	# Protocol SOLO
	def _proto_solo(self):
		for i in range(self._unit_num):
			id = 'RN_' + str(i)
			new_unit = copy.deepcopy(self._basic_unit)
			print('Unit', id, 'Ready')
			print('\nSending Unit', id, '\n')
			if self._silent: sys.stdout = open(os.devnull, 'w')
			new_unit.select_models()
			new_unit.launch()
			atlas = new_unit.generate_atlas()
			self._reports.append(atlas)
			del new_unit
			if self._silent: sys.stdout = sys.__stdout__
			print(id, 'RTB\n')

	# Protocol MULTI
	def _proto_multi(self):
		units = []
		for i in range(self._unit_num):
			id = 'RN_' + str(i)
			units.append(threading.Thread(target=self._multi_unit, name=id))
			print('Unit', id, 'Ready')

		# Time to work
		print('\nSending All Units\n')
		if self._silent: sys.stdout = open(os.devnull, 'w')

		# Start each unit
		for unit in units: unit.start()

		# Wait till all thread terminates
		for unit in units: unit.join()
		if self._silent: sys.stdout = sys.__stdout__
		print('Units RTB\n')

	# Model selection and network contruction part run parallel
	def _multi_unit(self,):
		new_unit = copy.deepcopy(self._basic_unit)
		new_unit.select_models()
		# lock here since SHAP would bring Error
		self._lockon.acquire()
		new_unit.launch()
		self._lockon.release()
		atlas = new_unit.generate_atlas()
		self._reports.append(atlas)
		del new_unit

	# Combine information from reports returned by each unit
	def _combine_unit_reports(self):
		all_grps = dict()
		for atlas in self._reports:
			for network in atlas.nets:
				for id, record in network.grps.items():
					if id not in all_grps:
						all_grps[id] = record
					else:
						all_grps[id] = self._combine_grp_records(
							record_1 = all_grps[id],
							record_2 = record
						)

		# now we build the atlas
		answer = extractor.Atlas()
		for id, grp in all_grps.items():
			answer.update_net_with_grp(grp = grp, meta_grn = self.meta.grn)
		answer.find_bridges(meta_grn = self.meta.grn)
		return answer

	# combine information of same GRP form different reports
	def _combine_grp_records(self, record_1, record_2):
		answer = copy.deepcopy(record_1)
		if answer.type != record_2.type:

			if answer.type == GRP_TYPES[2]:
				assert answer.score == 0
				if record_2.type != GRP_TYPES[2]:
					answer.type = record_2.type
					answer.score = record_2.score

			else:
				if record_2.type != GRP_TYPES[2]:
					answer.type = GRP_TYPES[3]
					answer.score = max(answer.score, record_2.score)

		else:
			answer.score = max(answer.score, record_2.score)

		return answer
