#!/usr/bin/env python3
"""
Ageas Reborn
Generate pseudo-cell GRNs (psGRNs) from GEMs

author: jy, nkmtmsys

ToDo:
__file_method not done at all
"""

import os
import re
import time
import copy
import statistics as sta
from scipy.stats import pearsonr
import ageas.tool as tool
import ageas.tool.grn as grn
import ageas.tool.gem as gem
import ageas.tool.json as json
import ageas.lib.meta_grn_caster as meta_grn
import ageas.database_setup.binary_class as binary_db



def Data_Preprocess(correlation_thread:float = 0.2,
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
					):
	"""
	Function to integrate database information and get pseudo-sample GRNs
	from gene expression data.

	:param correlation_thread: <float Default = 0.2>
		Gene expression correlation thread value of GRPs.

		Potential GRPs failed to reach this value will be dropped.

	:param database_path: <str Default = None>
		Database header.

		If specified, class1_path and class2_path will be rooted here.

	:param database_type: <str Default = 'gem_files'>
		Type of data class1_path and class1_path are directing to
		Supporting:

			'gem_files': Each path is directing to a GEM file.
			Pseudo samples will be generated with sliding window algo.

			'gem_folders': Each path is directing to a GEM folder. Files in
			each folder will be used to generate pseudo samples.

			'mex_folders': Each path is directing to a folder consisting MEX
			files(***matrix.mtx***, ***genes.tsv or features.tsv***,
			***barcodes.tsv***)

		Pseudo-sample GRNs will be generated with sliding window method.

	:param class1_path: <str Default = None>
		Path to file or folder of class 1 samples data

	:param class2_path: <str Default = None>
		Path to file or folder of class 2 samples data

	:param interaction_database: <str Default = 'gtrd'>
		Which interaction database to use for confirming a GRP has a
		high possibility to exist.
		Supporting:

			None: No database will be used. As long as a GRP can pass
			all related filters, it's good to go.

			'gtrd': Using GTRD as regulatory pathway reference.
			https://gtrd.biouml.org/

			'biogrid': Using BioGRID as regulatory pathway reference.
			Gene symbols must be given as index in GEM matrix or MEX feature
			file.
			https://thebiogrid.org/

	:param log2fc_thread: <float Default = None>
		Log2 fold change thread to filer non-differntial expressing genes.

		It's generally not encouraged to set up this filter since it can
		result in lossing key TFs not having great changes on overall
		expression volume but having changes on expression pattern.

		If local computational power is relatively limited, setting up
		this thread can help a lot to keep program runable.

	:param meta_load_path: <str Default = None>
		Path to load meta_GRN

	:param mww_p_val_thread: <str Default = 0.05>
		Gene expression Mann–Whitney–Wilcoxon test p-value thread.
		To make sure one gene's expression profile is not constant among
		differnt classes.

	:param normalize: <str Default = None>
		Choose of normalization method on input GEMs.
		Supporting:

			None: No normalization will be done.

			'CPM': Counts Per Million(CPM).

			'Min_Max_1000': Values multiplied by 100 after Min-Max
				Normalization

	:param prediction_thread: <str or float Default = 'auto'>
		The importance thread for a GRP predicted with GRNBoost2-like
		algo to be included.
		Supporting:

			'auto': Automatically set up thread value by minimum imporatnace
			value of a interaction database recorded GRP of TF having
			most amount of GRPs. If not using interaction database, it
			will be set by (1 / amount of genes)

			float type: Value will be set as thread directly

	:param psgrn_load_path: <str Default = None>
		Path to load pseudo-sample GRNs.

	:param specie: <str Default = 'mouse'>
		Specify which sepcie's interaction database shall be used.
		Supporting:

			'mouse': Mus Musculus.

			'human': Homo sapiens.

	:param sliding_window_size: <int Default = 100>
		Number of samples a pseudo-sample generated with
		sliding window technique contains.

	:param sliding_window_stride: <int Default = None>
		Stride of sliding window when generating pseudo-samples.

	:param std_value_thread: <float Default = 1.0>
		Set up gene expression standard deviation thread by value.
		To filter genes having relatively constant expression.

	:param std_ratio_thread: <float Default = None>
		Set up gene expression standard deviation thread by portion.
		Only genes reaching top portion will be kept.

	"""

	# Get database information
	database_info = binary_db.Setup(
		database_path,
		database_type,
		class1_path,
		class2_path,
		specie,
		interaction_database,
		sliding_window_size,
		sliding_window_stride
	)

	# if reading in GEMs, we need to construct pseudo-cGRNs first
	# or if we are reading in MEX, make GEM first and then mimic GEM mode
	if (re.search(r'gem' , database_info.type) or
		re.search(r'mex' , database_info.type)):
		gem_data = binary_db.Load_GEM(
			database_info,
			mww_p_val_thread,
			normalize,
			log2fc_thread,
		)

		# Load meta GRN if path specified
		if meta_load_path is not None:
			meta = meta_grn.Cast(load_path = meta_load_path)
		# Get meta GRN if not loaded
		else:
			start1 = time.time()
			meta = meta_grn.Cast(
				gem_data = gem_data,
				prediction_thread = prediction_thread,
				correlation_thread = correlation_thread,
				std_value_thread = std_value_thread,
				std_ratio_thread = std_ratio_thread,
			)
			print('Time to cast Meta GRN : ', time.time() - start1)

		# Load psGRNs if path specified
		if psgrn_load_path is not None:
			psGRNs = Make(load_path = psgrn_load_path)
		# Get pseudo samples if not loaded
		else:
			start = time.time()
			psGRNs = Make(
				database_info = database_info,
				std_value_thread = std_value_thread,
				std_ratio_thread = std_ratio_thread,
				correlation_thread = correlation_thread,
				gem_data = gem_data,
				meta_grn = meta.grn
			)
			print('Time to get Pseudo-Sample GRNs : ', time.time() - start)

	# if we are reading in GRNs directly, just process them
	elif re.search(r'grn' , database_info.type):
		psGRNs = None
		meta = None
		print('trainer.py: mode GRN need to be revised here')
	else:
		raise lib.Error('Unrecogonized database type: ', database_info.type)

	return database_info, meta, psGRNs



class Make:
	"""
	Make GRNs for gene expression datas
	"""
	def __init__(self,
				 database_info = None,
				 std_value_thread = None,
				 std_ratio_thread = None,
				 correlation_thread = 0.2,
				 gem_data = None,
				 meta_grn = None,
				 load_path = None
				):
		super(Make, self).__init__()
		# Initialize
		self.database_info = database_info
		self.std_value_thread = std_value_thread
		self.std_ratio_thread = std_ratio_thread
		self.correlation_thread = correlation_thread
		if self.correlation_thread is None: self.correlation_thread = 0

		# load in
		if load_path is not None:
			self.class1_psGRNs,self.class2_psGRNs= self.__load_psGRNs(load_path)

		# Make GRNs
		else:
			self.class1_psGRNs, self.class2_psGRNs = self.__make_psGRNs(
				gem_data = gem_data,
				meta_grn = meta_grn
			)

	# main controller to cast pseudo cell GRNs (psGRNs)
	def __make_psGRNs(self, gem_data, meta_grn):
		if gem_data is not None:
			class1_psGRNs = self.__loaded_gem_method(
				class_type = 'class1',
				gem = gem_data.class1.data,
				meta_grn = meta_grn
			)
			class2_psGRNs = self.__loaded_gem_method(
				class_type = 'class2',
				gem = gem_data.class2.data,
				meta_grn = meta_grn
			)
		elif self.database_info.type == 'gem_folders':
			class1_psGRNs = self.__folder_method(
				'class1',
				self.database_info.class1_path,
				meta_grn
			)
			class2_psGRNs = self.__folder_method(
				'class2',
				self.database_info.class2_path,
				meta_grn
			)
		elif self.database_info.type == 'gem_files':
			# need to revise here!
			class1_psGRNs = self.__file_method(
				'class1',
				self.database_info.class1_path,
				meta_grn
			)
			class2_psGRNs = self.__file_method(
				'class2',
				self.database_info.class2_path,
				meta_grn
			)
		else:
			raise tool.Error('psGRN Caster Error: Unsupported database type')
		return class1_psGRNs, class2_psGRNs

	# as named
	def __file_method(self, class_type, path, meta_grn):
		psGRNs = dict()
		print('psgrn_caster.py:class Make: need to do something here')
		return psGRNs

	# as named
	def __loaded_gem_method(self, class_type, gem, meta_grn):
		psGRNs = dict()
		sample_num = 0
		start = 0
		end = self.database_info.sliding_window_size
		# set stride
		if self.database_info.sliding_window_stride is not None:
			stride = self.database_info.sliding_window_stride
		else:
			stride = end
		# use sliding window techinque to set pseudo samples
		loop = True
		while loop:
			if start >= len(gem.columns):
				break
			elif end >= len(gem.columns):
				end = len(gem.columns)
				loop = False
			sample_id = 'sample' + str(sample_num)
			sample = gem.iloc[:, start:end]
			if meta_grn is not None:
				pseudo_sample = self.__process_sample_with_metaGRN(
					class_type,
					sample,
					sample_id,
					meta_grn
				)
			else:
				pseudo_sample = self.__process_sample_without_metaGRN(
					class_type,
					sample,
					sample_id
				)
			# Save data into psGRNs
			psGRNs[sample_id] = pseudo_sample
			start += stride
			end += stride
			sample_num += 1
		return psGRNs

	# as named
	def __folder_method(self, class_type, path, meta_grn):
		data = self.__readin_folder(path)
		psGRNs = dict()
		for sample_id in data:
			if meta_grn is not None:
				pseudo_sample = self.__process_sample_with_metaGRN(
					class_type,
					data[sample_id],
					path,
					meta_grn
				)
			else:
				pseudo_sample = self.__process_sample_without_metaGRN(
					class_type,
					data[sample_id],
					path
				)
			# Save data into psGRNs
			psGRNs[sample_id] = pseudo_sample
		return psGRNs

	# as named
	def __process_sample_with_metaGRN(self,
									  class_type,
									  gem,
									  sample_id,
									  meta_grn
									 ):
		pseudo_sample = grn.GRN(id = sample_id)
		for grp in meta_grn.grps:
			source_ID = meta_grn.grps[grp].regulatory_source
			target_ID = meta_grn.grps[grp].regulatory_target
			try:
				source_exp = list(gem.loc[[source_ID]].values[0])
				target_exp = list(gem.loc[[target_ID]].values[0])
			except:
				continue
			# No need to compute if one array is constant
			if len(set(source_exp)) == 1 or len(set(target_exp)) == 1:
				continue
			cor = pearsonr(source_exp, target_exp)[0]
			if abs(cor) > self.correlation_thread:
				if grp not in pseudo_sample.grps:
					pseudo_sample.add_grp(
						id = grp,
						source = source_ID,
						target = target_ID,
						correlations = {class_type: cor}
					)
				if source_ID not in pseudo_sample.genes:
					pseudo_sample.genes[source_ID] = copy.deepcopy(
						meta_grn.genes[source_ID]
					)
					pseudo_sample.genes[source_ID].expression_sum = {
						class_type: float(sta.mean(source_exp))
					}
				if target_ID not in pseudo_sample.genes:
					pseudo_sample.genes[target_ID] = copy.deepcopy(
						meta_grn.genes[target_ID]
					)
					pseudo_sample.genes[target_ID].expression_sum = {
						class_type: float(sta.mean(target_exp))
					}
		return pseudo_sample

	# Process data without guidance
	# May need to revise later
	def __process_sample_without_metaGRN(self, class_type, gem, sample_id):
		pseudo_sample = grn.GRN(id = sample_id)
		# Get source TF
		for source_ID in gem.index:
			# Get target gene
			for target_ID in gem.index:
				if source_ID == target_ID:
					continue
				grp_id = grn.GRP(source_ID, target_ID).id
				if grp_id not in pseudo_sample.grps:
					# No need to compute if one array is constant
					source_exp = gem[source_ID]
					target_exp = gem[target_ID]
					if len(set(source_exp)) == 1 or len(set(target_exp)) == 1:
						continue
					cor = pearsonr(source_exp, target_exp)[0]
					if abs(cor) > self.correlation_thread:
						if grp not in pseudo_sample.grps:
							pseudo_sample.add_grp(
								id = grp_id,
								source = source_ID,
								target = target_ID,
								correlations = {class_type: cor}
							)
						if source_ID not in pseudo_sample.genes:
							pseudo_sample.genes[source_ID] = grn.Gene(
								id = source_ID,
								expression_sum = {
									class_type: float(sta.mean(source_exp))
								}
							)
						if target_ID not in pseudo_sample.genes:
							pseudo_sample.genes[target_ID] = grn.Gene(
								id = target_ID,
								expression_sum = {
									class_type: float(sta.mean(target_exp))
								}
							)
					else:
						pseudo_sample[grp_id] = None
		return pseudo_sample

	# Readin Gene Expression Matrices in given class path
	def __readin_folder(self, path):
		result = dict()
		for filename in os.listdir(path):
			# read in GEM files
			result[path + '/' + filename] = gem.Reader(
				filename, header = 0, index_col = 0
			).data
		return result

	# as named
	def update_with_remove_list(self, remove_list):
		for sample in self.class1_psGRNs:
			for id in remove_list:
				if id in self.class1_psGRNs[sample].grps:
					del self.class1_psGRNs[sample].grps[id]
		for sample in self.class2_psGRNs:
			for id in remove_list:
				if id in self.class2_psGRNs[sample].grps:
					del self.class2_psGRNs[sample].grps[id]
		return

	# temporal psGRN saving method
	""" need to be revised later to save psGRNs file by file"""
	def save(self, save_path):
		json.encode(
			{
				'class1':{k:v.as_dict() for k,v in self.class1_psGRNs.items()},
				'class2':{k:v.as_dict() for k,v in self.class2_psGRNs.items()}
			},
			save_path
		)
		return

	# load in psGRNs from files
	""" need to be revised later with save_psGRNs"""
	def __load_psGRNs(self, load_path):
		data = json.decode(load_path)
		class1_psGRNs = dict()
		class2_psGRNs = dict()
		for k,v in data['class1'].items():
			temp = grn.GRN(id = k)
			temp.load_dict(dict = v)
			class1_psGRNs[k] = temp
		for k,v in data['class2'].items():
			temp = grn.GRN(id = k)
			temp.load_dict(dict = v)
			class2_psGRNs[k] = temp
		return class1_psGRNs, class2_psGRNs


	# OLD: Save GRN files as js.gz in new folder
	# def save_GRN(self, data, save_path):
	#     for sample in data:
	#         names = sample.strip().split('/')
	#         name = names[-1].split('.')[0] + '.js'
	#         path = '/'.join(names[:-3] + [save_path, names[-2], name])
	#         # Make dir if dir not exists
	#         folder = os.path.dirname(path)
	#         if not os.path.exists(folder):
	#             os.makedirs(folder)
	#         # Get GRN and save it
	#         grn = data[sample]
	#         json.encode(grn, out = path)
