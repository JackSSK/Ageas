#!/usr/bin/env python3
"""
Ageas Reborn

author: jy, nkmtmsys
"""

import warnings
import numpy as np
import pandas as pd
from collections import Counter
import ageas.lib as lib
import ageas.tool as tool
import ageas.tool.grn as grn
import ageas.tool.json as json
import ageas.lib.grp_predictor as grp



class Analysis(object):
	"""
	Find important factors simply by GRN degree.
	"""
	def __init__(self, meta_grn, top = None):
		super(Analysis, self).__init__()
		self.top = top
		temp = {}
		for ele in meta_grn.grps:
			source = meta_grn.grps[ele].regulatory_source
			target = meta_grn.grps[ele].regulatory_target

			if source not in temp:
				temp[source] = 1
			else:
				temp[source] += 1

			if target not in temp:
				temp[target] = 1
			else:
				temp[target] += 1

		if self.top is None: self.top = len(temp)
		temp = [[k[0],
				meta_grn.genes[k[0]].type,
				k[1]] for k in Counter(temp).most_common(self.top)]

		# adding log2FC
		for ele in temp:
			exp = meta_grn.genes[ele[0]].expression_sum
			ele.append(abs(np.log2(exp['group1']+1) - np.log2(exp['group2']+1)))

		# changing to dataframe type
		self.result = pd.DataFrame(temp,columns=['ID','Type','Degree','Log2FC'])

	def save(self, path):
		self.result.to_csv(path, index = False )



class Cast:
	"""
	Cast Meta GRN based on GEMs
	"""
	def __init__(self,
				 gem_data = None,
				 prediction_thread = None,
				 correlation_thread = 0.2,
				 load_path = None
				):
		super(Cast, self).__init__()
		# Initialization
		self.grn = grn.GRN(id = 'Meta')
		self.tfs_no_interaction_rec = dict()

		# Load or Cast
		if load_path is not None:
			self.grn.load_dict(dict = json.decode(load_path))
		else:
			self.__cast(gem_data, prediction_thread, correlation_thread)

	# Process to Cast out GRN construction guidance
	def __cast(self, gem_data, prediction_thread, correlation_thread):
		# proces guidance casting process based on avaliable information
		if gem_data.interactions is not None:
			if gem_data.database_info.interaction_db == 'gtrd':
				self.__with_gtrd(gem_data, correlation_thread)
			elif gem_data.database_info.interaction_db == 'biogrid':
				self.__with_biogrid(gem_data, correlation_thread)
		else:
			self.__no_interaction(gem_data, correlation_thread)
		self.tfs_no_interaction_rec = [x for x in self.tfs_no_interaction_rec]

		# print out amount of TFs not covered by selected interaction database
		print('	Predicting interactions for',
				len(self.tfs_no_interaction_rec),
				'TFs not covered in interaction DB')

		# Start GRNBoost2-like process if thread is set
		if prediction_thread is not None and len(self.tfs_no_interaction_rec)>0:
			gBoost = grp.Predict(gem_data, self.grn.grps, prediction_thread)
			self.grn = gBoost.expand_meta_grn(
				self.grn,
				self.tfs_no_interaction_rec,
				correlation_thread
			)
		print('	Total amount of GRPs in Meta GRN:', len(self.grn.grps))
		print('	Total amount of Genes in Meta GRN:', len(self.grn.genes))
		# else: raise lib.Error('Sorry, such mode is not supported yet!')
		""" ToDo: if more than 1 guide can be casted, make agreement """
		return

	# Make GRN casting guide with GTRD data
	def __with_gtrd(self, data, correlation_thread):
		# Iterate source TF candidates for GRP
		for source in data.genes:
			# Go through tf_list filter if avaliable
			if data.tf_list is not None and source not in data.tf_list:
				continue
			# Get Uniprot ID to use GTRD
			uniprot_ids = []
			try:
				for id in data.interactions.idmap[source].split(';'):
					if id in data.interactions.data:
						 uniprot_ids.append(id)
			except:
				warnings.warn(source, 'not in Uniprot ID Map.')

			# pass this TF if no recorded interactions in GTRD
			if len(uniprot_ids) == 0:
				self.tfs_no_interaction_rec[source] = None
				continue

			# get potential regulatory targets
			reg_target = {}
			for id in uniprot_ids:
				reg_target.update(data.interactions.data[id])

			# Handle source TFs with no record in target database
			if len(reg_target) == 0:
				if source not in self.tfs_no_interaction_rec:
					self.tfs_no_interaction_rec[source] = None
					continue
				else:
					raise lib.Error('Duplicat source TF when __with_gtrd')

			# Iterate target gene candidates for GRP
			for target in data.genes:
				# Handle source TFs with record in target database
				if target in reg_target:
					self.grn.update_grn(
						source = source,
						target = target,
						gem1 = data.group1,
						gem2 = data.group2,
						correlation_thread = correlation_thread
					)

		# update Gene informations
		for gene in self.grn.genes:
			if data.database_info.factor_name_type == 'gene_name':
				self.grn.genes[gene].add_name(gene)
			elif data.database_info.factor_name_type == 'ens_id':
				self.grn.genes[gene].add_ens_id(gene)
			if data.tf_list is not None and gene in data.tf_list:
				self.grn.genes[gene].type = 'TF'
			if gene in data.interactions.idmap:
				for id in data.interactions.idmap[gene].split(';'):
					self.grn.genes[gene].add_uniprot_id(id)
		return

	# Make GRN casting guide with bioGRID data
	def __with_biogrid(self, data, correlation_thread):
		# Iterate source TF candidates for GRP
		for source in data.genes:
			# Go through tf_list filter if avaliable
			if data.tf_list is not None and source not in data.tf_list:
				continue
			reg_target = {}
			if source in data.interactions.data:
				reg_target = {i:None for i in data.interactions.data[source]}
			elif source in data.interactions.alias:
				alias_list = data.interactions.alias[source]
				for ele in alias_list:
					temp = {tar:None for tar in data.interactions.data[ele]}
					reg_target.update(temp)
			else:
				self.tfs_no_interaction_rec[source] = None
				continue

			# Handle source TFs with no record in target database
			if len(reg_target) == 0:
				if source not in self.tfs_no_interaction_rec:
					self.tfs_no_interaction_rec[source] = None
					continue
				else:
					raise lib.Error('Duplicat source TF when __with_biogrid')

			for target in data.genes:
				passing = False
				# Handle source TFs with record in target database
				if target in reg_target:
					passing = True

				elif target in data.interactions.alias:
					for ele in data.interactions.alias[target]:
						if ele in reg_target:
							passing = True

				if passing:
					self.grn.update_grn(
						source = source,
						target = target,
						gem1 = data.group1,
						gem2 = data.group2,
						correlation_thread = correlation_thread
					)
		return

	# Kinda like GTRD version but only with correlation_thread and
	def __no_interaction(self, data, correlation_thread):
		# Iterate source TF candidates for GRP
		for source in data.genes:
			# Go through tf_list filter if avaliable
			if data.tf_list is not None and source not in data.tf_list:
				continue
			for target in data.genes:
				self.grn.update_grn(
					source = source,
					target = target,
					gem1 = data.group1,
					gem2 = data.group2,
					correlation_thread = correlation_thread
				)
		return
