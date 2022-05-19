#!/usr/bin/env python3
"""
Ageas Reborn

author: jy, nkmtmsys
"""

import itertools
import pandas as pd
import ageas.lib as lib
import ageas.tool as tool
from collections import OrderedDict


TYPES = ['Standard', 'Outer_Signifcant', 'Bridge']


class Extract(object):
	"""
	Extract key genes from the most important GRPs
	"""

	def __init__(self,
				grp_importances = None,
				score_thread = None,
				outlier_grps = {},
				top_grp_amount = None):
		super(Extract, self).__init__()
		self.regulons = []
		self.regulatory_sources = None
		self.outlier_grps = outlier_grps
		self.grps = grp_importances.stratify(score_thread,
											top_grp_amount,
											len(outlier_grps))

	# as named
	def build_regulon(self, meta_grn, header = 'regulon_'):
		# process standard grps
		for id in self.grps.index:
			try:
				grp = meta_grn[id]
			except Exception:
				raise lib.Error('GRP', id, 'not in Meta GRN')
			grp['type'] = TYPES[0]
			grp['score'] = self.grps.loc[id]['importance']
			self.__update_regulon_with_grp(grp)
		for id in self.outlier_grps:
			try:
				grp = meta_grn[id]
			except Exception:
				raise lib.Error('GRP', id, 'not in Meta GRN')
			grp['type'] = TYPES[1]
			grp['score'] = self.outlier_grps[id]
			self.__update_regulon_with_grp(grp)

		# combine regulons if sharing common genes
		i = 0
		j = 1
		checked_grps = {}
		while True:
			if i == len(self.regulons) or j == len(self.regulons): break
			combining = False
			reg1_genes = self.regulons[i]['genes'].keys()
			reg2_genes = self.regulons[j]['genes'].keys()
			assert len([x for x in reg1_genes if x in reg2_genes]) == 0
			for comb in list(itertools.product(reg1_genes, reg2_genes)):
				id = tool.Cast_GRP_ID(comb[0], comb[1])
				if id not in checked_grps:
					checked_grps[id] = None
					if id in meta_grn:
						if not combining: combining = True
						assert id not in self.regulons[i]['grps']
						meta_grn[id]['type'] = TYPES[2]
						meta_grn[id]['score'] = 0
						self.regulons[i]['grps'][id] = meta_grn[id]
			if combining:
				self.__combine_regulons(ind_1 = i, ind_2 = j)
				j = i + 1
			else:
				j += 1
				if j == len(self.regulons):
					i += 1
					j = i + 1
		# update and change regulon to dict type and find key genes
		for regulon in self.regulons:
			for grp in regulon['grps'].values():
				source = grp['regulatory_source']
				target = grp['regulatory_target']
				# update genes type based on GRP type
				if grp['type'] == TYPES[1]:
					if regulon['genes'][source]['type'] != TYPES[1]:
						regulon['genes'][source]['type'] = TYPES[1]
					if regulon['genes'][target]['type'] != TYPES[1]:
						regulon['genes'][target]['type'] = TYPES[1]
				elif grp['type'] == TYPES[0]:
					if (regulon['genes'][source]['type'] != TYPES[1] and
						regulon['genes'][source]['type'] != TYPES[0]):
						regulon['genes'][source]['type'] = TYPES[0]
					if (regulon['genes'][target]['type'] != TYPES[1] and
						regulon['genes'][target]['type'] != TYPES[0]):
						regulon['genes'][target]['type'] = TYPES[0]

				assert source not in regulon['genes'][target]['source']
				assert source not in regulon['genes'][target]['target']
				assert target not in regulon['genes'][source]['source']
				assert target not in regulon['genes'][source]['target']
				regulon['genes'][target]['source'].append(source)
				regulon['genes'][source]['target'].append(target)
				if grp['reversable']:
					regulon['genes'][source]['source'].append(target)
					regulon['genes'][target]['target'].append(source)
		self.regulons = {header + str(i):e for i, e in enumerate(self.regulons)}
		self.regulatory_sources = self.__get_reg_sources()
		del self.grps
		del self.outlier_grps

	# Use extra GRPs from meta GRN to link different Regulons
	def link_regulon(self, meta_grn = None, allowrance = 1):
		add_on = {k:{'grps':{},'genes':{}} for k in self.regulons}
		for i in range(allowrance):
			for id, grp in meta_grn.items():
				source = grp['regulatory_source']
				target = grp['regulatory_target']
				anchor = None
				adding = None
				if (source in self.regulatory_sources and
					target not in self.regulatory_sources):
					anchor = source
					adding = target
				if (target in self.regulatory_sources and
					source not in self.regulatory_sources):
					anchor = target
					adding = source
				if anchor is not None and adding is not None:
					regulon_id = self.regulatory_sources[anchor]['regulon_id']
					if (id not in self.regulons[regulon_id]['grps'] and
						id not in add_on[regulon_id]['grps']):
						add_on[regulon_id]['grps'][id] = grp
						if adding not in add_on[regulon_id]['genes']:
							add_on[regulon_id]['genes'][adding] = {}
						assert anchor not in add_on[regulon_id]['genes'][adding]
						add_on[regulon_id]['genes'][adding][anchor] = i
		print(add_on)
		# if type == 'regulatory_source': known = 'regulatory_target'
		# elif type == 'regulatory_target': known = 'regulatory_source'
		# dict = {}
		# for grp in meta_grn:
		# 	record = meta_grn[grp]
		# 	if record[known] in self.regulatory_sources:
		# 		target = record[type]
		# 		if target not in dict:
		# 			dict[target] = 1
		# 		else:
		# 			dict[target] += 1
		# dict = {ele:dict[ele] for ele in dict if dict[ele] >= allowrance}
		# dict = OrderedDict(sorted(dict.items(),key=lambda x:x[1], reverse=True))
		# if type == 'regulatory_source':      self.common_reg_source = dict
		# elif type == 'regulatory_target':    self.common_reg_target = dict

	# find factors by checking and regulaotry target number and impact score
	def report(self, target_num_thread = 0, influence_thread = 0):
		df = []
		for k, v in self.regulatory_sources.items():
			if v['target_num'] >= target_num_thread:
				df.append([k] + list(v.values()))
		df = pd.DataFrame(sorted(df, key=lambda x:x[-2], reverse = True))
		df.columns=['Gene', 'Regulon', 'Source_Num', 'Target_Num', 'Impact_Val']
		return df

	# combine regulons in self.regulons by index
	def __combine_regulons(self, ind_1, ind_2):
		self.regulons[ind_1]['grps'].update(self.regulons[ind_2]['grps'])
		self.regulons[ind_1]['genes'].update(self.regulons[ind_2]['genes'])
		del self.regulons[ind_2]

	# summarize key regulatory sources appearing in regulons
	def __get_reg_sources(self, target_num_thread = 0):
		dict = {}
		for regulon_id in self.regulons:
			regulon = self.regulons[regulon_id]
			for gene in regulon['genes']:
				source_num = len(regulon['genes'][gene]['source'])
				target_num = len(regulon['genes'][gene]['target'])
				if (gene not in dict and
					regulon['genes'][gene]['type'] != TYPES[2] and
					target_num > target_num_thread):
					dict[gene]= {
									'regulon_id': regulon_id,
									'source_num': source_num,
									'target_num': target_num,
									'impact_val': 0
								}
				elif gene in dict:
					raise lib.Error('Repeated Gene in regulons', gene)
		# filter by top_grp_amount
		return OrderedDict(sorted(dict.items(),
									key = lambda x:x[-1]['target_num'],
									reverse = True))

	# as named
	def __update_regulon_with_grp(self, grp):
		update_ind = None
		combine_ind = None
		source_regulon_ind = None
		target_regulon_ind = None
		source = grp['regulatory_source']
		target = grp['regulatory_target']

		# check whether GRP could be appended into an exist regulon
		for i, regulon in enumerate(self.regulons):
			if source in regulon['genes']:
				assert source_regulon_ind is None
				source_regulon_ind = i
			if target in regulon['genes']:
				assert target_regulon_ind is None
				target_regulon_ind = i

		if source_regulon_ind is None and target_regulon_ind is None:
			# make new regulon data
			regulon = {
				'grps':{grp['id']: grp},
				'genes':{	source: {'source':[], 'target':[], 'type':TYPES[2]},
							target: {'source':[], 'target':[], 'type':TYPES[2]}
						}
			}
			self.regulons.append(regulon)
			return
		elif source_regulon_ind is not None and target_regulon_ind is not None:
			if source_regulon_ind == target_regulon_ind:
				update_ind = source_regulon_ind
			# combine regulons if two are involved
			elif source_regulon_ind != target_regulon_ind:
				update_ind = source_regulon_ind
				combine_ind = target_regulon_ind
		elif source_regulon_ind is not None:
			update_ind = source_regulon_ind
		elif target_regulon_ind is not None:
			update_ind = target_regulon_ind
		else: print('Fuck')

		# update regulon if found destination
		if update_ind is not None:
			# append GRP into self.regulons[update_ind]
			assert grp['id'] not in self.regulons[update_ind]['grps']
			self.regulons[update_ind]['grps'][grp['id']] = grp
			# update gene list
			if source not in self.regulons[update_ind]['genes']:
				self.regulons[update_ind]['genes'][source] = {	'source':[],
																'target':[],
																'type':TYPES[2]}
			elif target not in self.regulons[update_ind]['genes']:
				self.regulons[update_ind]['genes'][target] = {	'source':[],
																'target':[],
																'type':TYPES[2]}

		# combine 2 regulons if new GRP can connect two
		if combine_ind is not None:
			self.__combine_regulons(ind_1 = update_ind, ind_2 = combine_ind)
