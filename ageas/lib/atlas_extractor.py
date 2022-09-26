#!/usr/bin/env python3
"""
Ageas Reborn

author: jy, nkmtmsys
"""

import copy
import numpy as np
import pandas as pd
import networkx as nx
from itertools import product
import ageas.lib as lib
import ageas.tool.grn as grn


GRP_TYPES = ['Standard', 'Outer', 'Bridge', 'Mix']


class Atlas(object):
	"""
	Object to extract Atalas of key networks from the most important GRPs.
	"""

	def __init__(self,
				 correlation_thread:float = 0.2,
				 grp_importances = None,
				 score_thread = None,
				 outlier_grps = dict(),
				 top_grp_amount = None
				):
		self.nets = list()
		self.key_atlas = None
		self.grps = grp_importances
		self.outlier_grps = outlier_grps
		self.correlation_thread = correlation_thread
		# will be deleted after recorded in self.nets
		if grp_importances is not None:
			self.top_grps=grp_importances.stratify(score_thread, top_grp_amount)

	# as named
	def build_network(self, meta_grn):
		# set outlier bonus score to max score of standard GRPs
		outlier_bonus_score = self.top_grps.iloc[0]['importance']
		# process standard grps
		for id in self.top_grps.index:
			try:
				grp = meta_grn.grps[id]
			except Exception:
				raise lib.Error('GRP', id, 'not in Meta GRN')
			grp.type = GRP_TYPES[0]
			grp.score = self.top_grps.loc[id]['importance']
			self.update_net_with_grp(grp, meta_grn)
		del self.top_grps
		for id in self.outlier_grps:
			try:
				grp = meta_grn.grps[id]
			except Exception:
				raise lib.Error('GRP', id, 'not in Meta GRN')
			grp.type = GRP_TYPES[1]
			grp.score = self.outlier_grps[id] + outlier_bonus_score
			self.update_net_with_grp(grp, meta_grn)
		self.key_atlas = self.get_key_atlas()

	def find_bridges(self, meta_grn = None):
		# link networks if bridge can be build with factors already in networks
		i = 0
		j = 1
		checked_grps = {}
		while True:
			if i == len(self.nets) or j == len(self.nets): break
			combining = False
			reg1_genes = self.nets[i].genes.keys()
			reg2_genes = self.nets[j].genes.keys()
			for comb in list(product(reg1_genes, reg2_genes)):
				id = grn.GRP(comb[0], comb[1]).id
				if id not in checked_grps:
					checked_grps[id] = None
					if id in meta_grn.grps:
						if not combining:
							combining = True
						meta_grn.grps[id].type = GRP_TYPES[2]
						meta_grn.grps[id].score = 0
						self.nets[i].grps[id] = meta_grn.grps[id]
			if combining:
				self.update_net_with_net(i, j)
				del self.nets[j]
				j = i + 1
			else:
				j += 1
				if j == len(self.nets):
					i += 1
					j = i + 1

		# Need to do something here to add Bridges linking TFs in same network
		for i, reg in enumerate(self.nets):
			for comb in list(product(reg.genes.keys(), reg.genes.keys())):
				if comb[0] == comb[1]: continue
				id = grn.GRP(comb[0], comb[1]).id
				if (id not in reg.grps and id in meta_grn.grps):
					meta_grn.grps[id].type = GRP_TYPES[2]
					meta_grn.grps[id].score = 0
					self.nets[i].grps[id] = meta_grn.grps[id]

		self.key_atlas = self.get_key_atlas()

	# Use extra GRPs from meta GRN to link different networks
	def add_reg_sources(self, meta_grn = None,):
		for gene in self.key_atlas.genes:
			for source in meta_grn.genes[gene].source:
				passing = True
				for net in self.nets:
					if source in net.genes:
						passing = False
						break

				if passing:
					grp = meta_grn.grps[grn.GRP(source, gene).id]
					grp.type = GRP_TYPES[2]
					grp.score = 0
					self.update_net_with_grp(grp, meta_grn)

	# find factors by checking and regulaotry target number and impact score
	def report(self, meta_grn, header = 'network_'):
		df = list()
		for rec in self.key_atlas.genes.values():
			exps = rec.expression_sum
			df.append([
				rec.id,
				rec.symbol,
				header + str(rec.net_id),
				rec.type,
				rec.source_num,
				rec.target_num,
				len(rec.target),
				abs(np.log2(exps['class1'] + 1) - np.log2(exps['class2'] + 1))
			])

		df = pd.DataFrame(sorted(df, key = lambda x:x[-1], reverse = True))
		df.columns = [
			'ID',
			'Gene Symbol',
			'Network',
			'Type',
			'Source_Num',
			'Target_Num',
			'Meta_Degree',
			'Log2FC'
		]
		return df

	def list_grp_as_df(self):
		answer = pd.concat([reg.list_grp_as_df() for reg in self.nets])
		answer.sort_values('Score')
		return answer


	# summarize key regulatory sources appearing in networks
	def get_key_atlas(self):
		answer = grn.GRN(id = 'key_regulatory_sources')
		for net_id, net in enumerate(self.nets):
			graph = net.as_digraph()

			# add nodes
			for node in graph.nodes:
				assert node not in answer.genes
				reg_target_num = len([x for x in graph.successors(node)])
				if reg_target_num > 0:
					reg_source_num = len([x for x in graph.predecessors(node)])
					answer.genes[node] = copy.deepcopy(net.genes[node])
					setattr(answer.genes[node], 'net_id', net_id)
					setattr(answer.genes[node], 'source_num', reg_source_num)
					setattr(answer.genes[node], 'target_num', reg_target_num)

			# add edge for nodes already in answer
			for grp in net.grps:
				if (net.grps[grp].regulatory_source in answer.genes and
					net.grps[grp].regulatory_target in answer.genes):
					answer.grps[grp] = copy.deepcopy(net.grps[grp])
					setattr(answer.grps[grp], 'net_id', net_id)
		return answer

	def update_net_with_grp(self, grp, meta_grn):
		update_ind = None
		combine_ind = None
		source_net_ind = None
		target_net_ind = None
		source = grp.regulatory_source
		target = grp.regulatory_target

		# check whether GRP could be appended into an exist network
		for i, net in enumerate(self.nets):
			if source in net.genes:
				assert source_net_ind is None
				source_net_ind = i
			if target in net.genes:
				assert target_net_ind is None
				target_net_ind = i

		# make new network data
		if source_net_ind is None and target_net_ind is None:
			net = grn.GRN()
			net.grps[grp.id] = grp
			net.genes[source] = copy.deepcopy(meta_grn.genes[source])
			net.genes[target] = copy.deepcopy(meta_grn.genes[target])
			self.nets.append(net)
			return
		elif source_net_ind is not None and target_net_ind is not None:
			if source_net_ind == target_net_ind:
				update_ind = source_net_ind

			# combine networks if two are involved
			elif source_net_ind != target_net_ind:
				update_ind = source_net_ind
				combine_ind = target_net_ind
		elif source_net_ind is not None:
			update_ind = source_net_ind
		elif target_net_ind is not None:
			update_ind = target_net_ind
		else:
			raise lib.Error('Something wrong with network updating process')

		# update network if found destination
		if update_ind is not None:

			# append GRP into self.nets[update_ind]
			assert grp.id not in self.nets[update_ind].grps
			self.nets[update_ind].grps[grp.id] = grp

			# update gene list
			if source not in self.nets[update_ind].genes:
				self.nets[update_ind].genes[source] = copy.deepcopy(
					meta_grn.genes[source]
				)
			elif target not in self.nets[update_ind].genes:
				self.nets[update_ind].genes[target] = copy.deepcopy(
					meta_grn.genes[target]
				)

		# combine 2 networks if new GRP can connect two
		if combine_ind is not None:
			self.update_net_with_net(update_ind, combine_ind)
			del self.nets[combine_ind]

	# combine networks in self.nets by index
	def update_net_with_net(self, net_id:int = None, source_id:int = None):
		self.nets[net_id].grps.update(self.nets[source_id].grps)
		self.nets[net_id].genes.update(self.nets[source_id].genes)
