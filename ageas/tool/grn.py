#!/usr/bin/env python3
"""
GRN related tools

author: jy, nkmtmsys
"""

import re
import pandas as pd
import networkx as nx
import statistics as sta
import ageas.tool as tool
import ageas.tool.json as json



class GRN(object):
	"""
	docstring for Meta_GRN.
	"""

	def __init__(self, id = None, **kwargs):
		super(GRN, self).__init__()
		self.id = id
		self.genes = dict()
		self.grps = dict()
		for key in kwargs:
			setattr(self, key, kwargs[key])

	def update_grn(self, source, target, gem1, gem2, correlation_thread):
		# Skip if processing self-regulating pathway
		if source == target: return
		grp_id = GRP(source, target).id
		if grp_id in self.grps:
			if not self.grps[grp_id].reversable:
				self.grps[grp_id].reversable = True
			return

		# Test out global scale correlation
		cor_class1 = None
		cor_class2 = None
		passed = False

		# check cor_class1
		if source in gem1.data.index and target in gem1.data.index:
			cor_class1 = tool.Get_Pearson(
				gem1.data.loc[[source]].values[0],
				gem1.data.loc[[target]].values[0]
			)

		# check cor_class2
		if source in gem2.data.index and target in gem2.data.index:
			cor_class2 = tool.Get_Pearson(
				gem2.data.loc[[source]].values[0],
				gem2.data.loc[[target]].values[0]
			)

		# Go through abs(correlation) threshold check
		if cor_class1 is None and cor_class2 is None:
			return
		if cor_class1 is None and abs(cor_class2) > correlation_thread:
			passed = True
			cor_class1 = 0
		elif cor_class2 is None and abs(cor_class1) > correlation_thread:
			passed = True
			cor_class2 = 0
		elif cor_class1 is not None and cor_class2 is not None:
			if abs(cor_class1 - cor_class2) > correlation_thread:
				passed = True

		# update GRN if passed correlation filter
		if passed:
			correlations = {
				'class1':float(cor_class1),
				'class2':float(cor_class2)
			}
			self.add_grp(grp_id, source, target, correlations)
			if source not in self.genes:
				self.add_gene(source, gem1, gem2)
			if target not in self.genes:
				self.add_gene(target, gem1, gem2)

	def add_grp(self, id, source, target, correlations):
		assert id not in self.grps
		# may change it to a class later
		self.grps[id] = GRP(
			id = id,
			regulatory_source = source,
			regulatory_target = target,
			correlations = correlations
		)
		return

	def add_gene(self, id, gem1, gem2):
		assert id not in self.genes
		# get expression sum values
		class1_exp = 0
		class2_exp = 0
		if id in gem1.data.index:
			class1_exp = sum(gem1.data.loc[[id]].values[0])
		if id in gem2.data.index:
			class2_exp = sum(gem2.data.loc[[id]].values[0])
		# may change it to a class later
		self.genes[id] = Gene(
			id = id,
			expression_sum = {
				'class1': float(class1_exp),
				'class2': float(class2_exp)
			}
		)
		return

	def as_dict(self):
		return {
			'genes': {id:record.as_dict() for id, record in self.genes.items()},
			'grps':{id:record.as_dict() for id, record in self.grps.items()}
		}

	def as_digraph(self, grp_ids = None):
		graph = nx.DiGraph()
		# Use all GRPs if not further specified
		if grp_ids is None: grp_ids = self.grps.keys()
		for grp_id in grp_ids:
			source = self.grps[grp_id].regulatory_source
			target = self.grps[grp_id].regulatory_target

			# add regulatory source and target genes to nodes
			if not graph.has_node(source):
				graph.add_node(source, **self.genes[source].as_dict())
			if not graph.has_node(target):
				graph.add_node(target, **self.genes[target].as_dict())

			# add GRP as an edge
			graph.add_edge(source, target, **self.grps[grp_id].as_dict())
			if self.grps[grp_id].reversable:
				graph.add_edge(target, source, **self.grps[grp_id].as_dict())
		return graph

	# recursively add up impact score with GRP linking gene to its target
	def get_grps_from_gene(self,
				   		   gene = None,
				   		   depth = 3,
				   		   dict = dict(),
				  		  ):
		"""
		Recursively find GRPs able to link with root_gene in given depth.

		depth: <int> Default = 3
			When assessing a TF's regulatory impact on other genes,
			how far the distance between TF and potential regulatory source
			can be.

			The correlation strength of stepped correlation strength of TF
			and gene still need to be greater than correlation_thread.
		"""
		if depth > 0:
			depth -= 1
			for target in self.genes[gene].target + self.genes[gene].source:
				link_grp = GRP(gene, target).id
				if link_grp not in dict and link_grp in self.grps:
					dict[link_grp] = self.grps[link_grp]
				dict = self.get_grps_from_gene(target, depth, dict)
		return dict

	def list_grp_as_df(self):
		answer = list()
		for id,rec in self.grps.items():
			answer.append([
				rec.regulatory_source,
				self.genes[rec.regulatory_source].symbol,
				rec.regulatory_target,
				self.genes[rec.regulatory_target].symbol,
				rec.reversable,
				rec.type,
				rec.correlations['class1'],
				rec.correlations['class2'],
				rec.score,
			])
		answer = pd.DataFrame(sorted(answer, key=lambda x:x[-1], reverse=True))
		answer.columns = [
			'Regulatory source ID',
			'Regulatory source Gene Symbol',
			'Regulatory target ID',
			'Regulatory target Gene Symbol',
			'Reversable',
			'Type',
			'Correlation in Class 1',
			'Correlation in Class 2',
			'Score'
		]
		return answer

	def save_json(self, path):
		json.encode(self.as_dict(), path)
		return

	def load_dict(self, dict):
		self.genes = {id: Gene(**dict['genes'][id]) for id in dict['genes']}
		self.grps = {id: GRP(**dict['grps'][id]) for id in dict['grps']}
		return



class Gene(object):
	"""
	Object to store information of a Gene in Gene Regulatory Network(GRN).
	"""

	def __init__(self,
				 id:str = None,
				 type:str = 'Gene',
				 expression_sum:float = None,
				 **kwargs
				):
		"""
		Initialize a object.

		:param id:str = None

		:param type:str = 'Gene'

		:param expression_sum:float = None

		:param **kwargs
		"""
		self.id = id
		self.type = type
		self.symbol = id
		self.source = list()
		self.target = list()
		self.expression_sum = expression_sum
		for key in kwargs:
			setattr(self, key, kwargs[key])

	def as_dict(self): return self.__dict__

	def add_ens_id(self, ens_id):
		if not hasattr(self, 'ens_id'):
			self.ens_id = ens_id

	def add_uniprot_id(self, uniprot_id):
		if not hasattr(self, 'uniprot_ids'): self.uniprot_ids = list()
		self.uniprot_ids.append(uniprot_id)



class GRP(object):
	"""
	Object to store information of a Gene Regulatory Pathway(GRP).
	"""

	def __init__(self,
				 regulatory_source:str = None,
				 regulatory_target:str = None,
				 id:str = None,
				 type:str = None,
				 score:str = None,
				 reversable:bool = False,
				 correlations:float = None,
				 **kwargs
				):
		"""
		Initialize a object.

		:param regulatory_source:str = None

		:param regulatory_target:str = None

		:param id:str = None

		:param type:str = None

		:param score:str = None

		:param reversable:bool = False

		:param correlations:float = None

		:param **kwargs

		"""
		self.id = id
		self.type = type
		self.score = score
		self.reversable = reversable
		self.correlations = correlations
		self.regulatory_source = regulatory_source
		self.regulatory_target = regulatory_target
		if self.id is None:
			self.id = self.cast_id(regulatory_source, regulatory_target)
		for key in kwargs:
			setattr(self, key, kwargs[key])

	def as_dict(self):
		"""
		Switch object to a dict.
		"""
		return self.__dict__

	def cast_id(self, source:str = None, target:str = None):
		"""
		Cast the id for GRP according to name of regulatory source and target.

		:param source:str = None
		:param target:str = None
		:return:str ID of GRP

		"""
		if source > target:
			return source + '_' + target
		else:
			return target + '_' + source
