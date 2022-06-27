#!/usr/bin/env python3
"""
Read in MEX format Single Cell Expression Data

author: jy, nkmtmsys
"""
import os
import re
import csv
import gzip
import scipy.io
import pandas as pd
from warnings import warn
import ageas.tool.gem as gem



class Reader(gem.Reader):
	"""
	Object to read in MEX data.

	"""

	def __init__(self,
				 matrix_path:str = None,
				 features_path:str = None,
				 barcodes_path:str = None,
				):
		"""
		Initialize a new MEX reader object.

		Parameters:
			matrix_path:str = None

			features_path:str = None

			barcodes_path:str = None

		"""
		# Process features
		with gzip.open(features_path, 'rt') as features:
			self.features = [
				{'id':x[0], 'name':x[1]} for x in csv.reader(
					features, delimiter = '\t'
				)
			]

		# Process matrix
		self.data = pd.DataFrame(scipy.io.mmread(matrix_path).toarray())

		# Process barcodes
		with gzip.open(barcodes_path, 'rt') as barcodes:
			self.data.columns = [
				rec[-1] for rec in csv.reader(barcodes, delimiter = '\t')
			]

	def get_gem(self, save_path:str = None, handle_repeat:str = 'sum',):
		"""
		Obtain GEM data frame from processed MEX file.

		Parameters:
			save_path:str = None

			handle_repeat:str = 'sum'
		"""
		self.data.index = [x['id'] for x in self.features]

		# sum up data sharing same gene name if any
		if len(self.data.columns) != len(list(set(self.data.columns))):
			warn('Found repeated barcodes in MEX! Merging.')
			if handle_repeat == 'first':
				self.data=self.data[~self.data.columns.duplicated(keep='first')]
			elif handle_repeat == 'sum':
				self.data = self.data.groupby(self.data.columns).sum()

		# summ up data sharing same barcode if any
		if len(self.data.index) != len(list(set(self.data.index))):
			warn('Found repeated genes in MEX! Merging.')
			if handle_repeat == 'first':
				self.data = self.data[~self.data.index.duplicated(keep='first')]
			elif handle_repeat == 'sum':
				self.data = self.data.groupby(self.data.index).sum()

		# assertion part
		assert len(self.data.columns) == len(list(set(self.data.columns)))
		assert len(self.data.index) == len(list(set(self.data.index)))

		# save GEM if specified path to save
		if save_path is not None: self.data.to_csv(save_path)
		return self.data.loc[~(self.data==0).all(axis=1)]


# if __name__ == '__main__':
#     gem = Reader(
#         matrix_path = 'GSM4085627_10x_5_matrix.mtx.gz',
# 		features_path = 'GSM4085627_10x_5_genes.tsv.gz',
#         barcodes_path = 'GSM4085627_10x_5_barcodes.tsv.gz'
#     )
#     gem.get_gem(save_path = '../pfA6w.csv')
