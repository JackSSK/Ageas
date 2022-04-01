#!/usr/bin/env python3
"""
Ageas Reborn
Gene Expression Matrix related tools

ToDo: make SQL Database based on folder/file
Find a way to deal with seurat objects
author: jy, nkmtmsys
"""

import os
import re
import pandas as pd
import statistics as sta
import ageas.tool as tool
from warnings import warn



class Reader_old(tool.Reader_Template):
	"""
	Class to read in scRNA-seq or bulk RNA-seq based Gene Expression Matrices
	Only suppordt .cvs and .txt for now
	"""
	def __init__(self, filename, skipFirst = False, stdevThread = None):
		# Initialization
		self.load(filename)
		self.entryCoords = {}
		self.iteration = 0
		# Determine file type
		if re.search(r'\.txt', self.filePath): self.split = '\t'
		elif re.search(r'\.csv', self.filePath): self.split = ','
		else: raise tool.Error(self.filePath, ' is not supported format')
		# Skip first line
		if skipFirst: line = self.file.readline()
		# Iterate through all lines
		while(True):
			coordinate = self.file.tell()
			line = self.file.readline().strip()
			# terminate at the end
			if line == '':break
			# skip comments
			elif line[:1] == '#': continue
			else:
				content = line.split(self.split)
				self._processLine(coordinate, content, stdevThread)

	# Process information in reading line
	def _processLine(self, coordinate, content, stdevThread):
		id, data = self._prepareInfo(content)
		stdev = sta.stdev(data)
		# Filter records based on stdev thredshould
		if stdevThread is None or stdev >= stdevThread:
			if id not in self.entryCoords:
				self.entryCoords[id] = coordinate
			# Keep one with high stdev
			else:
				message = id + 'is duplicated'
				warn(message)
				if stdev > sta.stdev(self.get(id)[1]):
					self.entryCoords[id] = coordinate
				# get back to original position
				self.file.seek(coordinate)
				line = self.file.readline().strip()

	# Output all Gene Expression data in dict format
	def makeGeneExpDict(self, stdevKpRatio):
		records = []
		for id in self.entryCoords:
			_, data = self.get(id)
			if stdevKpRatio is not None:
				stdev = sta.stdev(data)
				records.append([id, stdev, data])
			else:
				records.append([id, data])
		# Filter records based on keep ratio
		if stdevKpRatio is not None:
			records.sort(key = lambda x:x[1], reverse = True)
			records = records[:int(len(records) * stdevKpRatio)]
		return {record[0]: record[-1] for record in records}

	# Get info of selected id
	def get(self, id):
		self.file.seek(self.entryCoords[id])
		line = self.file.readline().strip()
		content = line.split(self.split)
		return self._prepareInfo(content)

	# Pattern info in each line
	def _prepareInfo(self, content):
		id = content[0].strip().upper()
		data = [float(x) for x in content[1:]]
		return id, data

	# For iteration
	def __next__(self):
		entryKeys = [*self.entryCoords]
		if self.iteration == len(entryKeys):
			self.iteration = 0
			raise StopIteration
		else:
			id = entryKeys[self.iteration]
			self.iteration += 1
			return self.get(self, id)



class Reader(object):
	"""
	Class to read in scRNA-seq or bulk RNA-seq based Gene Expression Matrices
	Only suppordt .cvs and .txt for now
	"""
	def __init__(self, path, **kwargs):
		# Decide which seperation mark to use
		if re.search(r'csv', path): 	self.sep = ','
		elif re.search(r'txt', path): 	self.vsep = '\t'
		# determine compression method
		if re.search(r'.gz', path): 	self.compression = 'gzip'
		elif re.search(r'.zip', path):	self.compression = 'zip'
		else: 							self.compression = 'infer'
		try:
			self.data = pd.read_csv(path,
									sep = self.sep,
									compression = self.compression,
									**kwargs)
		except Exception as GEM_Reader_Error:
			raise tool.Error('Unsupported File Type: ', path)

	# filter data frame based on standered deviations
	def STD_Filter(self, std_value_thread = None, std_ratio_thread = None):
		self.data = tool.STD_Filter(df = self.data,
									std_value_thread = std_value_thread,
									std_ratio_thread = std_ratio_thread)



class Folder(object):
	"""
	Manipulations on Gene Expressio Matrices in given folder
	"""
	def __init__(self,
				path,
				file_type = 'csv', 				# type of file considering GEM
				compression_method = 'infer',	# compression method of files
				header_row = 0, 				# header row for all GEM
				index_col = 0, 					# index column for all GEM
				):
		self.path = path
		self.header_row = header_row
		self.index_col = index_col

		# file type check
		if file_type == 'csv': self.sep = ','
		elif file_type == 'txt': self.sep = '\t'
		else: raise tool.Error('Folder: Unknown file type')
		self.file_type = file_type
		self.compression_method = compression_method

	# combine all GEMs to one unified GEM
	# all GEMs should have exactly same index (Gene list)
	def combine(self,
				method = 'inner',
				keep_repeated_samples = False,
				std_value_thread = None,
				std_ratio_thread = None,
				outpath = None
				):
		# Grab all necessary samples first
		result = None
		for filename in os.listdir(self.path):
			# Skip files without targeting appendix
			if not re.search(self.file_type, filename): continue
			filepath = self.path + '/' + filename
			gem = pd.read_csv(filepath,
								sep = self.sep,
								header = self.header_row,
								index_col = self.index_col,
								compression = self.compression_method)
			# Initialize output df if still empty
			if result is None:
				result = gem
				continue
			if keep_repeated_samples:
				result = result.join(gem, how = method)
			else:
				unique_samples = gem.columns.difference(result.columns)
				# if nothing new, move forward
				if len(unique_samples) == 0: continue
				result = pd.merge(result,
								gem[unique_samples],
								left_index = True,
								right_index = True,
								how = method)
			# Just in case
			result = result[~result.index.duplicated(keep='first')]
		del gem

		# filter by standard deviations if needed
		if std_value_thread is not None or std_ratio_thread is not None:
			result = tool.STD_Filter(result, std_value_thread, std_ratio_thread)

		# return or save matrix
		if outpath is None: return result
		else:
			result.to_csv(outpath,
							sep = self.sep,
							compression = self.compression_method)
