#!/usr/bin/env python3
"""
GRN related tools

author: jy, nkmtmsys
"""

import re
import statistics as sta
import ageas.tool as tool
import ageas.tool.json as json



class GRN(object):
    """
    docstring for Meta_GRN.
    """

    def __init__(self, id = None):
        super(GRN, self).__init__()
        self.id = id
        self.genes = dict()
        self.grps = dict()

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
        if source in gem1.index and target in gem1.index:
            cor_class1 = tool.Get_Pearson(
                gem1.loc[[source]].values[0],
                gem1.loc[[target]].values[0]
            )
        # check cor_class2
        if source in gem2.index and target in gem2.index:
            cor_class2 = tool.Get_Pearson(
                gem2.loc[[source]].values[0],
                gem2.loc[[target]].values[0]
            )
        # Go through abs(correlation) threshod check
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
            self.add_grp(grp_id, source, target, cor_class1, cor_class2)
            if source not in self.genes: self.add_gene(source, gem1, gem2)
            if target not in self.genes: self.add_gene(target, gem1, gem2)

    def add_grp(self, id, source, target, cor_class1, cor_class2):
        assert id not in self.grps
        # may change it to a class later
        self.grps[id] = GRP(
            id = id,
            regulatory_source = source,
            regulatory_target = target,
            correlations = {
                'class1':float(cor_class1),
                'class2':float(cor_class2)
            }
        )
        return

    def add_gene(self, id, gem1, gem2):
        assert id not in self.genes
        # get expression mean values
        if id in gem1.index:    class1_exp = sta.mean(gem1.loc[[id]].values[0])
        else:                   class1_exp = 0
        if id in gem2.index:    class2_exp = sta.mean(gem2.loc[[id]].values[0])
        else:                   class2_exp = 0
        # may change it to a class later
        self.genes[id] = Gene(
            id = id,
            expression_mean = {
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

    def save_json(self, path):
        json.encode(self.as_dict(), path)
        return

    def load_json(self, path):
        grn = json.decode(path)
        self.genes = {id:Gene(**grn['genes'][id]) for id in grn['genes']}
        self.grps = {id:GRP(**grn['grps'][id]) for id in grn['grps']}
        return



class Gene(object):
    """
    docstring for Gene.
    """

    def __init__(self,
                id = None,
                name = None,
                ens_id = None,
                uniprot_id = None,
                type = None,
                expression_mean = dict()):
        super(Gene, self).__init__()
        self.id = id
        self.type = type
        self.name = name
        self.ens_id = ens_id
        self.uniprot_id = uniprot_id
        self.expression_mean = expression_mean

    def as_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'ens_id': self.ens_id,
            'uniprot_id': self.uniprot_id,
            'type': self.type,
            'expression_mean': self.expression_mean
        }

    def get_name(self):
        print('Under Construction')
        self.name = None

    def get_ens_id(self):
        print('Under Construction')
        self.ens_id = None

    def get_uniprot_id(self):
        print('Under Construction')
        self.uniprot_id = None



class GRP(object):
    """
    docstring for GRP.
    """

    def __init__(self,
                regulatory_source = None,
                regulatory_target = None,
                id = None,
                type = None,
                score = None,
                reversable = False,
                correlations = dict()):
        super(GRP, self).__init__()
        self.id = id
        self.type = type
        self.score = score
        self.reversable = reversable
        self.correlations = correlations
        self.regulatory_source = regulatory_source
        self.regulatory_target = regulatory_target
        if self.id is None:
            self.id = self.cast_id(regulatory_source, regulatory_target)

    def as_dict(self):
        return {
            'id': self.id,
            'type': self.type,
            'score': self.score,
            'reversable': self.reversable,
            'regulatory_source': self.regulatory_source,
            'regulatory_target': self.regulatory_target,
            'correlations':self.correlations
        }

    def cast_id(self, source, target):
        if source > target: return source + '_' + target
        else:               return target + '_' + source


class Reader(tool.Reader_Template):
	"""
	NOTE:! Very outdated !
	NOTE:! Very outdated !
	NOTE:! Very outdated !
    NOTE:! Don't Use !
    NOTE:! Don't Use !
    NOTE:! Don't Use !


	Class to read in scRNA-seq or bulk RNA-seq based Gene Expression Matrices
	Only suppordt .cvs and .txt for now
	"""
	def __init__(self, filename, skipFirst = False, stdevThread = None):
		super(Reader, self).__init__()
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
		# Check file format
		if len(content) < 7:
			if content == ['\n']:
				raise tool.Error('Bad GRN format: empty line')
			else:
				raise tool.Error('Fatal GRN format: not enough info')
		# Process current record
		else:
			id = content[0]
			if id not in self.entryCoords:
				self.entryCoords[id] = coordinate
			else:
				raise tool.Error('Dulpicate GRP id in GRN: ' + self.filePath)

	# Pattern info in each line
	def _prepareInfo(self, content):
		return {
			'id':content[0],
			'regulatory_source':content[1],
			'sourceGroup':content[2],
			'regulatory_target':content[3],
			'targetGroup':content[4],
			'correlation':float(content[5]),
			'attribute':content[6],
		}

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


	""" Old GEM Reader """
	# # Pattern info in each line
	# def _prepareInfo(self, content):
	#     id = content[0].strip().upper()
	#     data = [float(x) for x in content[1:]]
	#     return id, data
	#
	# # Process information in reading line
	# def _processLine(self, coordinate, content, stdevThread):
	# 	id, data = self._prepareInfo(content)
	# 	stdev = sta.stdev(data)
	# 	# Filter records based on stdev thredshould
	# 	if stdevThread is None or stdev >= stdevThread:
	# 		if id not in self.entryCoords:
	# 			self.entryCoords[id] = coordinate
	# 		# Keep one with high stdev
	# 		else:
	# 			message = id + 'is duplicated'
	# 			warn(message)
	# 			if stdev > sta.stdev(self.get(id)[1]):
	# 				self.entryCoords[id] = coordinate
	# 			# get back to original position
	# 			self.file.seek(coordinate)
	# 			line = self.file.readline().strip()
