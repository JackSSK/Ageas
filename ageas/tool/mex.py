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
import ageas.tool.gem as gem

class Reader(gem.Reader):
    """ Read in MEX data for Gene expression information"""

    def __init__(self,
                matrix_path = None,
                features_path = None,
                barcodes_path = None):
        super(Reader, self).__init__()
        self.matrix_path = matrix_path
        self.features_path = features_path
        self.barcodes_path = barcodes_pae
        self.data = pd.DataFrame(scipy.io.mmread(self.matrix_path).toarray())
        self.features = csv.reader(gzip.open(self.features_path,'rt'),
                                            delimiter = '\t')
        self.barcodes = csv.reader(gzip.open(self.barcodes_path, 'rt'),
                                            delimiter = '\t')

    def get_gem(self):
    	barcodes = [row[-1] for row in self.barcodes]
    	genes = [x.upper() for x in [row[1] for row in self.features]]
    	self.data.index = genes
    	self.data.columns = barcodes
    	if len(barcodes) != len(list(set(barcodes))):
    		self.data = self.data.groupby(self.data.columns).sum()
    	if len(genes) != len(list(set(genes))):
    		self.data = self.data.groupby(self.data.index).sum()
    	assert len(self.data.columns) == len(list(set(barcodes)))
    	assert len(self.data.index) == len(list(set(genes)))
    	return self.data
