#!/usr/bin/env python3
"""
Pack GTRD data into .json based format with necessary data

ToDo:
Make sure load all tf target info at once would still be fine


author: jy, nkmtmsys
"""


import os
import pandas as pd
import ageas.tool as tool
import ageas.tool.json as json



class Processor:
    """
    Object to process stratified GTRD file.
    """
    def __init__(self, specie_path, feature_type):
        self.data = dict()
        self.idmap = json.decode(
            specie_path + 'uniprot_idmapping.stratified.js.gz'
        )[feature_type]

        if os.path.exists(specie_path + 'gtrd_whole_genes.js.gz'):
        	data = json.decode(specie_path + 'gtrd_whole_genes.js.gz')
        else:
        	data = json.decode(specie_path + 'gtrd_[-1000,+100].js.gz')

        if feature_type == 'gene_symbol':
            ind = 1
        elif feature_type == 'ens_id':
            ind = 0

        for source in data:
            self.data[source] = {ele[ind]:ele[-1] for ele in data[source]}


class Packer:
    """
    Object to stratify and pack interaction data from GTRD
    """
    def __init__(self, database_path:str = None, outpath:str = None):
        """
        Initialize an object.

        Args:
            database_path:str = None

            outpath:str = None

        """
        self.dict = {}
        filenames = os.listdir(database_path)
        for ele in filenames:
            uniprot_id = ele.split('.txt')[0].upper()
            if uniprot_id not in self.dict:
                self.dict[uniprot_id] = self._process(
                    database_path + '/' + ele,
                )
            else:
                raise tool.Error('Duplicated TF file in:', database_path)

        if outpath is not None:
            json.encode(self.dict, outpath)

    def _process(self, filepath, sep = '\t', header = 0):
        data = pd.read_csv(filepath, sep = sep, header = header)
        result = list(zip(data['ID'], data['Gene symbol'], data['SiteCount']))
        return result



# Packer('mouse/genes promoter[-1000,+100]', 'gtrd_promoter[-1000,+100].js.gz')
