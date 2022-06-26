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
    Process summarized GTRD file
    """
    def __init__(self, specie_path, factor_id_type, path):
        self.path = specie_path + path
        self.nameType = factor_id_type
        self.idmap = json.decode(
            specie_path + 'uniprot_idmapping.stratified.js.gz'
        )[factor_id_type]
        self.data = json.decode(self.path)



class Packer:
    def __init__(self, database_path, outpath = None):
        self.dict = {}
        filenames = os.listdir(database_path)
        for ele in filenames:
            uniprot_id = ele.split('.txt')[0].upper()
            if uniprot_id not in self.dict:
                self.dict[uniprot_id] = self._process_gene(
                    database_path + '/' + ele,
                    sep = "\t",
                    header = 0
                )
            else:
                raise tool.Error('Duplicated TF file in:', database_path)

        if outpath is not None:
            json.encode(self.dict, outpath)

    def _processFull(self, filepath, sep, header):
        result = {}
        data = pd.read_csv(filepath, sep = sep, header = header)
        for index, row in data.iterrows():
            result[row['Gene symbol'].upper()] = {
                'id':row['id'],
                'ENS_TransID':row['Ensembl ID'],
                'siteCount':row['SiteCount']
            }
        return result

    def _process_gene(self, filepath, sep, header):
        data = pd.read_csv(filepath, sep = sep, header = header)
        return dict(zip(data['Gene symbol'], data['SiteCount']))
