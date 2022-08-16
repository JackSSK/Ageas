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
    def __init__(self, specie_path:str = '', feature_type:str = 'ens_id'):
        self.data = dict()
        self.idmap = json.decode(
            specie_path + 'uniprot_idmapping.stratified.js.gz'
        )[feature_type]

        if os.path.exists(specie_path + 'gtrd_whole_genes.js.gz'):
            data = json.decode(specie_path + 'gtrd_whole_genes.js.gz')
        else:
            data = json.decode(specie_path + 'gtrd_promoter-1000.js.gz')

        if feature_type == 'gene_symbol':
            ind = 1
        elif feature_type == 'ens_id':
            ind = 0

        for source in data['data']:
            self.data[source] = {
                data['ens_idmap'][x[0]][ind]:x[-1] for x in data['data'][source]
            }


class Packer:
    """
    Object to stratify and pack interaction data from GTRD
    """
    def __init__(self, database_path:str = None, outpath:str = None):
        """
        Initialize an object.

        :param database_path:str = None
        :param outpath:str = None
        """
        self.dict = dict()
        self.token_rec = list()
        self.id_token = dict()
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
            json.encode(
                {'ens_idmap': self.token_rec, 'data': self.dict},
                outpath
            )


    def _process(self, filepath, sep = '\t', header = 0):
        result = list()
        data = pd.read_csv(filepath, sep = sep, header = header)

        for i, id in enumerate(data['ID']):
            symbol = data['Gene symbol'][i]
            if id not in self.id_token:
                self.id_token[id] = len(self.token_rec)
                self.token_rec.append([id, symbol])
            else:
                assert self.token_rec[self.id_token[id]][1] == symbol
            result.append([self.id_token[id], int(data['SiteCount'][i])])

        return result



# Packer('mouse/genes whole[-5000,+5000]', 'gtrd_whole_genes.js.gz')
# Processor()
