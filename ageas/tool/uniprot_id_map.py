#!/usr/bin/env python3
"""
Uniprot ID Map related tools

Source data:
https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/

author: jy, nkmtmsys
"""

import ageas.tool as tool
import ageas.tool.json as json



class Reader(tool.Reader_Template):
    """
    Read in ID Map file obtained from Uniprot
    """
    def __init__(self, filepath:str = None):
        """
        Initialize a Reader object.

        Args:
            filepath:str = None
        """
        self.load(filepath)

    def stratify(self, genra = ['Gene_Name', 'Gene_Synonym']):
        """
        Stratify idmapping file from Uniprot as {Selected Genra: Uniprot ID}.
        """
        result = {}
        while(True):
            line = self.file.readline()
            if line == '':
                break
            elif line[:1] == '#':
                continue
            content = line.split('\t')
            if content[1] in genra:
                feature_id = content[2].strip()
                uniprot_id = content[0].strip().upper()
                if feature_id in result:
                    result[feature_id] += ';' + uniprot_id
                else:
                    result[feature_id] = uniprot_id
        return result


# if __name__ == '__main__':
    # ensembl_ids = Reader(filepath = 'HUMAN_9606_idmapping.dat.gz').stratify(
    #     genra = ['Ensembl']
    # )
    # gene_symbols = Reader(filepath = 'HUMAN_9606_idmapping.dat.gz').stratify(
    #     genra = ['Gene_Name', 'Gene_Synonym']
    # )
    # data = {
    #     'gene_symbol': gene_symbols,
    #     'ens_id': ensembl_ids
    # }
    # json.encode(data = data, out = 'uniprot_idmapping.stratified.js.gz')
