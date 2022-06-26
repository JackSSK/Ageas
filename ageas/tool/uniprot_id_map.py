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
    def __init__(self, filename):
        self.load(filename)

    # Stratify the map file with selected genra -> Genra: Uniprot ID
    def stratify(self, genra = ['Gene_Name', 'Gene_Synonym']):
        result = {}
        while(True):
            line = self.file.readline()
            if line == '':
                break
            elif line[:1] == '#':
                continue
            content = line.split('\t')
            if content[1] in genra:
                gene_symbol = content[2].strip()
                uniprot_id = content[0].strip().upper()
                if gene_symbol in result:
                    result[gene_symbol] += ';' + uniprot_id
                else:
                    result[gene_symbol] = uniprot_id
        return result

# if __name__ == '__main__':
    # data = Reader(filename = 'HUMAN_9606_idmapping.dat.gz')
    # gene_symbols = data.stratify(
    #     genra = ['Gene_Name', 'Gene_Synonym']
    # )
    # ensembl_ids = data.stratify(
    #     genra = ['Ensembl']
    # )
    # data = {
    #     'gene_symbol': gene_symbols,
    #     'ens_id': ensembl_ids
    # }
    # json.encode(data = data, out = 'uniprot_idmapping.stratified.js.gz')
