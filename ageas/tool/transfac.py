#!/usr/bin/env python3
"""
TRANSFAC file related tools

author: jy, nkmtmsys
"""


import re
import ageas.tool as tool

class Reader(tool.Reader_Template):
    """
    Load in Transfac database for determining whether a gene is Transcription
    Factor(TF).
    Will need to be override if TF database in different format.
    """

    def __init__(self, filepath:str = None, feature_type:str = 'gene_symbol'):
        """
        Initialize a Reader Object.

        Args:
            filepath:str = None

            feature_type:str = 'gene_symbol'

        """
        self.load(filepath)
        self.tfs = {}
        if feature_type == 'gene_symbol':
            self._process(position = 3)
        elif feature_type == 'ens_id':
            self._process(position = 4)
        else:
            raise tool.Error('Unsupported factor name type!')
        self.close()


    def _process(self, position):
        """
        Iterate through lines in given file.
        """
        while(True):
            line = self.file.readline().strip()
            if line == '':
                break
            elif line[:1] == '#':
                continue
            content = line.split('\t')
            self._update(content[position])


    def _update(self, data):
        """
        Update dict with given data.
        """
        if re.search(r';', data):
            data = data.split(';')
            for ele in data:
                self.tfs[ele] = None
        else:
            self.tfs[data] = None
