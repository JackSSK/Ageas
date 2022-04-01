#!/usr/bin/env python3
"""
GRN related tools

author: jy, nkmtmsys
"""

import re
import ageas.tool as tool
import ageas.tool.gem as gem


class Reader(gem.Reader):
    """
    Read in GRN file from given path
    """
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
        return {'grp_ID':content[0],
                'sourceID':content[1],
                'sourceGroup':content[2],
                'targetID':content[3],
                'targetGroup':content[4],
                'correlation':float(content[5]),
                'attribute':content[6],}
