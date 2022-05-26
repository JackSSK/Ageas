#!/usr/bin/env python3
"""
Ageas Reborn

author: jy, nkmtmsys
"""
import numpy as np
import matplotlib.pyplot as plt
import ageas.tool.json as json
from netgraph import InteractiveGraph



class Regulon(object):
    """
    Visualize full or partial Regulon

    Interactive mode cannot be saved yet...
    save in PDF format
    """

    def __init__(self,
                regulon:dict = None,
                root_gene:str = None,
                impact_depth:int = None):
        super(Regulon, self).__init__()
        self.plot = None
        self.regulon = regulon
        self.root_gene = root_gene
        self.impact_depth = impact_depth
        weighted_cube = [
            ('MEF2C', 'b', -0.1),
            ('MEF2C', 'KLF4', -0.8),
            ('b', 'c', -0.2),
            ('c', 'd', -0.4),
            ('d', 'c',  0.0),
            ('d', 'MEF2C', -0.2),
            ('e', 'f',  0.7),
            ('f', 'g',  0.9),
            ('g', 'f', -0.2),
            ('g', 'h',  0.5),
            ('h', 'e',  0.1),
            ('MEF2C', 'e',  0.5),
            ('b', 'f', -0.3),
            ('f', 'b', -0.4),
            ('c', 'g',  0.8),
            ('d', 'h',  0.4)
        ]
        edge_width = {(u, v) : 10 * np.abs(w) for (u, v, w) in weighted_cube}
        edge_color = {(u, v) : 'brown' if w <=0 else 'green' for (u, v, w) in weighted_cube}
        node_shape = dict()
        for (u, v, w) in weighted_cube:
            if u not in node_shape or node_shape[u] != 's':
                node_shape[u] = 's'
            if v not in node_shape:
                node_shape[v] = 'o'
        self.plot = InteractiveGraph(
            weighted_cube,
            node_labels=True,
            node_size = 50,
            node_shape = node_shape,
            node_layout='circular',
            # node_label_offset=0.1,
            node_label_fontdict = dict(size=5),
            edge_width = 20,
            edge_color = edge_color,
            scale = (10,10),
            arrows=True
        )

    def save(self, path:str = None, format:str = 'pdf'):
        plt.savefig(path,  format = format)

    def show(self):
        plt.show()



a = Regulon()
a.save(path = 'temp.pdf')
