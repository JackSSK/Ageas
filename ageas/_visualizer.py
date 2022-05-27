#!/usr/bin/env python3
"""
Ageas Reborn

author: jy, nkmtmsys
"""
import numpy as np
import matplotlib.pyplot as plt
import ageas.tool.json as json
from netgraph import InteractiveGraph


TYPES = ['Standard', 'Outer', 'Bridge']


class Plot_Regulon(object):
    """
    Visualize full or partial Regulon

    Interactive mode cannot be saved yet...
    save in PDF format
    """

    def __init__(self,
                scale:int = 100,
                regulon_id:str = None,
                file_path:str = None,
                root_gene:str = None,
                impact_depth:int = 1):
        super(Plot_Regulon, self).__init__()
        self.plot = None
        self.scale = scale
        self.regulon = json.decode(file_path)[regulon_id]
        self.root_gene = root_gene
        self.impact_depth = impact_depth
        if self.root_gene is None:
            self.plot = self._process_full()
        else:
            self.plot = self._process_part(root_gene, impact_depth)

    def _process_full(self):
        graph = list()
        edge_color = dict()
        node_shape = dict()
        for id, grp in self.regulon['grps'].items():
            if grp['type'] == TYPES[2]: continue
            cors = grp['correlations']
            # if cors['class1'] == 0.0 or cors['class2'] == 0.0:
            #     cor = abs(max(cors['class1'], cors['class2']))
            # elif cors['class1'] != 0.0 and cors['class2'] != 0.0:
            #     cor = (abs(cors['class1']) + abs(cors['class2'])) / 2
            weight = 1
            color = 'red'
            source = grp['regulatory_source']
            target = grp['regulatory_target']
            if source not in node_shape:
                node_shape[source] = self.__set_node_shape(source)
            if target not in node_shape:
                node_shape[target] = self.__set_node_shape(target)
            edge_color[(source, target)] = color
            graph.append((source, target, weight))
            if grp['reversable']:
                edge_color[(target, source)] = color
                graph.append((target, source, weight))

        self._draw(graph = graph, node_shape = node_shape, edge_color = edge_color)

    def _process_part(self, root_gene, impact_depth):
        print('Under Construction')

    def _draw(self, graph, node_shape, edge_color, edge_width = 20):
        # edge_width = {(u, v) : 10 * np.abs(w) for (u, v, w) in graph}
        self.plot = InteractiveGraph(
            graph,
            node_labels = True,
            node_size = 50,
            node_shape = node_shape,
            node_layout ='circular',
            node_label_fontdict = dict(size=5),
            edge_width = edge_width,
            edge_color = edge_color,
            scale = (self.scale, self.scale),
            arrows=True
        )

    def __set_node_shape(self, gene):
        if len(self.regulon['genes'][gene]['target']) > 0:
            return 's'
        else:
            return 'o'

    def save(self, path:str = None, format:str = 'pdf'):
        plt.savefig(path,  format = format)

    def show(self):
        plt.show()

""" For testing """
if __name__ == '__main__':
    a = Regulon(file_path = 'regulons.js', regulon_id = 'regulon_0')
    # a.show()
    a.save(path = 'temp.pdf')
