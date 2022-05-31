#!/usr/bin/env python3
"""
Ageas Reborn

author: jy, nkmtmsys
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import ageas.tool as tool
import ageas.tool.json as json
from netgraph import InteractiveGraph


TYPES = ['Standard', 'Outer', 'Bridge', 'Mix']


class Plot_Regulon(object):
    """
    Visualize full or partial Regulon

    Interactive mode cannot be saved yet...
    save in PDF format
    """

    def __init__(self,
                scale:int = 10,
                regulon_id:str = 'regulon_0',
                remove_bridge:bool = True,
                cor_thread:float = 0.0,
                type:str = 'all',
                file_path:str = None,
                root_gene:str = None,
                impact_depth:int = 1):
        super(Plot_Regulon, self).__init__()
        self.plot = None
        self.type = str(type)
        self.scale = scale
        self.cor_thread = cor_thread
        self.remove_bridge = remove_bridge
        self.regulon = json.decode(file_path)[regulon_id]
        self.root_gene = root_gene
        self.impact_depth = impact_depth
        if self.root_gene is None:
            self.plot = self._process_full()
        else:
            self.plot = self._process_part(root_gene, impact_depth)

    def _process_full(self):
        grps_to_plot = dict()
        for id, grp in self.regulon['grps'].items():
            # ignore bridge GRPs
            if grp['type'] == TYPES[2] and self.remove_bridge:
                continue
            grps_to_plot[id] = None
        # now we make the plot
        self.plot = self._draw(grps_to_plot = grps_to_plot)

    def _process_part(self, root_gene, impact_depth):
        grps_to_plot = dict()
        self.__find_grps(root_gene, grps_to_plot, impact_depth)
        self.plot = self._draw(grps_to_plot = grps_to_plot)

    def _draw(self, grps_to_plot):
        graph = list()
        node_size = dict()
        node_shape = dict()
        node_color = dict()
        edge_color = dict()
        edge_width = dict()

        for id in grps_to_plot:
            grp = self.regulon['grps'][id]
            weight, color = self.get_correlation(grp['correlations'])

            # ignore GRPs with no significant correlation
            if abs(weight) <= self.cor_thread:
                continue
            # add GRP as an edge
            source = grp['regulatory_source']
            target = grp['regulatory_target']
            # change color to silver if it's a bridge
            if grp['type'] == TYPES[2]:
                color = 'silver'
            edge_color[(source, target)] = color
            edge_width[(source, target)] = 20 * (weight**2)
            graph.append((source, target, weight))
            # add reverse edge if reversable
            if grp['reversable']:
                edge_color[(target, source)] = color
                edge_width[(target, source)] = 20 * (weight**2)
                graph.append((target, source, weight))

            # process nodes if not seen yet
            self.__update_node(source, node_shape, node_size, node_color)
            self.__update_node(target, node_shape, node_size, node_color)


        plot = InteractiveGraph(
            graph,
            node_labels = True,
            node_size = node_size,
            node_shape = node_shape,
            node_color = node_color,
            # node_layout ='circular',
            # node_label_fontdict = dict(size=2),
            edge_width = edge_width,
            edge_color = edge_color,
            scale = (self.scale, self.scale),
            arrows=True
        )

        return plot


    def get_correlation(self, correlations):
        if self.type == 'class1' or self.type == '1':
            cor = correlations['class1']
        elif self.type == 'class2' or self.type == '2':
            cor = correlations['class2']
        elif self.type == 'all':
            cor = abs(correlations['class1']) - abs(correlations['class2'])
        if cor >= 0:
            return cor, 'red'
        else:
            return cor, 'blue'

    def __find_grps(self, gene, grps_to_plot, depth):
        if depth >= 1:
            depth -= 1
            reg_targets = self.regulon['genes'][gene]['target']
            reg_sources = self.regulon['genes'][gene]['source']
            for tar in reg_targets + reg_sources:
                grp_id = tool.Cast_GRP_ID(gene, tar)
                if grp_id not in grps_to_plot:
                    # ignore bridges if asked to
                    if (self.regulon['grps'][grp_id]['type'] == TYPES[2] and
                        self.remove_bridge):
                        continue
                    grps_to_plot[grp_id] = None
                self.__find_grps(tar, grps_to_plot, depth)


    def __set_node_shape_size(self, gene):
        target_num = len(self.regulon['genes'][gene]['target'])
        if target_num > 0:
            if target_num > 10:
                factor = math.log10(target_num) * 1.5
                return 'd', 30 * factor
            else:
                return 'd', 30
        else:
            return 'o', 20

    def __update_node(self, gene, node_shape, node_size, node_color):
        if gene not in node_shape:
            shape, size = self.__set_node_shape_size(gene)
            node_size[gene] = size
            node_shape[gene] = shape
            if shape == 'd':
                if self.regulon['genes'][gene]['type'] == TYPES[2]:
                    node_color[gene] = 'silver'
                else:
                    node_color[gene] = 'gold'
            else:
                node_color[gene] = 'lavender'


    # save the plot. PDF by default
    def save(self, path:str = None, format:str = 'pdf'):
        plt.savefig(path, format = format)

    # show the interactive graph
    def show(self):
        plt.show()

""" For testing """
if __name__ == '__main__':
    a = Plot_Regulon(
        file_path = 'regulons.js',
        regulon_id = 'regulon_0',
        remove_bridge = False,
        # type = 'class2',
        # root_gene = 'LHX2',
    )
    # a.show()
    a.save(path = 'temp.pdf')
