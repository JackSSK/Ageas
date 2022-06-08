#!/usr/bin/env python3
"""
Ageas Reborn

author: jy, nkmtmsys
"""
import math
import numpy as np
import networkx as nx
from warnings import warn
import matplotlib.pyplot as plt
from netgraph import InteractiveGraph
import ageas.tool.grn as grn
import ageas.tool.json as json



TYPES = ['Standard', 'Outer', 'Bridge', 'Mix']


class Plot_Regulon(object):
    """
    Visualize full or partial Regulon and save in PDF format by default
    """

    def __init__(self,
                scale:int = 10,
                root_gene:str = None,
                cor_thread:float = 0.0,
                graph_type:str = 'all',
                impact_depth:int = 1,
                hide_bridge:bool = True,
                bridge_color:str = None,
                file_path:str = None,
                regulon_id:str = 'regulon_0',):
        super(Plot_Regulon, self).__init__()
        self.scale = scale
        self.root_gene = root_gene
        self.cor_thread = cor_thread
        self.graph_type = str(graph_type)
        self.impact_depth = impact_depth

        self.hide_bridge = hide_bridge
        self.bridge_color = bridge_color
        # if removing all the bridges, why specify a color?
        if self.hide_bridge and self.bridge_color is not None:
            warn('bridge_color ignored since hiding bridges')

        # Load in regulon to plot
        self.regulon = grn.GRN(id = regulon_id)
        self.regulon.load_dict(json.decode(file_path)[regulon_id])

        # Plot the whole regulon or set a node as root?
        if self.root_gene is None:
            # add every qualified GRP in given regulon
            grps_to_plot = {
                k:0 for k,v in self.regulon.grps.items() if self.__check(v.type)
            }
        else:
            # only testify GRPs reachable to root_gene in given depth
            grps_to_plot = dict()
            self.__find_grps(root_gene, grps_to_plot, impact_depth)

        # filter GRPs based on correlation thread if set
        if self.cor_thread:
            print('Under Construction')

        # now we make the graph
        self.graph = self.regulon.as_digraph(grp_ids = grps_to_plot.keys())

    # check type
    def __check(self, type): return type != TYPES[2] or not self.hide_bridge

    # recursively find GRPs able to link with root_gene in given depth
    def __find_grps(self, gene, dict, depth):
        if depth >= 1:
            depth -= 1
            reg_targets = self.regulon.genes[gene].target
            reg_sources = self.regulon.genes[gene].source
            for tar in reg_targets + reg_sources:
                id = grn.GRP(gene, tar).id
                # add id to dict accordingly
                if id not in dict and self.__check(self.regulon.grps[id].type):
                    dict[id] = None
                # recurssion
                self.__find_grps(tar, dict, depth)

    # make sure which GRPs are qualified and plot them
    def _draw(self, grps_to_plot):
        # initialization
        graph = list()
        node_size = dict()
        node_shape = dict()
        node_color = dict()
        edge_color = dict()
        edge_width = dict()

        for id in grps_to_plot:
            grp = self.regulon.grps[id]
            weight, color = self.get_weight_color(grp.correlations)

            # ignore GRPs with no significant correlation
            if abs(weight) <= self.cor_thread:
                continue
            # add GRP as an edge
            source = grp.regulatory_source
            target = grp.regulatory_target
            # change color to silver if it's a bridge
            if self.bridge_color is not None and grp.type == TYPES[2]:
                color = self.bridge_color
            edge_color[(source, target)] = color
            edge_width[(source, target)] = 20 * (weight**2)
            graph.append((source, target, weight))
            # add reverse edge if reversable
            if grp.reversable:
                edge_color[(target, source)] = color
                edge_width[(target, source)] = 20 * (weight**2)
                graph.append((target, source, weight))

            # process nodes if not seen yet
            self.__update_node(source, node_shape, node_size, node_color)
            self.__update_node(target, node_shape, node_size, node_color)

        # make sure we have something to plot
        if len(graph) <= 0: raise Exception('No GRPs to plot!')

    # just as named
    def get_weight_color(self, correlations):
        if self.graph_type == 'class1' or self.graph_type == '1':
            weight = correlations['class1']
        elif self.graph_type == 'class2' or self.graph_type == '2':
            weight = correlations['class2']
        elif self.graph_type == 'all':
            weight = abs(correlations['class1']) - abs(correlations['class2'])
        if weight >= 0:
            return weight, 'red'
        else:
            return weight, 'blue'

    # Update Node's (which is a gene) plot setting
    def __update_node(self, gene, node_shape, node_size, node_color):
        if gene not in node_shape:
            target_num = len(self.regulon.genes[gene].target)
            # this gene does have regulatory power on others
            if target_num > 0:
                factor = 1
                # increase node size according to gene's reg power
                if target_num > 10:
                    factor = math.log10(target_num) * 1.5
                shape = 'd'
                size = 30 * factor
            # this gene can only be regulated by others
            else:
                shape = 'o'
                size = 20
            node_size[gene] = size
            node_shape[gene] = shape
            if shape == 'd':
                if (self.bridge_color is not None and
                    self.regulon.genes[gene].type == TYPES[2]):
                    node_color[gene] = self.bridge_color
                else:
                    node_color[gene] = 'gold'
            else:
                node_color[gene] = 'lavender'

    # save the plot. PDF by default
    def save(self, path:str = None, format:str = 'pdf'):
        plt.savefig(path, format = format)
        plt.close()

    # show the interactive graph
    def show(self):
        plt.show()

""" For testing """
# if __name__ == '__main__':
#     a = Plot_Regulon(
#         file_path = 'regulons.js',
#         regulon_id = 'regulon_0',
#         # hide_bridge = False,
#         # bridge_color = 'silver',
#         # type = 'class1',
#         # root_gene = 'ACTA2',
#         # impact_depth = 1,
#     )
#     # a.show()
#     a.save(path = 'temp.pdf')
