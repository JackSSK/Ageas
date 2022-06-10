#!/usr/bin/env python3
"""
Ageas Reborn

author: jy, nkmtmsys
"""
import math
import numpy as np
import networkx as nx
from warnings import warn
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.special import softmax
from netgraph import Graph
from netgraph import InteractiveGraph
import ageas.tool.grn as grn
import ageas.tool.json as json



TYPES = ['Standard', 'Outer', 'Bridge', 'Mix']


class Plot_Regulon(object):
    """
    Visualize full or partial Regulon and save in PDF format by default
    """

    def __init__(self,
                root_gene:str = None,
                weight_thread:float = 0.0,
                graph_type:str = 'all',
                impact_depth:int = 1,
                hide_bridge:bool = True,
                file_path:str = None,
                regulon_id:str = 'regulon_0',):
        super(Plot_Regulon, self).__init__()
        self.root_gene = root_gene
        self.graph_type = str(graph_type)
        self.hide_bridge = hide_bridge
        self.impact_depth = impact_depth
        self.weight_thread = weight_thread

        # Load in regulon to plot
        regulon = grn.GRN(id = regulon_id)
        regulon.load_dict(json.decode(file_path)[regulon_id])

        # Plot the whole regulon or set a node as root?
        if self.root_gene is None:
            # add every qualified GRP in given regulon
            grps_to_plot = {
                k:None for k,v in regulon.grps.items() if self.__check(v.type)
            }
        else:
            # only testify GRPs reachable to root_gene in given depth
            grps_to_plot = dict()
            self.__find_grps(regulon, root_gene, grps_to_plot, impact_depth)

        # get weight for GRPs and filter GRPs based on correlation thread
        grps_to_plot = self.__weight_filter(grps_to_plot, regulon)

        # now we make the graph
        self.graph = regulon.as_digraph(grp_ids = grps_to_plot.keys())
        # make sure we have something to play with
        if len(self.graph) <= 0: raise Exception('No GRPs in Graph!')

    # check type
    def __check(self, type): return type != TYPES[2] or not self.hide_bridge

    # filter GRPs based on correlation thread
    def __weight_filter(self, grp_ids, regulon):
        answer = {}
        for id in grp_ids:
            weight = self.get_weight(regulon.grps[id].correlations)
            if abs(weight) >= self.weight_thread:
                answer[id] = None
                regulon.grps[id].weight = weight
        return answer

    # recursively find GRPs able to link with root_gene in given depth
    def __find_grps(self, regulon, gene, dict, depth):
        if depth >= 1:
            depth -= 1
            for tar in regulon.genes[gene].target + regulon.genes[gene].source:
                id = grn.GRP(gene, tar).id
                # add id to dict accordingly
                if id not in dict and self.__check(regulon.grps[id].type):
                    dict[id] = None
                # recurssion
                self.__find_grps(regulon, tar, dict, depth)


    # make sure which GRPs are qualified and plot them
    def draw(self,
            scale:int = 1,
            seed:int = 1936,
            figure_size:int = 10,
            layout:str = 'spring',
            font_size:int = 5,
            hide_target_labels:bool = False,
            bridge_color:str = None,):
        # initialization
        node_size = list()
        node_color = list()
        node_alhpa = 0.8
        edge_width = list()
        edge_style = list()
        edge_color = list()
        edge_alpha = list()
        # cmap = plt.cm.coolwarm
        labels = {n:n for n in self.graph}
        plt.figure(figsize = (figure_size, figure_size))
        ax = plt.gca()
        ax.set_axis_off()

        # specify layout
        if layout == 'circular':
            pos = nx.circular_layout(self.graph, scale = scale,)
        elif layout == 'spring':
            pos = nx.spring_layout(
                self.graph,
                scale = scale,
                seed = seed,
                k = 0.1,
            )
        elif layout == 'randon':
            pos = nx.random_layout(self.graph, seed = seed)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.graph)
        elif layout == 'graphviz':
            pos = nx.nx_pydot.graphviz_layout(self.graph)
        elif layout == 'planar':
            pos = nx.planar_layout(self.graph)
        elif layout == 'spectral':
            pos = nx.spectral_layout(self.graph)
        # if removing all the bridges, why specify a color?
        if self.hide_bridge and bridge_color is not None:
            warn('bridge_color ignored since hiding bridges')

        # Get Node information
        for node, data in self.graph.nodes(data = True):
            factor = 1
            # target_num = len([i for i in self.graph.successors(node)])
            target_num = len(data['target'])
            if target_num > 0:
                node_color.append(3)
                # increase node size according to gene's reg power
                if target_num > 10:
                    factor = math.log2(target_num) * 2
            else:
                node_color.append(2)
                if hide_target_labels:
                    del labels[node]
            size = 300 * factor
            node_size.append(size)
        # Draw Nodes and Labels
        nodes = nx.draw_networkx_nodes(
            G = self.graph,
            pos = pos,
            cmap = plt.cm.Set3,
            node_size = node_size,
            node_color = node_color,
            alpha = node_alhpa,
        )
        labels = nx.draw_networkx_labels(
            G = self.graph,
            pos = pos,
            labels = labels,
            font_size = font_size,
            clip_on = True,
        )

        # Get Edge Information
        n_edge = self.graph.number_of_edges()
        for (source, target, data) in self.graph.edges(data = True):
            edge_width.append(abs(data['weight'])*4)

            if data['type'] == TYPES[2]:
                edge_style.append(':')
            else:
                edge_style.append('-')

            if bridge_color is not None and data['type'] == TYPES[2]:
                edge_color.append(bridge_color)
            elif data['weight'] < 0:
                edge_color.append('blue')
            else:
                edge_color.append('red')


        edges = nx.draw_networkx_edges(
            G = self.graph,
            pos = pos,
            node_size = node_size,
            arrowstyle = "-|>",
            arrowsize = 20,
            edge_color = edge_color,
            # edge_cmap = plt.cm.RdBu,
            width = edge_width,
            style = edge_style,
        )
        # set alpha value for each edge
        # edge_alphas = [(5 + i) / (n_edge + 4) for i in range(n_edge)]
        # for i in range(n_edge):
        #     edges[i].set_alpha(edge_alphas[i])


        # pc = mpl.collections.PatchCollection(edges, cmap = cmap)
        # pc.set_array(edge_color)
        # plt.colorbar(pc, ax = ax)

    # just as named
    def get_weight(self, correlations):
        if self.graph_type == 'class1' or self.graph_type == '1':
            weight = correlations['class1']
        elif self.graph_type == 'class2' or self.graph_type == '2':
            weight = correlations['class2']
        elif self.graph_type == 'all':
            weight = abs(correlations['class1']) - abs(correlations['class2'])
        return weight

    # save the plot. PDF by default
    def save(self, path:str = None, format:str = 'pdf'):
        plt.savefig(path, format = format)
        plt.close()

    # show the interactive graph
    def show(self):
        plt.show()


# if __name__ == '__main__':
#     import ageas
#
#
#     header = 'liverCCl4/hsc_pf_a6w/'
#     for i in range(1):
#         folder_path = header + 'run_' + str(i) + '/'
#         atlas_path = folder_path + 'key_atlas.js'
#         a = Plot_Regulon(
#             file_path = atlas_path,
#             regulon_id = 'regulon_0',
#             hide_bridge = False,
#             # graph_type = 'class1',
#             root_gene = 'HAND2',
#             # impact_depth = 1,
#         )
#         a.draw(figure_size = 20, font_size = 10, layout = 'spring')
#         # a.show()
#         a.save(path = folder_path + 'ultimate_useless_mess.pdf')

        # a = ageas.Plot_Regulon(
        #     figure_size = 10,
        #     file_path = atlas_path,
        #     regulon_id = 'regulon_0',
        #     # hide_bridge = False,
        #     # bridge_special_color = 'silver',
        #     # type = 'class1',
        #     # root_gene = 'ACTA2',
        #     # impact_depth = 1,
        # )
        # # a.show()
        # a.save(path = folder_path + 'useless_mess.pdf')
