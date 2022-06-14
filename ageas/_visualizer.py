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
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.colors as colors
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
                # call self recurssionly
                self.__find_grps(regulon, tar, dict, depth)

    # make sure which GRPs are qualified and plot them
    def draw(self,
             scale:int = 1,
             seed:int = 1936,
             node_base_size:int = 2,
             figure_size:int = 20,
             method:str = 'netgraph',
             legend_size:int = 10,
             layout:str = 'spring',
             colorbar_shrink:float = 0.25,
             font_size:int = 10,
             hide_target_labels:bool = False,
             edge_width_scale:float = 4.0,
             save_path:str = None):
        # initialization
        self.node_cmap = plt.cm.Set3
        self.edge_cmap = plt.cm.coolwarm

        # Color mapping
        self.edge_scalar_map = cmx.ScalarMappable(
            norm = colors.Normalize(vmin = -1, vmax = 1, clip = True),
            cmap = self.edge_cmap
        )
        self.node_scalar_map = cmx.ScalarMappable(
            norm = colors.Normalize(vmin = 0, vmax = 1, clip = True),
            cmap = self.node_cmap
        )

        fig = plt.figure(figsize = (figure_size, figure_size))
        ax = plt.gca()
        ax.set_axis_off()

        # draw with specified method
        if method == 'networkx':
            self.draw_with_networkx(
                ax = ax,
                scale = scale,
                layout = layout,
                base_size = node_base_size,
                font_size = font_size,
                hide_target_labels = hide_target_labels,
                edge_width_scale =edge_width_scale,
            )
        elif method == 'netgraph':
            self.draw_with_netgraph(
                ax = ax,
                scale = scale,
                seed = seed,
                layout = layout,
                base_size = node_base_size,
                font_size = font_size,
                hide_target_labels = hide_target_labels,
                edge_width_scale = edge_width_scale,
            )

        # add color bar and legend
        self.set_legend(
            legend_size = legend_size,
            method = method
        )
        self.set_color_bar(
            ax = ax,
            shrink = colorbar_shrink,
            font_size = font_size
        )
        fig.tight_layout()

        if save_path is not None:
            self.save(save_path)

    # Netgraph plot method
    def draw_with_netgraph(self,
                           ax = plt.gca(),
                           base_size:int = 2,
                           scale:int = 1,
                           seed:int = 1936,
                           layout:str = 'spring',
                           font_size:int = 5,
                           hide_target_labels:bool = False,
                           edge_width_scale:float = 0.1,):
        node_size, node_color, node_alhpa, node_labels = self.get_node_info(
            base_size = base_size,
            color_type = 'rgba',
            hide_target_labels = hide_target_labels,
        )
        edge_width, edge_style, edge_color, edge_alpha = self.get_edge_info(
            width_scale = edge_width_scale,
            color_type = 'rgba',
        )
        node_shape = {node:'o' for node in self.graph.nodes}
        for node, data in self.graph.nodes(data = True):
            if data['type'] == 'TF':
                node_shape[node] = 'd'
        plot = Graph(
            graph = self.graph,
            node_layout = layout,
            node_size = node_size,
            node_color = node_color,
            node_shape = node_shape,
            node_edge_color = node_color,
            node_label_fontdict = {'size':font_size},
            node_alpha = node_alhpa,
            node_labels = node_labels,
            edge_width = edge_width,
            edge_cmap = self.edge_cmap,
            edge_color = edge_color,
            edge_alpha = edge_alpha,
            arrows = True,
        )

    # Networkx plot method
    def draw_with_networkx(self,
                           ax = plt.gca(),
                           base_size:int = 600,
                           scale:int = 1,
                           seed:int = 1914,
                           layout:str = 'spring',
                           font_size:int = 5,
                           hide_target_labels:bool = False,
                           edge_width_scale:float = 1.0,):
        node_size, node_color, node_alhpa, node_labels = self.get_node_info(
            base_size = base_size,
            hide_target_labels = hide_target_labels,
        )
        edge_width, edge_style, edge_color, edge_alpha = self.get_edge_info(
            width_scale = edge_width_scale,
        )
        node_size = [node_size[node] for node in self.graph.nodes]
        node_color = [node_color[node] for node in self.graph.nodes]
        edge_width = [edge_width[(u,v)] for (u,v) in self.graph.edges]
        edge_style = [edge_style[(u,v)] for (u,v) in self.graph.edges]
        edge_color = [edge_color[(u,v)] for (u,v) in self.graph.edges]
        edge_alpha = [edge_alpha[(u,v)] for (u,v) in self.graph.edges]

        # specify layout
        if layout == 'circular':
            pos = nx.circular_layout(self.graph, scale = scale,)
        elif layout == 'spring':
            pos = nx.spring_layout(
                self.graph,
                scale = scale,
                seed = seed,
                k = max(node_size),
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

        # Draw Nodes, Labels, and Edges
        nodes = nx.draw_networkx_nodes(
            G = self.graph,
            pos = pos,
            cmap = self.node_cmap,
            node_size = node_size,
            node_color = node_color,
            alpha = node_alhpa,
        )
        labels = nx.draw_networkx_labels(
            G = self.graph,
            pos = pos,
            labels = node_labels,
            font_size = font_size,
            clip_on = True,
        )
        edges = nx.draw_networkx_edges(
            G = self.graph,
            pos = pos,
            node_size = node_size,
            arrowstyle = "-|>",
            arrowsize = 20,
            edge_color = edge_color,
            edge_cmap = self.edge_cmap,
            width = edge_width,
            style = edge_style,
        )

        # set alpha value for each edge
        for i in range(self.graph.number_of_edges()):
            edges[i].set_alpha(edge_alpha[i])

    # Get Edge Information
    def get_edge_info(self,
                      width_scale = 1,
                      color_type = 'int',
                      return_type = 'dict'):
        edge_width = dict()
        edge_style = dict()
        edge_color = dict()
        edge_alpha = dict()
        for (source, target, data) in self.graph.edges(data = True):
            key = (source, target)
            # set edge width
            edge_width[key] = abs(data['weight']) * width_scale
            # set info by type
            if data['type'] == TYPES[2]:
                edge_style[key] = ':'
            else:
                edge_style[key] = '-'
            # set up alpha value
            edge_alpha[key] = min(1, abs(data['weight']))
            # set color
            if data['weight'] < 0:
                edge_color[key] = -1
            else:
                edge_color[key] = 1
            # change color to rgba format if specified
            if color_type == 'rgba':
                edge_color[key] = self.edge_scalar_map.to_rgba(edge_color[key])

        return edge_width, edge_style, edge_color, edge_alpha


    # Get Node information
    def get_node_info(self,
                      base_size = 800,
                      hide_target_labels = False,
                      color_type = 'int',
                      return_type = 'dict'):
        node_size = dict()
        node_color = dict()
        node_alhpa = 0.8
        node_labels = {n:n for n in self.graph}
        for node, data in self.graph.nodes(data = True):
            factor = 1
            # target_num = len([i for i in self.graph.successors(node)])
            target_num = len(data['target'])
            if target_num > 0:
                node_color[node] = 1
                # increase node size according to gene's reg power
                if target_num > 10:
                    factor = math.log10(target_num) * 2
            else:
                node_color[node] = 0
                if hide_target_labels:
                    del node_labels[node]
            # change color to rgba format if specified
            if color_type == 'rgba':
                node_color[node]=self.node_scalar_map.to_rgba(node_color[node])
            size = base_size * factor
            node_size[node] = size

        return node_size, node_color, node_alhpa, node_labels

    # just as named
    def get_weight(self, correlations):
        if self.graph_type == 'class1' or self.graph_type == '1':
            weight = correlations['class1']
        elif self.graph_type == 'class2' or self.graph_type == '2':
            weight = correlations['class2']
        elif self.graph_type == 'all':
            weight = abs(correlations['class1']) - abs(correlations['class2'])
        return weight

    # Set up a color bar with fixed scale from -1.0 to 1.0
    def set_color_bar(self, ax, shrink = 1, font_size = 10):
        cbar = plt.colorbar(self.edge_scalar_map, ax = ax, shrink = shrink)
        if self.graph_type == 'all':
            labels = cbar.ax.get_yticklabels()
            labels[0] = 'Stronger in Class2'
            # only set mid tick label when it's odd length
            if (len(labels) % 2) == 1:
                labels[int(len(labels) / 2)] = 'No Difference'
            labels[-1] = 'Stronger in Class1'
            cbar.set_ticklabels(labels)
            cbar.ax.tick_params(labelsize = font_size)
        cbar.ax.set_ylabel(
            'Gene Expression Correlation',
            fontsize = font_size,
            fontweight = 'bold',
            labelpad = 12.0,
            rotation = 270
        )

    # Set Up Legend
    def set_legend(self, legend_size, method):
        plt.scatter(
            [],[],
            s = legend_size * 10,
            marker = 'o',
            c = [self.node_cmap(0)],
            label = 'Regulatory Target'
        )
        plt.scatter(
            [],[],
            s = legend_size * 10,
            marker = 'o',
            c = [self.node_cmap(1)],
            label = 'Regulatory Source'
        )
        if method == 'netgraph':
            plt.scatter(
                [],[],
                s = legend_size * 10,
                marker = 'd',
                c = ['black'],
                label = 'TF'
            )
            plt.scatter(
                [],[],
                s = legend_size * 10,
                marker = 'o',
                c = ['black'],
                label = 'Gene'
            )
        elif method == 'networkx':
            plt.plot(
                [], [],
                linestyle = 'dashed',
                c = 'black',
                label = 'Bridge GRP'
            )
            plt.plot(
                [], [],
                linestyle = 'solid',
                c = 'black',
                label = 'Key GRP'
            )
        plt.legend(loc = 1, prop = {'size': legend_size})

    # save the plot. PDF by default
    def save(self, path:str = None, format:str = 'pdf'):
        plt.savefig(path, format = format)
        plt.close()

    # show the interactive graph
    def show(self):
        plt.show()


if __name__ == '__main__':
    header = 'liverCCl4/hsc_pf_a6w/'
    for i in range(1):
        folder_path = header + 'run_' + str(i) + '/'
        atlas_path = folder_path + 'key_atlas.js'
        a = Plot_Regulon(
            file_path = atlas_path,
            regulon_id = 'regulon_0',
            hide_bridge = False,
            graph_type = 'class2',
            root_gene = 'HAND2',
            # impact_depth = 1,
        )
        a.draw(save_path = folder_path + 'ultimate_useless_mess.pdf',)
