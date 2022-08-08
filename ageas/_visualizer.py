#!/usr/bin/env python3
"""
Ageas Reborn

author: jy, nkmtmsys
"""
import math
import numpy as np
import networkx as nx
from warnings import warn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
from scipy.special import softmax
from netgraph import Graph
import ageas.tool.grn as grn
import ageas.tool.json as json



TYPES = ['Standard', 'Outer', 'Bridge', 'Mix']



class Plot_Regulon(object):
    """
    Visualize full or partial Regulon extracted with AGEAS.

    """

    def __init__(self,
                 depth:int = 1,
                 hide_bridge:bool = True,
                 plot_group:str = 'all',
                 regulon = None,
                 root_gene:str = None,
                 weight_thread:float = 0.0,
                ):
        """
        Initialize a new plot object.

        :param depth: <int Default = 1>
            If 'root_gene' is specified, borrowing concept of tree graph, this
            means how far the tree graph shall reach.

        :param hide_bridge: <bool Default = True>
            Whether hide non-AGEAS-extracted GRPs which can link key genes or
            not.

        :param plot_group: <str Default = 'all'>
            What kind of information to show for GRPs on plot.

            Supporting:

                'all': Which group of samples a GRP has higher correlation.

                'group1': GRP's correlation value calculated with samples in
                group1

                'group2': GRP's correlation value calculated with samples in
                group2

        :param regulon: <ageas.tool.grn.GRN Default = None>
            Which regulon to plot out.

        :param root_gene: <str Default = None>
            Specific gene of interest. Only GRPs being capable to link with this
            gene directly or indirectly will be visualized. Borrowing concept of
            tree graph, this will be the root to expand a tree.

        :param weight_thread: <float Default = 0.0>
            Minimun weight for a GRP to be kept on plot.
        """
        super(Plot_Regulon, self).__init__()
        self.plot_group = str(plot_group)

        if root_gene is not None:
            grps_to_plot = regulon.get_grps_from_gene(root_gene, depth)
        else:
            grps_to_plot = regulon.grps

        grps_to_plot = {
            k:None for k,v in grps_to_plot.items() if self.__check(
                v, weight_thread, hide_bridge
            )
        }
        assert len(grps_to_plot) > 0

        # now we make the graph
        self.graph = regulon.as_digraph(grp_ids = grps_to_plot.keys())

    def draw(self,
             colorbar_shrink:float = 0.25,
             edge_width_scale:float = 1.0,
             font_size:int = 10,
             figure_size:int = 20,
             hide_target_labels:bool = False,
             layout:str = 'spring',
             legend_pos:set = (1.05, 0.3),
             method:str = 'netgraph',
             node_base_size:int = 2,
             save_path:str = None,
             scale:int = 1,
             seed:int = 1936,
            ):
        """
        Draw the plot for selected regulon.

        :param colorbar_shrink:<float> Default = 0.25
            From the size having equal length with graph plot, what ratio
            should the color bar be shrinked to.

        :param edge_width_scale:<float> Default = 1.0

        :param font_size:<int> Default = 10
            Font size.

        :param figure_size:<int> Default = 20
            Figure size.

        :param hide_target_labels:<bool> Default = False

        :param layout:<str> Default = 'spring'

        :param legend_pos:<set> Default = (1.05, 0.3)
            Where the legend should be located.

        :param method:<str> Default = 'netgraph'
            What exact method to be used for plotting.

            Supporting:

                'netgraph':

                'networkx':

        :param node_base_size:<int> Default = 2

        :param save_path:<str> Default = None

        :param scale:<int> Default = 1

        :param seed:<int> Default = 1936


        """
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
            self._draw_with_networkx(
                ax = ax,
                scale = scale,
                layout = layout,
                base_size = node_base_size,
                font_size = font_size,
                hide_target_labels = hide_target_labels,
                edge_width_scale =edge_width_scale,
            )
        elif method == 'netgraph':
            self._draw_with_netgraph(
                ax = ax,
                scale = scale,
                seed = seed,
                layout = layout,
                base_size = node_base_size,
                font_size = font_size,
                hide_target_labels = hide_target_labels,
                edge_width_scale = edge_width_scale,
            )

        self._set_color_bar(
            ax = ax,
            shrink = colorbar_shrink,
            font_size = font_size
        )
        self._set_legend(
            ax = ax,
            font_size = font_size,
            legend_pos = legend_pos,
            method = method
        )
        fig.tight_layout()

        if save_path is not None:
            self.save(save_path)

    def save(self, path:str = None, format:str = 'pdf'):
        """
        Save the plot. PDF format by default.

        :param path:<str> Default = None

        :param format:<str> Default = 'pdf'
        """
        plt.savefig(path, format = format)
        plt.close()

    # check whether keep a GRP to plot or not
    def __check(self, grp, weight_thread, hide_bridge):
        grp.weight = self._get_weight(grp.correlations)
        if abs(grp.weight) >= weight_thread:
            return grp.type != TYPES[2] or not hide_bridge
        else:
            return False

    # Netgraph plot method
    def _draw_with_netgraph(self,
                            ax = plt.gca(),
                            base_size:int = 2,
                            scale:int = 1,
                            seed:int = 1936,
                            layout:str = 'spring',
                            font_size:int = 5,
                            hide_target_labels:bool = False,
                            edge_width_scale:float = 0.1,
                            ):
        node_size, node_color, node_alhpa, node_labels = self._get_node_info(
            base_size = base_size,
            color_type = 'rgba',
            hide_target_labels = hide_target_labels,
        )
        edge_width, edge_style, edge_color, edge_alpha = self._get_edge_info(
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
    def _draw_with_networkx(self,
                           ax = plt.gca(),
                           base_size:int = 600,
                           scale:int = 1,
                           seed:int = 1914,
                           layout:str = 'spring',
                           font_size:int = 5,
                           hide_target_labels:bool = False,
                           edge_width_scale:float = 1.0,
                          ):
        node_size, node_color, node_alhpa, node_labels = self._get_node_info(
            base_size = base_size,
            hide_target_labels = hide_target_labels,
        )
        edge_width, edge_style, edge_color, edge_alpha = self._get_edge_info(
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
                self.graph, scale = scale, seed = seed, k = max(node_size),
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
    def _get_edge_info(self,
                      width_scale = 1,
                      color_type = 'int',
                      return_type = 'dict'
                      ):
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
    def _get_node_info(self,
                       base_size = 800,
                       hide_target_labels = False,
                       color_type = 'int',
                       return_type = 'dict'
                       ):
        node_size = dict()
        node_color = dict()
        node_alhpa = 0.8
        node_labels = {k:v['symbol'] for k,v in self.graph.nodes.data()}
        for node, data in self.graph.nodes(data = True):
            factor = 1
            target_num = len([i for i in self.graph.successors(node)])
            # target_num = len(data['target'])
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
    def _get_weight(self, correlations):
        if self.plot_group == 'group1' or self.plot_group == '1':
            weight = correlations['group1']
        elif self.plot_group == 'group2' or self.plot_group == '2':
            weight = correlations['group2']
        elif self.plot_group == 'all':
            weight = abs(correlations['group1']) - abs(correlations['group2'])
        return weight

    # Set up a color bar with fixed scale from -1.0 to 1.0
    def _set_color_bar(self, ax, shrink = 1, font_size = 10):
        cbar = plt.colorbar(self.edge_scalar_map, ax = ax, shrink = shrink)
        if self.plot_group == 'all':
            cbar.set_ticks([-1,0,1])
            cbar.set_ticklabels(
                ['Stronger in group2', 'No Difference', 'Stronger in group1']
            )
            cbar.ax.tick_params(labelsize = font_size)
        cbar.ax.set_ylabel(
            'Gene Expression Correlation',
            fontsize = font_size,
            fontweight = 'bold',
            labelpad = 12.0,
            rotation = 270
        )

    # Set Up Legend
    def _set_legend(self, ax, font_size, legend_pos, method):
        ax.scatter(
            [],[],
            s = font_size * 10,
            marker = 'o',
            c = [self.node_cmap(0)],
            label = 'Regulatory Target'
        )
        ax.scatter(
            [],[],
            s = font_size * 10,
            marker = 'o',
            c = [self.node_cmap(1)],
            label = 'Regulatory Source'
        )
        if method == 'netgraph':
            ax.scatter(
                [],[],
                s = font_size * 10,
                marker = 'd',
                c = ['black'],
                label = 'TF'
            )
            ax.scatter(
                [],[],
                s = font_size * 10,
                marker = 'o',
                c = ['black'],
                label = 'Gene'
            )
        elif method == 'networkx':
            ax.plot(
                [], [],
                linestyle = 'dashed',
                c = 'black',
                label = 'Bridge GRP'
            )
            ax.plot(
                [], [],
                linestyle = 'solid',
                c = 'black',
                label = 'Key GRP'
            )
        ax.legend(
            bbox_to_anchor = (legend_pos[0], legend_pos[1]),
            prop = {'size': font_size}
        )

    # # show the interactive graph
    # def show(self):
    #     plt.show()


# if __name__ == '__main__':
#     i = 0
#     regulon = grn.GRN()
#
#     header = 'liverCCl4/hsc_pf_a6w/'
#     # header = 'mef_esc/'
#     # header = 'neural/'
#
#     folder_path = header + 'run_' + str(i) + '/'
#
#     # atlas_path = folder_path + 'key_atlas.js'
#     # regulon.load_dict(json.decode(atlas_path))
#
#     atlas_path = folder_path + 'full_atlas.js'
#     regulon.load_dict(json.decode(atlas_path)['regulon_0'])
#
#     a = Plot_Regulon(
#         regulon = regulon,
#         hide_bridge = False,
#         # plot_group = 'group2',
#         root_gene = 'ENSMUSG00000000247',
#         depth = 1,
#     )
#
#     a.draw(
#         # layout = 'circular',
#         figure_size = 20,
#         font_size = 15,
#         node_base_size = 2,
#         legend_pos = (1.23, 0.3),
#         edge_width_scale = 2.0,
#         # method = 'networkx',
#         save_path = folder_path + 'lhx2_plot.pdf',
#     )
