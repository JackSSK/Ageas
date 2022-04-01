#!/usr/bin/env python3
"""
Ageas Reborn
Generate pseudo-cell GRNs (pcGRNs) from GEMs

author: jy, nkmtmsys
"""

import os
import ageas.tool.gem as gem
import ageas.tool.json as json
import ageas.tool as tool
from scipy.stats import pearsonr



class Make:
    """
    Make grns for gene expression datas
    """
    def __init__(self, database,
                std_value_thread = None,
                std_ratio_thread = None,
                correlation_thread = 0.2,
                grn_guidance = None,
                save_path = None):
        # Initialize
        self.std_value_thread = std_value_thread
        self.std_ratio_thread = std_ratio_thread
        # Make GRNs
        self.class1_pseudo_cGRNs = self.__make_pcGRNs(database.class1_path,
                                                        grn_guidance,
                                                        correlation_thread)
        self.class2_pseudo_cGRNs = self.__make_pcGRNs(database.class2_path,
                                                        grn_guidance,
                                                        correlation_thread)
        # Save GRNs when asked
        if save_path is not None:
            self.saveGRN(self.class1_pseudo_cGRNs, save_path)
            self.saveGRN(self.class2_pseudo_cGRNs, save_path)

    # Readin Gene Expression Matrices in given class path
    def __readin(self, path):
        result = {}
        for filename in os.listdir(path):
            filename = path + '/' + filename
            # read in GEM files
            temp = gem.Reader(filename, header = 0, index_col = 0)
            temp.STD_Filter(std_value_thread = self.std_value_thread,
                            std_ratio_thread = self.std_ratio_thread)
            result[filename] = temp.data
        return result

    # Iteratively call makeGRN function in all samples of given class
    def __make_pcGRNs(self, path, grn_guidance, correlation_thread):
        data = self.__readin(path)
        pcGRNs = {}
        for sample in data:
            grn = {}
            if grn_guidance is not None:
                for grp in grn_guidance:
                    source_ID = grn_guidance[grp]['sourceID']
                    target_ID = grn_guidance[grp]['targetID']
                    try:
                        source = list(data[sample].loc[[source_ID]].values[0])
                        target = list(data[sample].loc[[target_ID]].values[0])
                    except:
                        continue
                    # No need to compute if one array is constant
                    if len(set(source)) == 1 or len(set(target)) == 1: continue
                    cor = pearsonr(source, target)[0]
                    if abs(cor) > correlation_thread:
                        grn[grp] = {
                            'grp_ID': grp,
                            'sourceID': source_ID,
                            'targetID': target_ID,
                            'correlation':cor
                        }
            # Process data without guidance
            # May need to revise later
            else:
                # Get source TF
                for source_ID in data[sample].index:
                    # Get target gene
                    for target_ID in data[sample].index:
                        if source_ID == target_ID: continue
                        grp_ID = tool.Cast_GRP_ID(source_ID, target_ID)
                        if grp_ID not in grn:
                            # No need to compute if one array is constant
                            if len(set(data[sample][source_ID])) == 1:
                                continue
                            elif len(set(data[sample][target_ID])) == 1:
                                continue
                            cor = pearsonr(data[sample][source_ID],
                                            data[sample][target_ID])[0]
                            if abs(cor) > correlation_thread:
                                grn[grp_ID] = {
                                    'grp_ID': grp_ID,
                                    'sourceID': source_ID,
                                    'targetID': target_ID,
                                    'correlation':cor
                                }
                            else:
                                grn[grp_ID] = None
            # Save data into pcGRNs
            pcGRNs[sample] = {pth:data
                            for pth,data in grn.items()
                            if data is not None}
        return pcGRNs

    # Save GRN files as js.gz in new folder
    def save_GRN(self, data, save_path):
        for sample in data:
            names = sample.strip().split('/')
            name = names[-1].split('.')[0] + '.js'
            path = '/'.join(names[:-3] + [save_path, names[-2], name])
            # Make dir if dir not exists
            folder = os.path.dirname(path)
            if not os.path.exists(folder):
                os.makedirs(folder)
            # Get GRN and save it
            grn = data[sample]
            json.encode(grn, out = path)
