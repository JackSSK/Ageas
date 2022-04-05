#!/usr/bin/env python3
"""
Ageas Reborn
Generate pseudo-cell GRNs (pcGRNs) from GEMs

author: jy, nkmtmsys

ToDo:
__file_method not done at all
gem_data method also need to be done <- higher priority
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
    def __init__(self,
                database_info,
                std_value_thread = None,
                std_ratio_thread = None,
                correlation_thread = 0.2,
                gem_data = None,
                grn_guidance = None,
                save_path = None):
        # Initialize
        self.database_info = database_info
        self.std_value_thread = std_value_thread
        self.std_ratio_thread = std_ratio_thread
        self.correlation_thread = correlation_thread
        # Make GRNs
        self.class1_pcGRNs, self.class2_pcGRNs= self.__make_pcGRNs(gem_data,
                                                                grn_guidance)
        # Save GRNs when asked
        if save_path is not None:
            self.saveGRN(self.class1_pcGRNs, save_path)
            self.saveGRN(self.class2_pcGRNs, save_path)

    # main controller to cast pseudo cell GRNs (pcGRNs)
    def __make_pcGRNs(self, gem_data, grn_guidance):
        if gem_data is not None:
            print('we loaded data before')
        elif self.database_info.type == 'gem_folder':
            class1_pcGRNs = self.__folder_method(self.database_info.class1_path,
                                                grn_guidance)
            class2_pcGRNs = self.__folder_method(self.database_info.class2_path,
                                                grn_guidance)
        elif self.database_info.type == 'gem_file':
            print('ToDo')
        else:
            print('Make an Error here')
        return class1_pcGRNs, class2_pcGRNs

    # as named
    def __file_method(self,):
        print('something here')

    # as named
    def __folder_method(self, path, grn_guidance):
        data = self.__readin_folder(path)
        pcGRNs = {}
        for sample in data:
            grn = {}
            if grn_guidance is not None:
                grn = self.__process_sample_with_guidance(data[sample],
                                                            grn_guidance)
            else:
                grn = self.__process_sample_without_guidance(data[sample])
            # Save data into pcGRNs
            pcGRNs[sample] = {pth:data
                            for pth,data in grn.items()
                            if data is not None}
        return pcGRNs

    # as named
    def __process_sample_with_guidance(self, gem, grn_guidance):
        grn = {}
        for grp in grn_guidance:
            source_ID = grn_guidance[grp]['regulatory_source']
            target_ID = grn_guidance[grp]['regulatory_target']
            try:
                source = list(gem.loc[[source_ID]].values[0])
                target = list(gem.loc[[target_ID]].values[0])
            except:
                continue
            # No need to compute if one array is constant
            if len(set(source)) == 1 or len(set(target)) == 1:
                continue
            cor = pearsonr(source, target)[0]
            if abs(cor) > self.correlation_thread:
                grn[grp] = {
                    'id': grp,
                    'regulatory_source': source_ID,
                    'regulatory_target': target_ID,
                    'correlation':cor
                }
        return grn

    # Process data without guidance
    # May need to revise later
    def __process_sample_without_guidance(self, gem):
        grn = {}
        # Get source TF
        for source_ID in gem.index:
            # Get target gene
            for target_ID in gem.index:
                if source_ID == target_ID:
                    continue
                grp = tool.Cast_GRP_ID(source_ID, target_ID)
                if grp not in grn:
                    # No need to compute if one array is constant
                    if len(set(gem[source_ID])) == 1:
                        continue
                    elif len(set(gem[target_ID])) == 1:
                        continue
                    cor = pearsonr(gem[source_ID],
                                    gem[target_ID])[0]
                    if abs(cor) > self.correlation_thread:
                        grn[grp] = {
                            'id': grp,
                            'regulatory_source': source_ID,
                            'regulatory_target': target_ID,
                            'correlation':cor
                        }
                    else:
                        grn[grp] = None
        return grn

    # Readin Gene Expression Matrices in given class path
    def __readin_folder(self, path):
        result = {}
        for filename in os.listdir(path):
            filename = path + '/' + filename
            # read in GEM files
            temp = gem.Reader(filename, header = 0, index_col = 0)
            temp.STD_Filter(std_value_thread = self.std_value_thread,
                            std_ratio_thread = self.std_ratio_thread)
            result[filename] = temp.data
        return result

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
