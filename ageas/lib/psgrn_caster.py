#!/usr/bin/env python3
"""
Ageas Reborn
Generate pseudo-cell GRNs (psGRNs) from GEMs

author: jy, nkmtmsys

ToDo:
__file_method not done at all
"""

import os
import statistics as sta
import ageas.tool as tool
import ageas.tool.grn as grn
import ageas.tool.gem as gem
import ageas.tool.json as json
from scipy.stats import pearsonr



class Make:
    """
    Make grns for gene expression datas
    """
    def __init__(self,
                database_info = None,
                std_value_thread = None,
                std_ratio_thread = None,
                correlation_thread = 0.2,
                gem_data = None,
                meta_grn = None,
                load_path = None):
        super(Make, self).__init__()
        # Initialize
        self.database_info = database_info
        self.std_value_thread = std_value_thread
        self.std_ratio_thread = std_ratio_thread
        self.correlation_thread = correlation_thread
        if self.correlation_thread is None: self.correlation_thread = 0
        # load in
        if load_path is not None:
            self.class1_psGRNs,self.class2_psGRNs= self.__load_psGRNs(load_path)
        # Make GRNs
        else:
            self.class1_psGRNs, self.class2_psGRNs = self.__make_psGRNs(
                gem_data = gem_data,
                meta_grn = meta_grn
            )

    # main controller to cast pseudo cell GRNs (psGRNs)
    def __make_psGRNs(self, gem_data, meta_grn):
        if gem_data is not None:
            class1_psGRNs = self.__loaded_gem_method(
                class_type = 'class1',
                gem = gem_data.class1,
                meta_grn = meta_grn
            )
            class2_psGRNs = self.__loaded_gem_method(
                class_type = 'class2',
                gem = gem_data.class2,
                meta_grn = meta_grn
            )
        elif self.database_info.type == 'gem_folders':
            class1_psGRNs = self.__folder_method(
                'class1',
                self.database_info.class1_path,
                meta_grn
            )
            class2_psGRNs = self.__folder_method(
                'class2',
                self.database_info.class2_path,
                meta_grn
            )
        elif self.database_info.type == 'gem_files':
            # need to revise here!
            class1_psGRNs = self.__file_method(
                'class1',
                self.database_info.class1_path,
                meta_grn
            )
            class2_psGRNs = self.__file_method(
                'class2',
                self.database_info.class2_path,
                meta_grn
            )
        else:
            raise tool.Error('psGRN Caster Error: Unsupported database type')
        return class1_psGRNs, class2_psGRNs

    # as named
    def __file_method(self, class_type, path, meta_grn):
        psGRNs = dict()
        print('psgrn_caster.py:class Make: need to do something here')
        return psGRNs

    # as named
    def __loaded_gem_method(self, class_type, gem, meta_grn):
        psGRNs = dict()
        sample_num = 0
        start = 0
        end = self.database_info.sliding_window_size
        # set stride
        if self.database_info.sliding_window_stride is not None:
            stride = self.database_info.sliding_window_stride
        else:
            stride = end
        # use sliding window techinque to set pseudo samples
        loop = True
        while loop:
            if start >= len(gem.columns):
                break
            elif end >= len(gem.columns):
                end = len(gem.columns)
                loop = False
            sample_id = 'sample' + str(sample_num)
            sample = gem.iloc[:, start:end]
            if meta_grn is not None:
                pseudo_sample = self.__process_sample_with_metaGRN(
                    class_type,
                    sample,
                    sample_id,
                    meta_grn
                )
            else:
                pseudo_sample = self.__process_sample_without_guidance(
                    class_type,
                    sample,
                    sample_id
                )
            # Save data into psGRNs
            psGRNs[sample_id] = pseudo_sample
            start += stride
            end += stride
            sample_num += 1
        return psGRNs

    # as named
    def __folder_method(self, class_type, path, meta_grn):
        data = self.__readin_folder(path)
        psGRNs = dict()
        for sample_id in data:
            if meta_grn is not None:
                pseudo_sample = self.__process_sample_with_metaGRN(
                    class_type,
                    data[sample_id],
                    path,
                    meta_grn
                )
            else:
                pseudo_sample = self.__process_sample_without_guidance(
                    class_type,
                    data[sample_id],
                    path
                )
            # Save data into psGRNs
            psGRNs[sample_id] = pseudo_sample
        return psGRNs

    # as named
    def __process_sample_with_metaGRN(self,
                                        class_type,
                                        gem,
                                        sample_id,
                                        meta_grn):
        pseudo_sample = grn.GRN(id = sample_id)
        for grp in meta_grn.grps:
            source_ID = meta_grn.grps[grp].regulatory_source
            target_ID = meta_grn.grps[grp].regulatory_target
            try:
                source_exp = list(gem.loc[[source_ID]].values[0])
                target_exp = list(gem.loc[[target_ID]].values[0])
            except:
                continue
            # No need to compute if one array is constant
            if len(set(source_exp)) == 1 or len(set(target_exp)) == 1:
                continue
            cor = pearsonr(source_exp, target_exp)[0]
            if abs(cor) > self.correlation_thread:
                if grp not in pseudo_sample.grps:
                    pseudo_sample.add_grp(
                        id = grp,
                        source = source_ID,
                        target = target_ID,
                        correlations = {class_type: cor}
                    )
                if source_ID not in pseudo_sample.genes:
                    pseudo_sample.genes[source_ID] = grn.Gene(
                        id = source_ID,
                        expression_mean = {
                            class_type: float(sta.mean(source_exp))
                        }
                    )
                if target_ID not in pseudo_sample.genes:
                    pseudo_sample.genes[target_ID] = grn.Gene(
                        id = target_ID,
                        expression_mean = {
                            class_type: float(sta.mean(target_exp))
                        }
                    )
        return pseudo_sample

    # Process data without guidance
    # May need to revise later
    def __process_sample_without_guidance(self, class_type, gem, sample_id):
        pseudo_sample = grn.GRN(id = sample_id)
        # Get source TF
        for source_ID in gem.index:
            # Get target gene
            for target_ID in gem.index:
                if source_ID == target_ID:
                    continue
                grp_id = grn.GRP(source_ID, target_ID).id
                if grp_id not in pseudo_sample.grps:
                    # No need to compute if one array is constant
                    source_exp = gem[source_ID]
                    target_exp = gem[target_ID]
                    if len(set(source_exp)) == 1 or len(set(target_exp)) == 1:
                        continue
                    cor = pearsonr(source_exp, target_exp)[0]
                    if abs(cor) > self.correlation_thread:
                        if grp not in pseudo_sample.grps:
                            pseudo_sample.add_grp(
                                id = grp_id,
                                source = source_ID,
                                target = target_ID,
                                correlations = {class_type: cor}
                            )
                        if source_ID not in pseudo_sample.genes:
                            pseudo_sample.genes[source_ID] = grn.Gene(
                                id = source_ID,
                                expression_mean = {
                                    class_type: float(sta.mean(source_exp))
                                }
                            )
                        if target_ID not in pseudo_sample.genes:
                            pseudo_sample.genes[target_ID] = grn.Gene(
                                id = target_ID,
                                expression_mean = {
                                    class_type: float(sta.mean(target_exp))
                                }
                            )
                    else:
                        pseudo_sample[grp_id] = None
        return pseudo_sample

    # Readin Gene Expression Matrices in given class path
    def __readin_folder(self, path):
        result = dict()
        for filename in os.listdir(path):
            filename = path + '/' + filename
            # read in GEM files
            temp = gem.Reader(filename, header = 0, index_col = 0)
            temp.STD_Filter(
                std_value_thread = self.std_value_thread,
                std_ratio_thread = self.std_ratio_thread
            )
            result[filename] = temp.data
        return result

    # as named
    def update_with_remove_list(self, remove_list):
        for sample in self.class1_psGRNs:
            for id in remove_list:
                if id in self.class1_psGRNs[sample].grps:
                    del self.class1_psGRNs[sample].grps[id]
        for sample in self.class2_psGRNs:
            for id in remove_list:
                if id in self.class2_psGRNs[sample].grps:
                    del self.class2_psGRNs[sample].grps[id]
        return

    # temporal psGRN saving method
    """ need to be revised later to save psGRNs file by file"""
    def save(self, save_path):
        json.encode(
            {
                'class1':{k:v.as_dict() for k,v in self.class1_psGRNs.items()},
                'class2':{k:v.as_dict() for k,v in self.class2_psGRNs.items()}
            },
            save_path
        )
        return

    # load in psGRNs from files
    """ need to be revised later with save_psGRNs"""
    def __load_psGRNs(self, load_path):
        data = json.decode(load_path)
        class1_psGRNs = dict()
        class2_psGRNs = dict()
        for k,v in data['class1'].items():
            temp = grn.GRN(id = k)
            temp.load_dict(dict = v)
            class1_psGRNs[k] = temp
        for k,v in data['class2'].items():
            temp = grn.GRN(id = k)
            temp.load_dict(dict = v)
            class2_psGRNs[k] = temp
        return class1_psGRNs, class2_psGRNs


    # OLD: Save GRN files as js.gz in new folder
    # def save_GRN(self, data, save_path):
    #     for sample in data:
    #         names = sample.strip().split('/')
    #         name = names[-1].split('.')[0] + '.js'
    #         path = '/'.join(names[:-3] + [save_path, names[-2], name])
    #         # Make dir if dir not exists
    #         folder = os.path.dirname(path)
    #         if not os.path.exists(folder):
    #             os.makedirs(folder)
    #         # Get GRN and save it
    #         grn = data[sample]
    #         json.encode(grn, out = path)
