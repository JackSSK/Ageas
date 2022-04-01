#!/usr/bin/env python3
"""
Ageas Reborn

author: jy, nkmtmsys
"""

import re
import ageas.tool as tool
import ageas.tool.gem as gem
import ageas.tool.json as json
import ageas.tool.grtd as grtd
import ageas.operator as operator
import ageas.tool.transfac as transfac
import ageas.database_setup.binary_class as binary_db
from ageas.lib.deg_finder import Find
from ageas.database_setup import get_specie_path



class Load:
    """
    Load in GEM data sets
    """
    def __init__(self,
                database_path = None,
                database_type = 'gem_folder',
                class1_path = None,
                class2_path = None,
                specie = 'mouse',
                facNameType = 'gn',
                mww_thread = 0.05,
                log2fc_thread = 0.1,
                stdevThread = 100,
                stdRatio = None):
        # Load TF databases based on specie
        specie_path = get_specie_path(__name__, specie)
        # Load TRANSFAC databases
        self.tf_list = transfac.Reader(specie_path + 'Tranfac201803_MotifTFsF.txt',
                                        facNameType).tfs
        # Load GRTD database
        self.interactions = grtd.Processor(specie_path,
                                            facNameType,
                                            path = 'wholeGene.js.gz')
        # Set up database
        self.db = binary_db.Setup(database_path,
                                    class1_path,
                                    class2_path,
                                    database_type)
        if self.db.type == 'gem_folder':
            class1, class2 = self.__process_gem_folder(stdevThread, stdRatio)
        elif self.db.type == 'gem_file':
            class1, class2 = self.__process_gem_file(stdevThread, stdRatio)
        # Distribuition Filter if threshod is specified
        if mww_thread is not None or log2fc_thread is not None:
            self.genes = Find(class1,
                                class2,
                                mww_thread = mww_thread,
                                log2fc_thread = log2fc_thread).degs
            self.class1 = class1.loc[class1.index.intersection(self.genes)]
            self.class2 = class2.loc[class2.index.intersection(self.genes)]
        else:
            self.genes = class1.index.union(class2.index)
            self.class1 = class1
            self.class2 = class2


    # Process in expression matrix file (dataframe) scenario
    def __process_gem_file(self, stdevThread, stdRatio):
        class1 = self.__read_df(self.db.class1_path, stdevThread, stdRatio)
        class2 = self.__read_df(self.db.class2_path, stdevThread, stdRatio)
        return class1, class2

    # Read in gem file
    def __read_df(self, path, stdevThread, stdRatio):
        # Decide which seperation mark to use
        if re.search(r'csv', path): sep = ','
        elif re.search(r'txt', path): sep = '\t'
        else: raise operator.Error('Unsupported File Type: ', path)
        # Decide which compression method to use
        if re.search(r'.gz', path): compression = 'gzip'
        else: compression = 'infer'
        df = pd.read_csv(path,
                        sep = sep,
                        compression = compression,
                        header = 0,
                        index_col = 0)
        return tool.STD_Filter(df, stdevThread, stdRatio)

    # Process in Database scenario
    def __process_gem_folder(self, stdevThread, stdRatio):
        class1 = gem.Folder(self.db.class1_path).combine(
                                                std_value_thread = stdevThread,
                                                std_ratio_thread = stdRatio
                                                )
        class2 = gem.Folder(self.db.class2_path).combine(
                                                std_value_thread = stdevThread,
                                                std_ratio_thread = stdRatio
                                                )
        return class1, class2
