#!/usr/bin/env python3
"""
Ageas Reborn

author: jy, nkmtmsys
"""

import os
from sklearn.preprocessing import LabelEncoder



class Setup:
    """
    Storage database related setting variables
    """

    def __init__(self,
                database_path = None,
                database_type = 'gem_file',
                class1_path = 'CT1',
                class2_path = 'CT2',
                specie = 'mouse',
                sliding_window_size = None,
                sliding_window_stride = None):
        # Auto GEM folder finder
        if database_path is None:
            assert os.path.exists(class1_path) and os.path.exists(class2_path)
        elif class1_path is None or class2_path is None:
            if len(os.listdir(database_path)) != 2:
                raise DB_Error('Please specify classes for binary clf')
            else:
                class1_path = os.listdir(database_path)[0]
                class2_path = os.listdir(database_path)[1]

        # Initialization
        self.db_path = database_path
        self.type = database_type
        self.specie = specie
        self.sliding_window_size = sliding_window_size
        self.sliding_window_stride = sliding_window_stride
        # Get classes'correspond folder paths
        self.class1_path = self.__cast_path(class1_path)
        self.class2_path = self.__cast_path(class2_path)
        # Perform label encoding
        self.label_transformer = Label_Encode(class1_path, class2_path)
        self.label1 = self.label_transformer.get_label1()
        self.label2 = self.label_transformer.get_label2()

    # make path str for the input class based on data path and folder name
    def __cast_path(self, path):
        # no need to concat if path is already completed
        if self.db_path is None:
            return path
        elif path[0] == '/':
            return self.db_path + path
        else:
            return self.db_path + '/' + path



class Label_Encode:
    """
    Transform labels into ints
    """

    def __init__(self, class1_path, class2_path):
        # Initialization
        self.encoder = LabelEncoder().fit([class1_path, class2_path])
        self.transformed_labels = self.encoder.transform([class1_path,
                                                        class2_path])

    # Perform inverse_transform
    def getOriginLable(self, query):
        return list(self.encoder.inverse_transform(query))

    # As named
    def get_label1(self): return self.transformed_labels[0]
    def get_label2(self): return self.transformed_labels[1]
