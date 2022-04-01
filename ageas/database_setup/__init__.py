#!/usr/bin/env python3
"""
Ageas Reborn

author: jy, nkmtmsys
"""

import os
from pkg_resources import resource_filename



class Error(Exception):
    """
    Database setting related error handling
    """
    pass


# Get resource filenames for specie specified data
def get_specie_path(path, specie):
    answer = resource_filename(path, '../data/'+ specie + '/')
    if not os.path.exists(answer): raise Error(specie, 'is not supported!')
    return answer
