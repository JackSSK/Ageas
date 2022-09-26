#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Copyright (C) 2022 Jack Yu <gyu17@jh.edu>
# Author: Jack Yu <gyu17@jh.edu> <Shenzhen Mozhou Technology Co., Ltd.>
# Author: Masayoshi Nakamoto <nkmtmsys@gmail.com> <Shenzhen Mozhou Technology Co., Ltd.>

# This program is free software
# You can redistribute it and/or modify it
# under the terms of the GNU General Public License
# as published by the Free Software Foundation;
# either version 3 of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""
AGEAS
========
Summary
-------
AGEAS (AutoML-based Genetics fEatrue extrAction System)
is to find key genetics factors, including genes and regulatory pathways,
in determining cellular phenotype.

For more information, please visit our GitHub repo:
https://github.com/JackSSK/Ageas/
--------

"""

__version__ = "0.0.1a5"
__author__ = "JackSSK"
__author__ = "nkmtmsys"
__email__ = "gyu17@jh.edu"

from ._main import (
    Launch
)

from ._unit import (
    Unit
)

from ._visualizer import (
    Plot
)

from ._psgrn import (
    Data_Preprocess
)

from .test import (
    Test
)

__all__ = [
    'Test',
    'Data_Preprocess',
    'Launch',
    'Unit',
    'Plot',
]
