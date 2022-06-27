#!/usr/bin/env python3
"""
JSON file related tools

author: jy, nkmtmsys
"""
import re
import gzip
import json


def encode(data = None, out:str = 'out.js', indent:int = 4):
    """
    Encode data in json format file.

    Args:
        data = None

        out:str = 'out.js'

        indent:int = 4

    """
    if re.search(r'\.gz$', out):
        with gzip.open(out, 'w+') as output:
            output.write(json.dumps(data).encode('utf-8'))
    else:
        with open(out, 'w+') as output:
            json.dump(data, output, indent = indent)


def decode(path:str = None):
    """
    Decode data from JSON format file.

    Args:
        path:str = None
    """
    if re.search(r'\.gz$', path):
        with gzip.open(path, 'r') as json_file:
            data = json.load(json_file)
        return data
    else:
        with open(path, 'r') as json_file:
            data = json.load(json_file)
        return data
