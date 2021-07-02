"""
The load_file module supplies some basic IO methods with a more comprehensive auto detection. It's easy to use
without learning cost.

This module are based on the Python standard library shutil.
"""

import json


def load_json(path: str, encoding='utf-8'):
    """
    Load the json file, which will return a dictionary.
    """

    with open(path, 'r', encoding=encoding) as f:
        return json.load(f)


def save_json(_dict: dict, path: str, indent=4, ensure_ascii=False, encoding='utf-8'):
    """
    Save a dictionary to json file
    """

    with open(path, 'w', encoding=encoding) as f:
        f.write(json.dumps(_dict, indent=indent, ensure_ascii=ensure_ascii))
