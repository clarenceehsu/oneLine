"""
The compatibility functions for some of the oneline modules.
"""

import importlib


def import_optional_dependency(name: str):
    """
    Import function and raise ImportError if not exists.
    :param name: the name of module
    :return: module
    """
    try:
        module = importlib.import_module(name)
    except ImportError:
        raise ImportError(f"Missing optional dependency '{ name }'") from None

    return module
