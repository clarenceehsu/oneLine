"""
The io module supplies some basic IO methods with a more comprehensive auto detection. It's easy to use
without learning cost.

This module are based on the Python standard library shutil.
"""

import os
import shutil
from os.path import isfile, isdir, exists


def ignore(*ignorance):
    """
    A built-in ignore_patterns function.
    """

    return shutil.ignore_patterns(*ignorance)


def copy(source: str, to: str, type=None, ignore=None, follow_symlinks=True):
    """
    Copy a file or a folder to destination, which can be a folder or a file (automatically detect if type
    isn't specified), and you can use io.ignore to ignore the particular files from copy process.
    ** For preventing conflicts, type can be specified when needed.

    For example: io.copy('input', 'new', ignore=io.ignore('*.csv'))
    """
    if type == "file":
        shutil.copy(source, to, follow_symlinks=follow_symlinks)
    elif type == "folder" or isdir(source):
        if ignore:
            shutil.copytree(source, to, ignore=ignore)
        else:
            shutil.copytree(source, to)
    elif isfile(source):
        shutil.copy(source, to, follow_symlinks=follow_symlinks)
    else:
        raise TypeError(f"The file or directory *{ source }* is not exist.")


def delete(source):
    """
    Delete the file or folder.
    """

    shutil.rmtree(source)


def move(source: str = None, to: str = None):
    """
    Move file or folder.
    """

    shutil.move(source, to)


def mkdir(path: str):
    """
    Create a new directory.
    """

    if exists(path):
        raise AttributeError(f"The path *{ path }* is exist.")
    else:
        os.mkdir(path)


def read(path: str, mode: str = 'r', encoding: str = 'utf-8', errors=None):
    """
    That's a simple and fast read function for reading a file.

    WARNING: This is for only small size of files, which means that the bigger size will lead to a memory
    exception.
    """

    with open(path, mode, encoding=encoding, errors=errors) as f:
        return f.read()


def readlines(path: str, mode: str = 'r', encoding: str = 'utf-8', errors=None, ):
    """
    That's a simple and fast readlines function for reading a file.

    WARNING: This is for only small size of files, which means that the bigger size will lead to a memory
    exception.
    """

    with open(path, mode, encoding=encoding, errors=errors) as f:
        return f.readlines()


def write(content, path, mode: str = 'w+', encoding: str = 'utf-8', errors=None):
    """
    Write string or list to file.

    If the content is a list, it will be written with writelines mode.
    And also the write mode will be used if the content is string.
    """

    with open(path, mode, encoding=encoding, errors=errors) as f:
        if isinstance(content, list):
            f.writelines(content)
        elif isinstance(content, str):
            f.write(content)
