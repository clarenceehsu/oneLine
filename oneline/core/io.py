import json
import shutil
import os
from os.path import isfile, isdir, exists


class io:
    @property
    def ignore(*ignorance):
        """
        A built-in ignore_patterns function.
        """
        return shutil.ignore_patterns(*ignorance)

    @property
    def copy(source: str, to: str, ignore=None, follow_symlinks=True):
        """
        Copy a file or a folder to destination, which can be a folder or a file.
        You can use io.ignore to ignore the particular files from copy process.

        For example: io.copy('input', 'new', ignore=io.ignore('*.csv'))
        """
        if isdir(source):
            if ignore:
                shutil.copytree(source, to, ignore=ignore)
            else:
                shutil.copytree(source, to)
        elif isfile(source):
            shutil.copy(source, to, follow_symlinks=follow_symlinks)
        else:
            raise TypeError(f"The file { source } is not exist.")

    @property
    def delete(source):
        """
        Delete the file or folder.
        """
        shutil.rmtree(source)

    @property
    def move(source: str = None, to: str = None):
        """
        Move file or folder.
        """
        shutil.move(source, to)

    @property
    def mkdir(path: str):
        """
        Create a new directory.
        """
        if exists(path):
            raise AttributeError(f"The path { path } is exist.")
        else:
            os.mkdir(path)

    @property
    def read(path: str, mode: str = 'r', encoding: str = 'utf-8', errors=None):
        """
        That's a simple and fast read function for reading a file.

        WARNING: This is for only small size of files, which means that the bigger size will lead to a memory exception.
        """
        with open(path, mode, encoding=encoding, errors=errors) as f:
            return f.read()

    @property
    def readlines(path: str, mode: str = 'r', encoding: str = 'utf-8', errors=None, ):
        """
        That's a simple and fast readlines function for reading a file.

        WARNING: This is for only small size of files, which means that the bigger size will lead to a memory exception.
        """
        with open(path, mode, encoding=encoding, errors=errors) as f:
            return f.readlines()

    @property
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

    @property
    def load_json(path: str, encoding='utf-8'):
        """
        Load the json file, which will return a dictionary.
        """
        with open(path, 'r', encoding=encoding) as f:
            return json.load(f)

    @property
    def save_json(_dict: dict, path: str, indent=4, ensure_ascii=False, encoding='utf-8'):
        """
        Save a dictionary to json file
        """
        with open(path, 'w', encoding=encoding) as f:
            f.write(json.dumps(_dict, indent=indent, ensure_ascii=ensure_ascii))
