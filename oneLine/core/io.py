import json
import shutil
from os.path import isfile, isdir


def ignore(*ignorance):
    """
    A built-in ignore_patterns function.
    """
    return shutil.ignore_patterns(*ignorance)


def copy(source: str, to: str, ignore=None):
    """
    Copy a file or a folder to destination.
    You can use io.ignore to ignore the particular files from copy process.

    For example: io.copy('input', 'new', ignore=io.ignore('*.csv'))
    """
    if isdir(source):
        if ignore:
            shutil.copytree(source, to, ignore=ignore)
        else:
            shutil.copytree(source, to)
    elif isfile(source) and isfile(to):
        shutil.copyfile(source, to)
    else:
        raise TypeError(f"Sorry, the source is { 'directory' if isdir(source) else 'file' } but destination is { 'directory' if isdir(to) else 'file' }")


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


def compress(source: str = None, to: str = None, _format: str = None):
    """
    Compress files to a file.
    """
    shutil.make_archive(to, _format, root_dir=source)


def extract(source: str = None, to: str = None):
    """
    Extract a compress file.
    """
    _format = source.split('.')[-1]
    shutil.unpack_archive(filename=source, extract_dir=to, format=_format)


def read(path: str, mode: str = 'r+', readlines: bool = False, encoding: str = 'utf-8'):
    """
    That's a simple and fast function for reading a file.
    It will return a string that contains all the content when readlines = False.
    And will return a list if readlines = True.

    WARNING: This is for only small size of files, which means that the bigger size will lead to a memory exception.
    """
    with open(path, mode, encoding=encoding) as f:
        if readlines:
            return f.readlines()
        else:
            return f.read()


def write(content, path, mode: str = 'w+'):
    """
    Write string or list to file.

    If the content is a list, it will be written with writelines mode.
    And also the write mode will be used if the content is string.
    """
    with open(path, mode) as f:
        if isinstance(content, list):
            f.writelines(content)
        elif isinstance(content, str):
            f.write(content)


def load_json(path: str):
    """
    Load the json file, which will return a dictionary.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(_dict: dict, path: str, indent=4, ensure_ascii=False):
    """
    Save a dictionary to json file
    """
    with open(path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(_dict, indent=indent, ensure_ascii=ensure_ascii))
