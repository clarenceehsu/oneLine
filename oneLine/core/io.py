import json
import shutil


class io:
    """
    That's a io class for file manufacture using.
    """
    @staticmethod
    def copy(source: str, to: str, seq: dict = None, folder: bool = False, ignore: list = None):
        """
        Copy a file or a folder to destination.

        The seq is a dictionary for {source path : destination path} key-value pairs, which is for multiple copy uses.
        """
        try:
            if folder:
                shutil.copytree(source, to, ignore=shutil.ignore_patterns(ignore))
            else:
                shutil.copyfile(source, to)
        except TypeError:
            for src, des in seq:
                shutil.copyfile(src, des)

    @staticmethod
    def delete(source):
        """
        Delete the file or folder, you can also input a list to delete files together.
        """
        if isinstance(source, str):
            shutil.rmtree(source)
        elif isinstance(source, list):
            for n in source:
                shutil.rmtree(n)

    @staticmethod
    def move(source: str = None, to: str = None, seq: dict = None):
        """
        Move file or folder.

        The seq is a dictionary for {source path : destination path} key-value pairs, which is for multiple move uses.
        """
        try:
            shutil.move(source, to)
        except TypeError:
            for src, des in seq.items():
                shutil.move(src, des)

    @staticmethod
    def compress(source: str = None, to: str = None, seq: dict = None, format: str = None):
        """
        Compress files to a file.

        The seq is a dictionary for {source path : destination path} key-value pairs, which is for multiple compress uses.
        """
        try:
            shutil.make_archive(to, format, root_dir=source)
        except TypeError:
            for src, des in seq.items():
                shutil.make_archive(des, format, root_dir=src)

    @staticmethod
    def extract(source: str = None, to: str = None, seq: dict = None):
        """
        Extract a compress file.

        The seq is a dictionary for {source path : destination path} key-value pairs, which is for multiple extract uses.
        """
        try:
            format = source.split('.')[-1]
            shutil.unpack_archive(filename=source, extract_dir=to, format=format)
        except TypeError:
            for src, des in seq.items():
                format = src.split('.')[-1]
                shutil.unpack_archive(filename=src, extract_dir=des, format=format)

    @staticmethod
    def read(path: str, mode: str = 'r+', readlines: bool = False):
        """
        Read a file.
        """
        with open(path, mode) as f:
            if readlines:
                return f.readlines()
            else:
                return f.read()

    @staticmethod
    def write(content, path, mode: str = 'w+'):
        """
        Write string or list to file.
        """
        with open(path, mode) as f:
            if isinstance(content, list):
                f.writelines(content)
            elif isinstance(content, str):
                f.write(content)

    @staticmethod
    def load_json(path: str):
        """
        Load the json file
        """
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def save_json(_dict: dict, path: str, indent=4, ensure_ascii=False):
        """
        Save a dictionary to json file
        """
        with open(path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(_dict, indent=indent, ensure_ascii=ensure_ascii))
