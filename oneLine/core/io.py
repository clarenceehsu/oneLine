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
        if folder:
            shutil.copytree(source, to, ignore=shutil.ignore_patterns(ignore))
        elif seq:
            for src, des in seq:
                shutil.copyfile(src, des)
        else:
            shutil.copyfile(source, to)

    @staticmethod
    def delete(source):
        """
        Delete the file or folder.
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
        if seq:
            for src, des in seq.items():
                shutil.move(src, des)
        else:
            shutil.move(source, to)

    @staticmethod
    def compress(source: str = None, to: str = None, seq: dict = None, format: str = None):
        """
        Compress files to a file.

        The seq is a dictionary for {source path : destination path} key-value pairs, which is for multiple compress uses.
        """
        if seq:
            for src, des in seq.items():
                shutil.make_archive(des, format, root_dir=src)
        else:
            shutil.make_archive(to, format, root_dir=source)

    @staticmethod
    def extract(source: str = None, to: str = None, seq: dict = None):
        """
        Extract a compress file.

        The seq is a dictionary for {source path : destination path} key-value pairs, which is for multiple extract uses.
        """
        if seq:
            for src, des in seq.items():
                format = source.split('.')[-1]
                shutil.unpack_archive(filename=src, extract_dir=des, format=format)
        else:
            format = source.split('.')[-1]
            shutil.unpack_archive(filename=source, extract_dir=to, format=format)

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
