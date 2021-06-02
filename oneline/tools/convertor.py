"""
A set of methods that can make conversion related codes easier, faster and safer. This module is based on
built-in md5 and base64, and some of the methods can prevent memory overflow.
"""

from hashlib import md5
import base64


def file2md5(path: str, batch_byte=4096):
    """
    A md5 hash function that have the ability to prevented memory overflow.
    :param path: the path of file
    :param batch_byte: the size of every batch
    :return: a string of md5
    """
    md5_obj = md5()
    with open(path, 'rb') as f:
        while True:
            data = f.read(batch_byte)
            if not data:
                break
            md5_obj.update(data)

    return md5_obj.hexdigest()


def str2md5(s: str):
    """
    A sister function for fast using.
    :param s: the string which will be hashed
    :return: a string of md5
    """
    return md5(s.encode()).hexdigest()


def file2base64(path: str, batch_byte=4096):
    """
    A base64 hash function that have the ability to prevented memory overflow.
    :param path: the path of file
    :param batch_byte: the size of every batch
    :return: base64 bytes result
    """
    base64_list = bytes()
    with open(path, 'rb') as f:
        while True:
            data = f.read(batch_byte)
            if not data:
                break
            base64_list += base64.b64encode(data)

    return base64_list


def str2base64(s: str):
    """
    A sister function for fast using.
    :param s: the string which will be converted to base64
    :return: a string of based64
    """
    return base64.b64encode(s.encode())


def base642file(s: str, path: str):
    """
    Generate a file from base64 string.
    :param s: the base64 string
    :param path: the path of file
    :return: None
    """
    with open(path, "wb")as f:
        f.write(base64.b64decode(s))


def base642str(s: str):
    """
    Convert a base64 to string.
    :param s: the base64 string
    :return: a string
    """
    return base64.b64decode(s).decode()
