"""
The compatibility functions for some of the oneline modules.
"""

import sys
import platform
import importlib


def import_optional_dependency(name: str, error_raise: bool = True):
    """
    Import function and raise ImportError if not exists.

    :param name: the name of module
    :param error_raise: raise the import error
    :return: module
    """

    try:
        module = importlib.import_module(name)
    except ImportError:
        if error_raise:
            raise ImportError(f"Missing optional dependency '{ name }'") from None
        else:
            return None

    return module


def is_platform_little_endian():
    """
    Checking if the running platform is little endian.

    :return: bool
    """

    return sys.byteorder == "little"


def is_platform_windows():
    """
    Checking if the running platform is windows.

    :return: bool
    """

    return sys.platform in ["win32", "cygwin"]


def is_platform_linux():
    """
    Checking if the running platform is linux.

    :return: bool
    """

    return sys.platform == "linux"


def is_platform_mac():
    """
    Checking if the running platform is mac.

    :return: bool
    """

    return sys.platform == "darwin"


def is_platform_arm():
    """
    Checking if he running platform use ARM architecture.

    :return: bool
    """

    return platform.machine() in ("arm64", "aarch64")
