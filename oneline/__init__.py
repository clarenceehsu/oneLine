"""
It's a personal code set for learning and researching, and it provides advanced encapsulation of some
libraries, designing to provide a simpler mode of operation and help people to simplify their codes
concisely.

WARNING: Since this library is a collection of code used for learning and analysis, the content is not
perfect for the time being.
"""

# Core
from .core import *

# MachineLearning
from .machinelearning import *

# Information
__version__ = '0.2.5'
__author__ = 'Zeesain Tsui'

# check the essential dependencies
hard_dependencies = ("numpy", "pandas", "matplotlib")
missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append(f"{dependency}: {e}")

if missing_dependencies:
    raise ImportError(
        "Unable to import required dependencies:\n" + "\n".join(missing_dependencies)
    )
del hard_dependencies, dependency, missing_dependencies
