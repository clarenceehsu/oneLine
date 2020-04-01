from .io import io
from .onedata import OneData
from .plot import line_plot, bar_plot


def test_enviorment():
    try:
        import torch
    except ImportError:
        raise ImportError('Module *torch* is not exist, and the Machine Learning is unavailable.')
    try:
        import panda
    except ImportError:
        raise ImportError('Module *pandas* is not exist, and the core may not work.')
    try:
        import scipy
    except ImportError:
        raise ImportError('Module *scipy* is not exist, and the core may not work.')
    try:
        import seaborn
    except ImportError:
        raise ImportError('Module *seaborn* is not exist, and the core may not work.')
