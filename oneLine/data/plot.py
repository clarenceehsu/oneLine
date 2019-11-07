from ..modules.data import *

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import spline


class Plot:
    def __init__(self, *args):
        self.data = pd.DataFrame()

    def fast_plot(self, x='', y=[], figsize=[], title='', xlabel='', ylabel='', smooth=False):
        sns.set()
        if figsize:
            plt.figure(figsize=figsize)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if x:
            x = self.data[x]
        else:
            x = list(self.data.index)
        if y:
            for name in y:
                y_val = self.data[name]
                if smooth:
                    x_new = np.linspace(min(x), max(x), len(x) * 50)
                    y_smooth = spline(x, y_val, x_new)
                    plt.plot(x_new, y_smooth, label=name)
                else:
                    plt.plot(x, y_val, label=name)
                plt.legend(loc='upper left')
        else:
            for name in list(self.data.columns):
                y_val = self.data[name]
                if smooth:
                    x_new = np.linspace(min(x), max(x), len(x) * 50)
                    y_smooth = spline(x, y_val, x_new)
                    plt.plot(x_new, y_smooth, label=name)
                else:
                    plt.plot(x, y_val, label=name)
                plt.legend(loc='upper left')

        plt.show()

    def corr_plot(self, figsize=[], annot=True):
        sns.set()
        if figsize:
            plt.figure(figsize=figsize)
        sns.heatmap(self.data.corr(), annot=annot)
        plt.show()
