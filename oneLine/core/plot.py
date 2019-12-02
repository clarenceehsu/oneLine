from ..modules.data import *

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d


class Plot:
    """
    This is a plot module for data visualization.
    """
    def __init__(self, *args):
        self.data = pd.DataFrame()

    def fast_plot(self, x='', y=[], figsize=[], title='', xlabel='', ylabel='', smooth=False, insert_num=50):
        """
        It's a fast plot function to generate graph rapidly.
        :param figsize: The size of figure.
        :param smooth: Set it True if the curve smoothing needed.
        """
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
                    y_smooth = interp1d(x, y_val, kind='cubic')
                    plt.plot(x_new, y_smooth(x_new), label=name)
                else:
                    plt.plot(x, y_val, label=name)
                plt.legend(loc='upper left')
        else:
            for name in list(self.data.columns):
                y_val = self.data[name]
                if smooth:
                    x_new = np.linspace(min(x), max(x), len(x) * 50)
                    y_smooth = interp1d(x, y_val, kind='cubic')
                    plt.plot(x_new, y_smooth(x_new), label=name)
                else:
                    plt.plot(x, y_val, label=name)
                plt.legend(loc='upper left')

        plt.show()

    def corr_plot(self, parameters=[], figsize=[], annot=True):
        """
        Generate the correction graph
        :param parameters: The parameters selected.
        :param figsize: The size of figure.
        :param annot: Display the annotation or not.
        """
        sns.set()
        if figsize:
            plt.figure(figsize=figsize)
        if parameters:
            sns.heatmap(self.data[parameters].corr(), annot=annot)
        else:
            sns.heatmap(self.data.corr(), annot=annot)
        plt.show()

    def comparing_variables(self, variable1, variable2):
        print(self.data[[variable1, variable2]][self.data[variable2].isnull() == False].groupby([variable1], as_index=False).mean().sort_values(by=variable2, ascending=False))
        sns.FacetGrid(self.data, col=variable2).map(sns.distplot, variable1)
        plt.show()

    def counting_values(self, variable1, variable2):
        return self.data[[variable1, variable2]][self.data[variable2].isnull() == False].groupby([variable1], as_index=False).mean().sort_values(by=variable2, ascending=False)


def fast_plot(x, y, figsize=[], title='', xlabel='', ylabel='', smooth=False, insert_num=50):
    """
    That's a isolate fast_plot for faster usage.
    """
    sns.set()
    if figsize:
        plt.figure(figsize=figsize)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    for (name, i) in y.items():
        if smooth:
            x_new = np.linspace(min(x), max(x), len(x) * insert_num)
            y_smooth = interp1d(x, i, kind='cubic')
            if name:
                plt.plot(x_new, y_smooth(x_new), label=name)
            else:
                plt.plot(x_new, y_smooth(x_new))
        elif name:
            plt.plot(x, i, label=name)
        else:
            plt.plot(x, i)
        plt.legend(loc='upper left')

    plt.show()
