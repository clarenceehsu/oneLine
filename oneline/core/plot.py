import sys
import traceback
import numpy as np
from numpy.lib.arraysetops import isin
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class Plot:
    """
    This is a plot module for data visualization.
    """

    def _raise_plot_value_error(self, s: str):
        raise ValueError(f'The parameter {s} missed.')

    def __init__(self, *args):
        self = pd.DataFrame()

    def line_plot(self, x: str = None,
                  y = None,
                  figsize: list = None,
                  title: str = None,
                  xlabel: str = None,
                  ylabel: str = None,
                  smooth: bool = False,
                  kind: str = 'cubic',
                  interval: int = 50,
                  legend_loc: str = 'upper left',
                  show: bool = True):
        """
        It's a fast plot function to generate graph rapidly.

        :param x: the x
        :param y: the y
        :param figsize: The size of figure.
        :param title: the title of plot
        :param xlabel: label of x
        :param ylabel: label of y
        :param smooth: Set it True if the curve smoothing needed.
        :param kind: The method for smooth.
        :param interval: define the number for smooth function.
        :param legend_loc: The location of the legend in plot.
        :param show: plt.show will run if true
        """
        sns.set()

        # data x pre-configuration process
        # x will be used if x is specified, otherwise the default index would be used
        if x:
            x = self[x]
        else:
            x = list(self.index)

        # data y pre-configuration process
        # y can be a string or a list for multiple series plot
        # oneline.Plot use inter1d for 1D data smoothing
        if isinstance(y, str):
            y_val = self[y]
            if smooth:
                x_new = np.linspace(min(x), max(x), len(x) * interval)
                y_smooth = interp1d(x, y_val, kind=kind)
                plt.plot(x_new, y_smooth(x_new), label=y)
            else:
                plt.plot(x, y_val, label=y)
        elif isinstance(y, list):
            for name in y:
                y_val = self[name]
                if smooth:
                    x_new = np.linspace(min(x), max(x), len(x) * interval)
                    y_smooth = interp1d(x, y_val, kind=kind)
                    plt.plot(x_new, y_smooth(x_new), label=name)
                else:
                    plt.plot(x, y_val, label=name)

        # the default plot if no argument input
        else:
            for name in list(self.columns):
                y_val = self[name]
                if smooth:
                    x_new = np.linspace(min(x), max(x), len(x) * interval)
                    y_smooth = interp1d(x, y_val, kind='cubic')
                    plt.plot(x_new, y_smooth(x_new), label=name)
                else:
                    plt.plot(x, y_val, label=name)

        # plt configuration
        if figsize:
            plt.figure(figsize=figsize)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc=legend_loc)
        if show:
            plt.show()

        # return for advanced adjustment
        return plt

    def count_plot(self, x: str = None,
                   hue: str = None,
                   figsize: list = None,
                   show: bool = True):
        """
        Generate the count graph
        :param x: The x.
        :param hue: The hue.
        :param figsize: The size of figure.
        :param show: plt.show will run if true
        """
        if not x:
            self._raise_plot_value_error('x')
        sns.set()
        if figsize:
            plt.figure(figsize=figsize)
        sns.countplot(x=x, hue=hue, data=self)
        if show:
            plt.show()

    def corr_plot(self, parameters: list = None,
                  figsize: list = None,
                  annot: bool = True,
                  show: bool = True):
        """
        Generate the correction graph
        :param parameters: The parameters selected.
        :param figsize: The size of figure.
        :param annot: Display the annotation or not.
        :param show: plt.show will run if true
        """
        sns.set()
        if figsize:
            plt.figure(figsize=figsize)
        if parameters:
            sns.heatmap(self[parameters].corr(), annot=annot)
        else:
            sns.heatmap(self.corr(), annot=annot)
        if show:
            plt.show()

        return plt

    def comparing_variables(self, variable1: str = None,
                            variable2: str = None,
                            show: bool = True):
        try:
            if not variable1 or not variable2:
                self._raise_plot_value_error('variable1 and variable2')
            elif self[variable1].dtype == 'object' or self[variable2].dtype == 'object':
                raise TypeError('The type of parameters should be int or float, rather than object.')
        except Exception:
            traceback.print_exc(limit=1, file=sys.stdout)
            quit()

        print(self[[variable1, variable2]][self[variable2].isnull() == False].groupby([variable1],
                                                                                                as_index=False).mean().sort_values(
            by=variable2, ascending=False))
        sns.FacetGrid(self, col=variable2).map(sns.distplot, variable1)
        if show:
            plt.show()

        return plt


def line_plot(x: list = None,
              y: dict or list = None,
              figsize: list = None,
              title: str = None,
              xlabel: str = None,
              ylabel: str = None,
              smooth: bool = False,
              kind: str = 'cubic',
              interval: int = 50,
              show: bool = True):
    """
    That's a isolate line_plot for faster usage.
    """
    sns.set()
    if figsize:
        plt.figure(figsize=figsize)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if isinstance(y, dict):
        for (name, i) in y.items():
            if smooth:
                x_new = np.linspace(min(x), max(x), len(x) * interval)
                y_smooth = interp1d(x, i, kind=kind)
                if name:
                    plt.plot(x_new, y_smooth(x_new), label=name)
                else:
                    plt.plot(x_new, y_smooth(x_new))
            elif name:
                plt.plot(x, i, label=name)
            else:
                plt.plot(x, i)
            plt.legend(loc='upper left')
    elif isinstance(y, list):
        if smooth:
            x_new = np.linspace(min(x), max(x), len(x) * interval)
            y_smooth = interp1d(x, y, kind='cubic')
            plt.plot(x_new, y_smooth(x_new))
        else:
            plt.plot(x, y)
    if show:
        plt.show()

    return plt


def bar_plot(x: list = None,
             y: list = None,
             figsize: list = None,
             title: str = None,
             xlabel: str = None,
             ylabel: str = None,
             show: bool = True):
    """
    That's a isolate bar_plot for faster usage.
    """
    sns.set()
    if figsize:
        plt.figure(figsize=figsize)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    sns.barplot(x=x, y=y)
    if show:
        plt.show()

    return plt
