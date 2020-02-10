import sys
import traceback
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class Plot:
    """
    This is a plot module for data visualization.
    """

    def _raise_plot_value_error(self, s: str):
        raise ValueError(f'The parameter { s } missed.')

    def __init__(self, *args):
        self.data = pd.DataFrame()

    def line_plot(self, x: str = None,
                        y: list = None,
                        figsize: list = None,
                        title: str = None,
                        xlabel: str = None,
                        ylabel: str = None,
                        smooth: bool = False,
                        insert_num: int = 50,
                        label_loc: str = 'upper left'):
        """
        It's a fast plot function to generate graph rapidly.
        :param figsize: The size of figure.
        :param smooth: Set it True if the curve smoothing needed.
        :param insert_num: define the number for smooth function.
        :param label_loc: The location of the labels in plot.
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
                    x_new = np.linspace(min(x), max(x), len(x) * insert_num)
                    y_smooth = interp1d(x, y_val, kind='cubic')
                    plt.plot(x_new, y_smooth(x_new), label=name)
                else:
                    plt.plot(x, y_val, label=name)
                plt.legend(loc=label_loc)
        else:
            for name in list(self.data.columns):
                y_val = self.data[name]
                if smooth:
                    x_new = np.linspace(min(x), max(x), len(x) * insert_num)
                    y_smooth = interp1d(x, y_val, kind='cubic')
                    plt.plot(x_new, y_smooth(x_new), label=name)
                else:
                    plt.plot(x, y_val, label=name)
                plt.legend(loc='upper left')

        plt.show()

    def count_plot(self, x: str = None,
                         hue: str = None,
                         figsize: list = None):
        """
        Generate the correction graph
        :param x: The x.
        :param hue: The hue.
        :param figsize: The size of figure.
        """
        try:
            if not x:
                self._raise_plot_value_error('x')
        except Exception:
            traceback.print_exc(limit=1, file=sys.stdout)
            quit()

        sns.set()
        if figsize:
            plt.figure(figsize=figsize)
        sns.countplot(x=x, hue=hue, data=self.data)
        plt.show()

    def corr_plot(self, parameters: list = None,
                        figsize: list = None,
                        annot: bool = True):
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

    def comparing_variables(self, variable1: str = None, variable2: str = None):
        try:
            if not variable1 or not variable2:
                self._raise_plot_value_error('variable1 and variable2')
        except Exception:
            traceback.print_exc(limit=1, file=sys.stdout)
            quit()

        print(self.data[[variable1, variable2]][self.data[variable2].isnull() == False].groupby([variable1], as_index=False).mean().sort_values(by=variable2, ascending=False))
        sns.FacetGrid(self.data, col=variable2).map(sns.distplot, variable1)
        plt.show()

    def counting_values(self, variable1: str = None, variable2: str = None):
        try:
            if not variable1 or not variable2:
                self._raise_plot_value_error('variable1 and variable2')
        except Exception:
            traceback.print_exc(limit=1, file=sys.stdout)
            quit()

        return self.data[[variable1, variable2]][self.data[variable2].isnull() == False].groupby([variable1], as_index=False).mean().sort_values(by=variable2, ascending=False)


def line_plot(x: list = None,
              y: dict = None,
              figsize: list = None,
              title: str = None,
              xlabel: str = None,
              ylabel: str = None,
              smooth: bool = False,
              insert_num: int = 50):
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
    elif isinstance(y, list):
        if smooth:
            x_new = np.linspace(min(x), max(x), len(x) * insert_num)
            y_smooth = interp1d(x, y, kind='cubic')
            plt.plot(x_new, y_smooth(x_new))
        else:
            plt.plot(x, i)

    plt.show()
