import numpy as np
import seaborn as sns
from pandas import Series
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class OneSeries(Series):
    def __init__(self, data):
        super().__init__(data=data)

    @property
    def _one_data(self):
        from .onedata import OneData
        return OneData

    def summary(self, info: bool = True):
        sum_info = {}
        length = self.shape[0]
        if str(self.dtype)[:3] == "int":
            sum_info['Unique'] = len(self.unique())
            sum_info['Unique(%)'] = sum_info['Unique'] / length
            sum_info['Missing'] = self.isnull().sum()
            sum_info['Missing(%)'] = sum_info['Missing'] / length
            sum_info['Means'] = self.sum() / length
            sum_info['Minimum'] = self.min()
            sum_info['Maximum'] = self.max()
            sum_info['Zeros'] = (self == 0).sum()
            sum_info['Zeros(%)'] = sum_info['Zeros'] / length
            sum_info['Memory Size(KB)'] = self.memory_usage() / 1024

            if info:
                print('Unique: {}({:.2f}%)'.format(sum_info['Unique'], sum_info['Unique(%)']))
                print('Missing: {}({:.2f}%)'.format(sum_info['Missing'], sum_info['Missing(%)']))
                print('Zeros: {}({:.2f}%)'.format(sum_info['Zeros'], sum_info['Zeros(%)']))
                print('Means: {:.2f}'.format(sum_info['Means']))
                print('Minimum: {}'.format(sum_info['Minimum']))
                print('Maximum: {}'.format(sum_info['Maximum']))
                print('Memory Size: {:.1f}KB'.format(sum_info['Memory Size(KB)']))

        elif str(self.dtype) == "object":
            sum_info['Unique'] = len(self.unique())
            sum_info['Unique(%)'] = sum_info['Unique'] / length
            sum_info['Missing'] = self.isnull().sum()
            sum_info['Missing(%)'] = sum_info['Missing'] / length
            sum_info['Memory Size(KB)'] = self.memory_usage() / 1024

            if info:
                print('Unique: {}({:.2f}%)'.format(sum_info['Unique'], sum_info['Unique(%)']))
                print('Missing: {}({:.2f}%)'.format(sum_info['Missing'], sum_info['Missing(%)']))
                print('Memory Size: {:.2f}KB'.format(sum_info['Memory Size(KB)']))

        return self._one_data(sum_info, index=[0])

    def count_plot(self,
                   figsize: list = None,
                   title: str = None,
                   xlabel: str = None,
                   ylabel: str = None):
        """
        Generate the count graph.
        """
        sns.set()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if figsize:
            plt.figure(figsize=figsize)
        sns.countplot(x=self)
        plt.show()

    def line_plot(self,
                  figsize: list = None,
                  title: str = None,
                  xlabel: str = None,
                  ylabel: str = None,
                  smooth: bool = False,
                  insert_num: int = 50,
                  label_loc: str = 'upper left',
                  show: bool = True):
        """
        It's a fast plot function to generate graph rapidly.
        """
        sns.set()
        if figsize:
            plt.figure(figsize=figsize)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        x_val = list(range(len(self)))

        if smooth:
            x_new = np.linspace(min(x_val), max(x_val), len(x_val) * insert_num)
            y_smooth = interp1d(x_val, self, kind='cubic')
            plt.plot(y_smooth(x_new), x_new)
        else:
            plt.plot(x_val, self)
        if show:
            plt.show()

    def to_frame(self, name=None):
        if name is None:
            df = self._onedata(self)
        else:
            df = self._onedata({name: self})
        return df
