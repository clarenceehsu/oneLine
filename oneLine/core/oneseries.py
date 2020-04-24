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

    def __add__(self, other):
        if isinstance(other, OneSeries):
            return self._one_data({self.name: self, other.name: other})
        elif isinstance(other, self._one_data):
            other.insert(loc=0, column=self.name, value=self)
            return other


    def summary(self, info: bool = True):
        """
        Return a summary of the whole OneSeries dataset.
        the stats from scipy is used to calculate the Entropy.
        :param info: stay True if a display of information is required
        """
        from scipy import stats

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
            sum_info['Entropy'] = round(
                stats.entropy(self.value_counts(normalize=True), base=2), 2)
            sum_info['Memory Size(KB)'] = self.memory_usage() / 1024

            if info:
                print('Unique: {}({:.2f}%)'.format(sum_info['Unique'], sum_info['Unique(%)'] * 100))
                print('Missing: {}({:.2f}%)'.format(sum_info['Missing'], sum_info['Missing(%)'] * 100))
                print('Zeros: {}({:.2f}%)'.format(sum_info['Zeros'], sum_info['Zeros(%)'] * 100))
                print('Means: {:.2f}'.format(sum_info['Means']))
                print('Minimum: {}'.format(sum_info['Minimum']))
                print('Maximum: {}'.format(sum_info['Maximum']))
                print('Entropy: {}'.format(sum_info['Entropy']))
                print('Memory Size: {:.1f}KB'.format(sum_info['Memory Size(KB)']))

        elif str(self.dtype) == "object":
            sum_info['Unique'] = len(self.unique())
            sum_info['Unique(%)'] = sum_info['Unique'] / length
            sum_info['Missing'] = self.isnull().sum()
            sum_info['Missing(%)'] = sum_info['Missing'] / length
            sum_info['Entropy'] = round(
                stats.entropy(self.value_counts(normalize=True), base=2), 2)
            sum_info['Memory Size(KB)'] = self.memory_usage() / 1024

            if info:
                print('Unique: {}({:.2f}%)'.format(sum_info['Unique'], sum_info['Unique(%)'] * 100))
                print('Missing: {}({:.2f}%)'.format(sum_info['Missing'], sum_info['Missing(%)'] * 100))
                print('Entropy: {}'.format(sum_info['Entropy']))
                print('Memory Size: {:.2f}KB'.format(sum_info['Memory Size(KB)']))

        return self._one_data(sum_info, index=[0])

    def reverse(self, reset_index: bool = False):
        """
        Method for reversing the dataset
        :param reset_index: true for reset the index of series
        """
        if reset_index:
            return self.loc[::-1].reset_index(drop=True)
        else:
            return self.loc[::-1]

    def top(self, n: int = 1):
        """
        Return the top of the values.
        :param n: the number of value to return
        """
        return self.value_counts().nlargest(n=n)

    def bottom(self, n: int = 1):
        """
        Return the bottom of the values
        :param n: the number of value to return
        """
        return self.value_counts().nsmallest(n=n)

    # ========================= Plot ========================= #

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
