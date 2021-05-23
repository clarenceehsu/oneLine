import numpy as np
import seaborn as sns
import pandas as pd
from pandas import Series
from pandas import DataFrame
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from .plot import Plot


class OneSeries(Series, Plot):

    def __init__(self, data):
        super().__init__(data=data)

    @property
    def _one_data(self):
        from .onedata import OneData
        return OneData

    def r_append(self, other):
        """
        Append another OneSeries data to the right of this series and return a new OneData.
        :param other: the other OneSeries
        :return: OneData
        """
        if isinstance(other, OneSeries):
            return self._one_data({self.name: self, other.name: other})
        elif isinstance(other, self._one_data) or isinstance(other, DataFrame):
            return self._one_data(pd.concat([self, other], axis=1))

    def l_append(self, other):
        """
        Append another OneSeries data to the left of this series and return a new OneData.
        :param other: the other OneSeries
        :return: OneData
        """
        if isinstance(other, OneSeries):
            return self._one_data({other.name: other, self.name: self})
        elif isinstance(other, self._one_data) or isinstance(other, DataFrame):
            return self._one_data(pd.concat([other, self], axis=1))

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

    def shuffle(self, reset_index: bool = False, random_seed: int = None):
        """
        A non-inplace shuffle method.
        :param reset_index: reset the index if it sets True
        :param random_seed: the random seed of shuffle
        :return: OneSeries
        """
        if reset_index:
            return self.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        else:
            return self.sample(frac=1, random_state=random_seed)

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

    def to_frame(self, name=None):
        if name is None:
            df = self._onedata(self)
        else:
            df = self._onedata({name: self})
        return df

    # ========================= Plot ========================= #

    def count_plot(self,
                   inherit: plt = None,
                   figsize: list = None,
                   title: str = None,
                   xlabel: str = None,
                   ylabel: str = None,
                   show: bool = True):
        """
        Generate the count graph

        :param inherit: the inherit plt
        :param figsize: The size of figure.
        :param title: the title of plot
        :param xlabel: label of x
        :param ylabel: label of y
        :param show: plt.show will run if true
        """

        # previous configuration of plot
        plt = self._plot_prev_config(inherit, figsize, "seaborn-darkgrid")

        count_base_data = self.value_counts()
        plt.bar(list(count_base_data.keys()), list(count_base_data))

        # return for advanced adjustment
        return self._plot_post_config(plt, '', title, xlabel, ylabel, show)

    def line_plot(self,
                  inherit: plt = None,
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

        :param inherit: the inherit plt
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

        # previous configuration of plot
        plt = self._plot_prev_config(inherit, figsize, "seaborn-darkgrid")

        # data x pre-configuration process
        x = list(self.index)

        # data y pre-configuration process
        plt = self._meta_line_plot(plt, self, x, smooth, kind, interval, None)

        # return for advanced adjustment
        return self._plot_post_config(plt, legend_loc, title, xlabel, ylabel, show)
