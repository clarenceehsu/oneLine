"""
The plot function of oneline. It contains a series of plot methods for a fast plot using.
"""

import numpy as np
import matplotlib.pyplot as plt
from ..tools.compat import import_optional_dependency


class Plot(object):
    """
    This is a plot module for data visualization.
    """

    @property
    def _plt(self):
        import matplotlib.pyplot as plt
        return plt

    def _plot_prev_config(self, inherit: plt = None,
                          figsize: list = None,
                          style: str = None):
        """
        A general pre-configuration of plot process.

        :param inherit: the outscope configuration of plot
        :param figsize: the size of figuration
        :param style: the style of plot
        :return: plt
        """

        # inherit previous plt configuration if exists
        if inherit:
            plt = inherit
        else:
            plt = self._plt
            plt.style.use(style)

        # set the size of figuration if it's required
        if figsize:
            plt.figure(figsize=figsize)

        return plt

    @staticmethod
    def _plot_post_config(plt,
                          legend_loc: str,
                          title: str,
                          xlabel: str,
                          ylabel: str,
                          show: bool):
        """
        The post-configuration of plot process.

        :param plt: the plt
        :param legend_loc: the position of legend
        :param title: the title
        :param xlabel: the xlabel
        :param ylabel: the ylabel
        :param show: show the plot if True
        :return: plt
        """

        # Other post configuration
        if legend_loc:
            plt.legend(loc=legend_loc)
        if title:
            plt.title(title)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)

        # show the plot if turns True
        if show:
            plt.show()

        # return for advanced adjustment
        return plt

    @staticmethod
    def _meta_line_plot(plt, y_val, x, smooth, kind, interval, name):
        """
        A meta function of line_plot.

        :param plt: the inherit plt
        :param y_val: y value
        :param x: x value
        :param smooth: smooth the line if True
        :param kind: the kind of smooth method
        :param interval: the number of interval
        :param name: the name of line
        :return: plt
        """

        if smooth:
            interpolate = import_optional_dependency("scipy.interpolate")
            x_new = np.linspace(min(x), max(x), len(x) * interval)
            y_smooth = interpolate.interp1d(x, y_val, kind=kind)
            plt.plot(x_new, y_smooth(x_new), label=name)
        else:
            plt.plot(x, y_val, label=name)

        return plt
