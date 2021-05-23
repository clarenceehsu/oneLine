import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class Plot:
    """
    This is a plot module for data visualization.
    """

    def _raise_plot_value_error(self, s: list):
        raise ValueError(f'The parameter { ", ".join(s) } {"is" if len(s) == 1 else "are"} required.')

    def _raise_plot_format_error(self, s: list, format: str):
        raise ValueError(f'The parameter { ", ".join(s) } should be { format }, rather than { ", ".join([str(type(n)) for n in s]) }.')

    @property
    def _plt(self):
        import matplotlib.pyplot as plt
        return plt

    def _plot_prev_config(self, inherit: plt = None,
                          figsize: list = None,
                          style: str = None):

        # inherit previous plt configuration if exists
        if inherit:
            plt = inherit
        else:
            plt = self._plt
            plt.style.use(style)

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
        if smooth:
            x_new = np.linspace(min(x), max(x), len(x) * interval)
            y_smooth = interp1d(x, y_val, kind=kind)
            plt.plot(x_new, y_smooth(x_new), label=name)
        else:
            plt.plot(x, y_val, label=name)

        return plt
