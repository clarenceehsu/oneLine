import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib.ticker as mtick


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

    def line_plot(self, y,
                  x: str = None,
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

        :param x: the x
        :param y: the y, which should be imported as list or str
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

        # Check if y is list or str
        if not isinstance(y, list) and not isinstance(y, str):
            self._raise_plot_format_error(["y"], "list or str")

        # previous configuration of plot
        plt = self._plot_prev_config(inherit, figsize, "seaborn-darkgrid")

        # data x pre-configuration process
        # x will be used if x is specified, otherwise the default index [0, len()] would be used
        if x:
            x = self[x]
        else:
            x = list(self.index)

        # data y pre-configuration process
        # y should import as a list for multiple series plot
        if isinstance(y, list):
            for name in y:
                y_val = self[name]
                if smooth:
                    x_new = np.linspace(min(x), max(x), len(x) * interval)
                    y_smooth = interp1d(x, y_val, kind=kind)
                    plt.plot(x_new, y_smooth(x_new), label=name)
                else:
                    plt.plot(x, y_val, label=name)
        else:
            y_val = self[y]
            if smooth:
                x_new = np.linspace(min(x), max(x), len(x) * interval)
                y_smooth = interp1d(x, y_val, kind=kind)
                plt.plot(x_new, y_smooth(x_new), label=y)
            else:
                plt.plot(x, y_val, label=y)

        # return for advanced adjustment
        return self._plot_post_config(plt, legend_loc, title, xlabel, ylabel, show)

    def count_plot(self, variable,
                   hue: str = None,
                   inherit: plt = None,
                   figsize: list = None,
                   title: str = None,
                   xlabel: str = None,
                   ylabel: str = None,
                   show: bool = True):
        """
        Generate the count graph

        :param variable: The variable that should be counted
        :param hue: the hue parameter for using
        :param inherit: the inherit plt
        :param figsize: The size of figure.
        :param title: the title of plot
        :param xlabel: label of x
        :param ylabel: label of y
        :param show: plt.show will run if true
        """

        # Check if y is list or str
        if not isinstance(variable, str):
            self._raise_plot_format_error(["variable"], "str")

        # previous configuration of plot
        plt = self._plot_prev_config(inherit, figsize, "seaborn-darkgrid")

        if hue:
            # unique value of hue
            unique = self[hue].unique().tolist()

            # generate the plot
            fig, axes = plt.subplots(1, len(unique))
            fig.set_size_inches(figsize)
            for index, ax in enumerate(axes):
                ax.set_xlabel(variable)
                ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
                ax.set_title(f"{ hue } = { unique[index] }")
                temp_data = self[self[hue] == unique[index]][variable].value_counts()
                plt.bar(list(temp_data.keys()), list(temp_data))
        else:
            count_base_data = self[variable].value_counts()
            plt.bar(list(count_base_data.keys()), list(count_base_data))

        # return for advanced adjustment
        return self._plot_post_config(plt, '', title, xlabel, ylabel, show)

    def corr_plot(self, parameters: list = None,
                  inherit: plt = None,
                  figsize: list = None,
                  title: str = None,
                  annot: bool = True,
                  show: bool = True):
        """
        Generate the correction graph
        :param parameters: The parameters selected.
        :param inherit: the inherit plt
        :param figsize: The size of figure.
        :param title: the title of plot
        :param annot: Display the annotation or not.
        :param show: plt.show will run if true
        """

        # Check if y is list or str
        if not isinstance(parameters, list):
            self._raise_plot_format_error(["parameters"], "list or None")

        # previous configuration of plot
        plt = self._plot_prev_config(inherit, figsize, "seaborn-dark")
        fig, ax = plt.subplots()

        data = self[parameters].corr()
        score = data.values
        col = data.columns
        length = len(col)
        im = ax.imshow(score, cmap='rocket_r')
        ax.xaxis.set_ticks_position('top')
        ax.set_xticks(np.arange(length))
        ax.set_yticks(np.arange(length))
        ax.set_xticklabels(col)
        ax.set_yticklabels(col)
        fig.colorbar(im, pad=0.03)

        # the annotation part
        if annot:
            for i in range(length):
                for j in range(length):
                    if score[i, j] > 0.4:
                        color = "w"
                    else:
                        color = "black"
                    ax.text(j, i, round(score[i, j], 2),
                                   ha="center", va="center", color=color)

        # return for advanced adjustment
        return self._plot_post_config(plt, '', title, '', '', show)

    def hist_plot(self, variable: str = None,
                  hue: str = None,
                  inherit: plt = None,
                  figsize: list = None,
                  show: bool = True):

        # check the format of variable1 and variable2
        if not variable or not hue:
            self._raise_plot_value_error(["variable", "hue"])
        elif self[variable].dtype == 'object' or self[hue].dtype == 'object':
            self._raise_plot_format_error(["variable", "hue"], "int or float")

        # previous configuration of plot
        plt = self._plot_prev_config(inherit, figsize, "seaborn-darkgrid")

        # unique value of variable2
        unique = self[hue].unique().tolist()

        # generate the plot
        fig, axes = plt.subplots(1, len(unique))
        fig.set_size_inches(figsize)
        for index, ax in enumerate(axes):
            ax.set_xlabel(variable)
            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
            ax.set_title(f"{ hue } = { unique[index] }")
            temp_data = self[self[hue] == unique[index]][variable]
            ax.hist(temp_data, density=True, stacked=True)

        # return for advanced adjustment
        return self._plot_post_config(plt, '', '', '', '', show)


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
