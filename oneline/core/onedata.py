"""
OneData is an extended DataFrame based on pandas.DataFrame, which also inherits all the pandas.DataFrame
features and ready to use. It contains many useful methods for a better experience on data analysis.

WARNING: Because this module is still pre-alpha, so many features are unstable.
"""

import collections
import numpy as np
from abc import ABC
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# pandas
import pandas as pd
from pandas import DataFrame
from pandas._libs import lib
from pandas.util._decorators import doc
from pandas.api.types import is_categorical_dtype
from pandas.api.types import is_datetime64_any_dtype as is_datetime

# pandas.core
from pandas.core import common as com
from pandas.core.generic import NDFrame
from pandas.core.dtypes.common import (
    is_hashable,
    is_integer,
    is_iterator,
    is_list_like,
)
from pandas.core.dtypes.generic import ABCMultiIndex
from pandas.core.indexing import convert_to_index_sliceable

# oneline module
from .plot import Plot
from .oneseries import OneSeries

_shared_doc_kwargs = {
    "axes": "index, columns",
    "klass": "DataFrame",
    "axes_single_arg": "{0 or 'index', 1 or 'columns'}",
    "axis": """axis : {0 or 'index', 1 or 'columns'}, default 0
        If 0 or 'index': apply function to each column.
        If 1 or 'columns': apply function to each row.""",
    "optional_by": """
        by : str or list of str
            Name or list of names to sort by.

            - if `axis` is 0 or `'index'` then `by` may contain index
              levels and/or column labels.
            - if `axis` is 1 or `'columns'` then `by` may contain column
              levels and/or index labels.""",
    "optional_labels": """labels : array-like, optional
            New labels / index to conform the axis specified by 'axis' to.""",
    "optional_axis": """axis : int or str, optional
            Axis to target. Can be either the axis name ('index', 'columns')
            or number (0, 1).""",
}


class OneData(DataFrame, ABC, Plot):
    """
    That's the initial of the OneData class.
    Acceptable type:
        1. OneData
        2. DataFrame
        3. List
        4. Numpy Array
        5. String (the string should be a address of particular file, which can be automatically input based on format
                   by OneData)
    OneData is a advanced class of Pandas.DataFrame with more methods and all the features of DataFrame inherited.
    """

    def _raise_format_error(self):
        """
        The exception of ValueError when format was unsupported.
        :return: ValueError
        """
        raise ValueError('Input format error, please input a valid dataset that satisfied OneData.')

    def _raise_parameter_error(self, args):
        """
        The exception of ValueError when format was unsupported.
        :return: ValueError
        """
        raise ValueError('Input parameter {} {} not exist.'.format(", ".join(args), "is" if len(args) == 1 else "are"))

    def __init__(self, *args, index=None, columns=None, dtype=None, copy: bool = False):
        """
        Input data or address string, and convert it to OneData format.
        Now the input args supported:
            1. str: a string will be recognized as the location of a file, and input with the format correctly
            2. DataFrame & OneData
            3. Dictionary
            4. Numpy Array
            ...
        OneData inherits all the features of DataFrame.
        """
        data = {}
        if not args:
            pass
        elif isinstance(args[0], str):
            file_form = args[0].split('.')[-1]
            if file_form == 'csv':
                data = pd.read_csv(args[0])
            elif file_form == 'xls' or file_form == 'xlsx':
                data = pd.read_excel(args[0])
            elif file_form == 'json':
                data = pd.read_json(args[0])
            elif file_form == 'pkl':
                data = pd.read_pickle(args[0])
            elif file_form == 'hdf':
                data = pd.read_hdf(args[0])
            else:
                self._raise_format_error()
        else:
            data = args[0]
        try:
            super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)
        except (ValueError, TypeError):
            self._raise_format_error()

    def __getitem__(self, key):
        """
        The rebuilt method of __getitem__, which will redirect the DataFrame and Series to OneData and OneSeries.
        """
        key = lib.item_from_zerodim(key)
        key = com.apply_if_callable(key, self)

        if is_hashable(key):
            if self.columns.is_unique and key in self.columns:
                if self.columns.nlevels > 1:
                    return OneSeries(self._getitem_multilevel(key))
                return OneSeries(self._get_item_cache(key))

        indexer = convert_to_index_sliceable(self, key)
        if indexer is not None:
            return OneData(self._slice(indexer, axis=0))

        if isinstance(key, DataFrame) or isinstance(key, OneData):
            return self.where(key)

        if com.is_bool_indexer(key):
            return OneData(self._getitem_bool_array(key))

        is_single_key = isinstance(key, tuple) or not is_list_like(key)

        if is_single_key:
            if self.columns.nlevels > 1:
                return self._getitem_multilevel(key)
            indexer = self.columns.get_loc(key)
            if is_integer(indexer):
                indexer = [indexer]
        else:
            if is_iterator(key):
                key = list(key)
            indexer = self.loc._get_listlike_indexer(key, axis=1, raise_missing=True)[1]

        if getattr(indexer, "dtype", None) == bool:
            indexer = np.where(indexer)[0]

        data = self._take_with_is_copy(indexer, axis=1)

        if is_single_key:
            if data.shape[1] == 1 and not isinstance(self.columns, ABCMultiIndex):
                data = data[key]

        return OneData(data)

    def show(self, info: bool = False,
             all_columns: bool = False,
             all_rows: bool = False,
             max_columns: int = None,
             max_rows: int = None,
             precision: int = None):
        """
        Display the info and details of data.
        The display options would be reset to the previous state after printing data, in other words, the options that
        was edited in show() would not be inherited.
        :param info: stay True if a display of information is required
        :param all_columns: reset the display of columns if columns equals to True
        :param all_rows: reset the display of rows if rows equals to True
        :param max_columns: the max columns of display
        :param max_rows: the max rows of display
        :param precision: a fast adjustment to the precision of display
        :return None
        """
        if info:
            self.info(memory_usage=False)
            print('\n\n - Shape: {}\n - Index: {}\n - Memory usage: {:.3f} MB\n'
                  .format(self.shape, ", ".join(self.columns), self.memory_usage().sum() / 1024 ** 2))
        original_max_columns = pd.get_option('display.max_columns')
        original_max_rows = pd.get_option('display.max_rows')
        original_display_precision = pd.get_option('display.precision')
        if max_columns:
            pd.set_option('display.max_columns', max_columns)
        if max_rows:
            pd.set_option('display.max_rows', max_rows)
        if all_columns:
            pd.set_option('display.max_columns', None)
        if all_rows:
            pd.set_option('display.max_rows', None)
        if precision:
            pd.set_option('display.precision', precision)
        print(self)
        pd.set_option('display.max_rows', original_max_rows)
        pd.set_option('display.max_columns', original_max_columns)
        pd.set_option('display.precision', original_display_precision)

    def make_dataset(self, train_frac: float = None,
                     random: bool = False,
                     random_seed: int = None):
        """
        Create the separated datasets based on the original one.
        The proportion of the train data and hold-out data should be specified.
        :param train_frac: the proportion of train data
        :param random: set True if random dataset is needed
        :param random_seed: the random seed of making dataset procession
        :returns train_data, test_data
        """
        if random:
            train_data = self.sample(frac=train_frac, random_state=random_seed)
            test_data = self.drop(train_data.index)
            return OneData(train_data), OneData(test_data)
        elif train_frac:
            index_num = int(self.shape[0] * train_frac)
            train_data = self.iloc[:index_num, :]
            test_data = self.iloc[index_num:, :]
            return OneData(train_data), OneData(test_data)
        else:
            self._raise_parameter_error(["train_frac or random"])

    def reverse(self, axis: str = 'row', reset_index: bool = False):
        """
        Method for reversing the dataset.
        :param axis: set 'row' if the order of the rows is to be reversed, and set 'column' if the order of the column
        is to be reversed
        :param reset_index: true for reset the index of data frame
        :param inplace: inplace modification if it sets True
        :return OneData
        """
        if axis == 'row':
            if reset_index:
                return OneData(self.loc[::-1].reset_index(drop=True))
            else:
                return OneData(self.loc[::-1])
        elif axis == 'column':
            return OneData(self.loc[:, ::-1])

    def summary(self, info: bool = True):
        """
        Return a summary of the whole dataset.
        the stats from scipy is used to calculate the Entropy.
        :param info: stay True if a display of information is required
        :return the detail of summary
        """
        from scipy import stats

        pd.set_option('display.max_columns', None)
        print(f"Dataset Shape: {self.shape}")
        summary_info = pd.DataFrame(self.dtypes, columns=['dtypes'])
        summary_info = summary_info.reset_index()
        summary_info['Name'] = summary_info['index']
        summary_info = summary_info[['Name', 'dtypes']]
        summary_info['Missing'] = self.isnull().sum().values
        summary_info['Uniques'] = self.nunique().values
        summary_info['First Value'] = self.loc[self.index[0]].values
        summary_info['Second Value'] = self.loc[self.index[1]].values
        summary_info['Third Value'] = self.loc[self.index[2]].values

        for name in summary_info['Name'].value_counts().index:
            summary_info.loc[summary_info['Name'] == name, 'Entropy'] = round(
                stats.entropy(self[name].value_counts(normalize=True), base=2), 2)
        if info:
            print(summary_info)
        return summary_info

    @doc(NDFrame.fillna, **_shared_doc_kwargs)
    def fillna(
            self,
            value=None,
            method=None,
            axis=None,
            inplace=False,
            limit=None,
            downcast=None,
    ):
        """
        Fill the NaN values, which is an extended function of original fillna().
        :param method: the way to fill the NaN values, which contains mode and nan methods
        :param inplace: inplace modification
        :param value: inherit
        :param axis: inherit
        :param limit: inherit
        :param downcast: inherit
        """
        if method == 'mode':
            if inplace:
                for key, value in self.isnull().sum().items():
                    if value:
                        self[key].fillna(self[key].mode()[0], inplace=True)
            else:
                data = self
                for key, value in data.isnull().sum().items():
                    if value:
                        data[key].fillna(self[key].mode()[0], inplace=True)
                return data
        elif method == 'nan':
            if inplace:
                self.fillna(np.nan, inplace=True)
            else:
                data = self
                data = data.fillna(np.nan)
                return data
        else:
            return super().fillna(
                value=value,
                method=method,
                axis=axis,
                inplace=inplace,
                limit=limit,
                downcast=downcast,
            )

    def remove(self, column: list = None, row: list = None):
        """
        The advanced function of DataFrame.drop, you can input a list of index to drop them.
        :param column: the list of column you want to remove
        :param row: the list of row you want to remove
        """
        data = self
        if column:
            data = data.drop(column, axis=1)
        if row:
            for n in row:
                data = data.drop(n)
        return OneData(data)

    def add_var(self, exist, new, mapper=None):
        """
        Generate a new variable based on the calculating of exist variable using map().
        :param exist: the exist column of data
        :param new: the name of new column
        :param mapper: the mapper applied to the exist variable
        For example:
        >>       0    1    2
            0  1.0  2.0  3.0
            1  1.0  2.0  2.0
            2  1.0  2.0  2.0
            3  8.0  8.0  2.0

        >> data.add_var(0, 3, lambda x: x * 2)

        >>       0    1    2     3
            0  1.0  2.0  3.0   2.0
            1  1.0  2.0  2.0   2.0
            2  1.0  2.0  2.0   2.0
            3  8.0  8.0  2.0  16.0
        """
        data = self
        if mapper:
            data[new] = list(map(mapper, data[exist]))
        else:
            data[new] = data[exist]
        return data

    def mean(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        return OneSeries(NDFrame.mean(self, axis, skipna, level, numeric_only, **kwargs))

    def reduce_mem_usage(self, use_float16: bool = False, info: bool = True):
        """
        Automatically distinguish the type of one single data and reset a suitable type.
        :param use_float16: use float16 or not
        :param info: stay True if a display of information is required
        """
        start_mem = self.memory_usage().sum() / 1024 ** 2
        if info:
            print("Memory usage of DataFrame is {:.3f} MB".format(start_mem))

        for col in self.columns:
            if is_datetime(self[col]) or is_categorical_dtype(self[col]):
                continue
            col_type = self[col].dtype

            if col_type != object:
                c_min = self[col].min()
                c_max = self[col].max()
                if str(col_type)[:3] == "int":
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        self[col] = self[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        self[col] = self[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        self[col] = self[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        self[col] = self[col].astype(np.int64)
                else:
                    if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        self[col] = self[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        self[col] = self[col].astype(np.float32)
                    else:
                        self[col] = self[col].astype(np.float64)
            else:
                self[col] = self[col].astype("category")

        if info:
            end_mem = self.memory_usage().sum() / 1024 ** 2
            print("Memory usage after optimization is: {:.3f} MB".format(end_mem))
            print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    def iter(self, index=True, name="OneData"):
        """
        Iteration methods for fast using, the override of itertuples().
        :param index: If True, return the index as the first element of the tuple.
        :param name: The name of the returned named tuples or None to return regular.
        """
        arrays = []
        fields = list(self.columns)
        if index:
            arrays.append(self.index)
            fields.insert(0, "Index")
        arrays.extend(self.iloc[:, k] for k in range(len(self.columns)))

        if name is not None:
            itertuple = collections.namedtuple(name, fields, rename=True)
            return map(itertuple._make, zip(*arrays))

        return zip(*arrays)

    def r_append(self, other):
        return OneData(pd.concat([self, other], axis=1))

    def l_append(self, other):
        return OneData(pd.concat([other, self], axis=1))

    # ================== Plot Functions ================== #

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
                plt = self._meta_line_plot(plt, self[name], x, smooth, kind, interval, name)
        else:
            plt = self._meta_line_plot(plt, self[y], x, smooth, kind, interval, y)

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
