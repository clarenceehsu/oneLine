import numpy as np
from abc import ABC
import pandas as pd
from pandas import DataFrame
from pandas.api.types import is_categorical_dtype
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas._libs import lib
from pandas.core.dtypes.common import (
    is_hashable,
    is_integer,
    is_iterator,
    is_list_like,
)
from pandas.core.dtypes.generic import ABCMultiIndex
from pandas.core import common as com
from pandas.core.indexing import convert_to_index_sliceable

from .plot import Plot
from .oneseries import OneSeries


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

    def _raise_value_error(self):
        """
        The exception of ValueError.
        :return: ValueError
        """
        raise ValueError('Input format error, please in put a valid dataset that satisfied OneData.')

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
                self._raise_value_error()
        else:
            data = args[0]
        try:
            super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)
        except (ValueError, TypeError):
            self._raise_value_error()

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

        if isinstance(key, DataFrame):
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

        return data

    # ---------------------------------------------------------------- #

    def show(self, info: bool = False):
        """
        Display the info of data.
        :param info: stay True if a display of information is required
        :return None
        """
        if info:
            self.info()
            print('\n - Shape: {}\n - Index:{}\n - Memory usage: {:.3f} MB\n'
                  .format(self.shape, list(self.columns), self.memory_usage().sum() / 1024 ** 2))
        print(self)

    def make_dataset(self, train: float = 0.0,
                     holdout: float = 0.0,
                     save: bool = False,
                     filepath: str = None):
        """
        Create dataset function.
        The proportion of the train data and hold-out data should be specified.
        :param train: the proportion of train data
        :param holdout: the proportion of holdout data
        :param save: set True if saving dataset is required
        :param filepath: the path of dataset
        """
        index_num = int(self.shape[0] * train)
        train_data = self.iloc[:index_num, :]
        holdout_data = None

        if holdout:
            holdout_num = int(self.shape[0] * holdout)
            test_data = self.iloc[index_num:index_num + holdout_num, :]
            holdout_data = self.iloc[index_num + holdout_num:, :]
        else:
            test_data = self.iloc[index_num:, :]
        if save:
            train_data.to_csv(path_or_buf=filepath.split('.')[0] + '_train.csv', index=False)
            test_data.to_csv(path_or_buf=filepath.split('.')[0] + '_test.csv', index=False)
            if holdout:
                holdout_data.to_csv(path_or_buf=filepath.split('.')[0] + '_holdout.csv', index=False)
        elif holdout:
            return OneData(train_data), OneData(holdout_data), OneData(test_data)
        else:
            return OneData(train_data), OneData(test_data)

    def summary(self, info: bool = True):
        """
        Return a summary of the whole dataset.
        the stats from scipy is used to calculate the Entropy.
        :param info: stay True if a display of information is required
        """
        from scipy import stats

        pd.set_option('display.max_columns', None)
        print(f"Dataset Shape: {self.shape}")
        summ = pd.DataFrame(self.dtypes, columns=['dtypes'])
        summ = summ.reset_index()
        summ['Name'] = summ['index']
        summ = summ[['Name', 'dtypes']]
        summ['Missing'] = self.isnull().sum().values
        summ['Uniques'] = self.nunique().values
        summ['First Value'] = self.loc[self.index[0]].values
        summ['Second Value'] = self.loc[self.index[1]].values
        summ['Third Value'] = self.loc[self.index[2]].values

        for name in summ['Name'].value_counts().index:
            summ.loc[summ['Name'] == name, 'Entropy'] = round(
                stats.entropy(self[name].value_counts(normalize=True), base=2), 2)
        if info:
            print(summ)
        return summ

    def fill_na(self, method: str = 'mode'):
        """
        Fill the NaN values.
        :param method: the way to fill the NaN values, which contains mode and nan methods
        """
        data = self
        if method == 'mode':
            for key, value in data.isnull().sum().items():
                if value:
                    data[key].fillna(self[key].mode()[0], inplace=True)
        elif method == 'nan':
            data = data.fillna(np.nan)
        return OneData(data)

    def remove(self, column: list = None, row: list = None):
        """
        The advanced function of DataFrame.drop, you can input a list of index to drop them.
        :param column: the list of column you want to remove
        :param row: the list of row you want to remove
        """
        data = self
        if column is None:
            column = []
        if column:
            data = data.drop(column, axis=1)
        if row:
            for n in row:
                data = data.drop(n)
        return OneData(data)

    def add_var(self, exist, new, mapper=None):
        """
        Add a new variable based on the calculating of exist variable.
        :param exist: the exist variable column
        :param new: the new variable column
        :param mapper: the mapper applied to the exist variable
        """
        data = self
        if mapper:
            data[new] = list(map(mapper, data[exist]))
        else:
            data[new] = data[exist]
        return data

    def means(self, variable: str = None, hue: str = None):
        """
        An advanced method of mean, which can calculate means with hue
        :param variable: the variable to calculate means
        :param hue: the hue for calculation
        """
        if hue:
            return self[[hue, variable]][self[variable].isnull() == False].groupby([hue],
                                                                                   as_index=False).mean().sort_values(
                by=variable, ascending=False)
        elif variable:
            return self.mean()[variable]
        else:
            return self.mean()

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

        return OneData(self)

    def iter(self, index=True, name="OneData"):
        """
        Iteration methods for fast using
        :param index: If True, return the index as the first element of the tuple.
        :param name: The name of the returned named tuples or None to return regular.
        """
        return self.itertuples(index=index, name=name)

    def select_raw(self, indices: dict, reset_index: bool = False):
        """
        Select raw with particular indices.

        For example:
        a = a.select_raw({'Gender': 'Male'})  # it will returns the data that matched variable'Gender' == values'Male'
        a = a.select_raw({'Gender': 'Male', 'Age': 20})  # it can also use for multiple matching
        """
        df = self
        for variable, value in indices.items():
            df = df[(df[variable] == value)]
        if reset_index:
            df = df.reset_index()
        return OneData(df)
