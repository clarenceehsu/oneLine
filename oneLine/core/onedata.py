import sys
import traceback
import numpy as np
from abc import ABC
import pandas as pd
from pandas import DataFrame
from pandas.api.types import is_categorical_dtype
from pandas.api.types import is_datetime64_any_dtype as is_datetime

from .plot import Plot


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
        raise ValueError('Input format error, please in put a valid dataset that satisfied OneData.')

    def __init__(self, *args):
        """
        Input data and convert it to DataFrame as a sub type of OneData.
        """
        try:
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
                elif file_form == 'sql':
                    data = pd.read_sql(args[0])
                else:
                    self._raise_value_error()
            elif isinstance(args[0], (pd.DataFrame, OneData)):
                data = args[0]
            elif isinstance(args[0], np.ndarray):
                data = pd.DataFrame(args[0])
            elif isinstance(args[0], list):
                data = pd.DataFrame(args[0])
            else:
                self._raise_value_error()
        except Exception:
            traceback.print_exc(limit=1, file=sys.stdout)
            quit()
        super().__init__(data=data)

    # ---------------------------------------------------------------- #

    def show(self, info: bool = False):
        """
        Display the info of data.
        :param info: stay True if a display of information is required
        :return None
        """
        if info:
            self.info()
            print('\n - Shape: {}\n - Index:{}\n - Memory usage: {:.3f} MB\n'.format(self.shape, list(self.columns), self.memory_usage().sum() / 1024 ** 2))
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
        summ['First Value'] = self.loc[0].values
        summ['Second Value'] = self.loc[1].values
        summ['Third Value'] = self.loc[2].values

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
        data = DataFrame(self)
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
        if info:
            start_mem = self.memory_usage().sum() / 1024 ** 2
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


def data_input(*args):
    """
    A independent method for data input as DataFrame, which will automatically input based on the type of files.
    """

    file_form = args[0].split('.')[-1]
    if file_form == 'csv':
        return pd.read_csv(args[0])
    elif file_form == 'xls' or file_form == 'xlsx':
        return pd.read_excel(args[0])
    elif file_form == 'json':
        return pd.read_json(args[0])
    elif file_form == 'sql':
        return pd.read_sql(args[0])
