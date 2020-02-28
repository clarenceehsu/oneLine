import sys
import traceback
import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype
from pandas.api.types import is_datetime64_any_dtype as is_datetime

from .plot import Plot
from .data_pd import Pandas


class OneData(Plot, Pandas):
    """
    That's the initial of the OneData Object.
    A input list, DataFrame or a list can fit a OneData object, and the object will be automatically convert to
    DataFrame, which is the sub-object of OneData.
    The structure of OneData:
        1. self.data
            It's the main DataFrame dataset. And all the built-in functions were based on it.
        *2. self.train_data
        *3. self.test_data
        *4. self.holdout_data
        These three parameters is unstable now, and it will be substituted in the future.
    """
    def _raise_value_error(self):
        raise ValueError('Input format error, please in put a valid dataset that satisfied OneData.')

    def __init__(self, *args):
        """
        Input data and convert it to DataFrame as a sub type of OneData.
        """
        super().__init__(*args)
        self.train_data = None
        self.model = None

        try:
            if not args:
                pass

            elif isinstance(args[0], str):
                file_form = args[0].split('.')[-1]
                if file_form == 'csv':
                    self.data = pd.read_csv(args[0])
                elif file_form == 'xls' or file_form == 'xlsx':
                    self.data = pd.read_excel(args[0])
                elif file_form == 'json':
                    self.data = pd.read_json(args[0])
                elif file_form == 'sql':
                    self.data = pd.read_sql(args[0])
                else:
                    self._raise_value_error()
            elif isinstance(args[0], pd.DataFrame):
                self.data = args[0]
            elif isinstance(args[0], np.ndarray):
                self.data = pd.DataFrame(args[0])
            elif isinstance(args[0], OneData):
                self.data = args[0].data
                self.holdout_data = args[0].holdout_data
                self.model = args[0].model
            elif isinstance(args[0], list):
                self.data = pd.DataFrame(args[0])
            else:
                self._raise_value_error()
        except Exception:
            traceback.print_exc(limit=1, file=sys.stdout)
            quit()

    def __len__(self):
        """
        Return the length of data.
        """
        return len(self.data)

    def __getitem__(self, item):
        """
        getitem function.
        """
        return OneData(pd.DataFrame(self.data[item]))

    def __repr__(self):
        """
        Change the print to pandas style.
        """
        print(self.data)
        return ''

    # ---------------------------------------------------------------- #

    def show(self, info: bool = False):
        """
        Display the info of data.

        While info = True, the output will contain the details of data like the shape property or others.
        """
        if not info:
            print(self.data)
        elif info:
            all_usage = 0.0
            usage = self.data.memory_usage().sum() / 1024 ** 2
            shape = self.data.shape
            columns = list(self.data.columns)
            all_usage += usage
            print(self.data)
            print('\n - Shape: {}\n - Index:{}\n - Memory usage: {:.3f} MB\n'.format(shape, columns, usage))

    def make_dataset(self, train: float = 0.0,
                     hold_out: float = 0.0,
                     save: bool = False,
                     filepath: str = ''):
        """
        Create dataset function.
        The proportion of the train data and hold-out data should be specified.
        """
        index_num = int(self.shape()[0] * train)
        self.train_data = self.data.iloc[:index_num, :]

        if hold_out:
            holdout_num = int(self.shape()[0] * hold_out)
            self.test_data = self.data.iloc[index_num:index_num + holdout_num, :]
            self.holdout_data = self.data.iloc[index_num + holdout_num:, :]
        else:
            self.test_data = self.data.iloc[index_num:, :]
        if save:
            self.train_data.to_csv(path_or_buf=filepath.split('.')[0] + '_train.csv', index=False)
            self.test_data.to_csv(path_or_buf=filepath.split('.')[0] + '_test.csv', index=False)
            if hold_out:
                self.holdout_data.to_csv(path_or_buf=self.filepath.split('.')[0] + '_holdout.csv', index=False)
        elif hold_out:
            return OneData(self.train_data), OneData(self.holdout_data), OneData(self.test_data)
        else:
            return OneData(self.train_data), OneData(self.test_data)

    def summary(self):
        """
        Return a summary of the whole dataset.
        the stats from scipy is used to calculate the Entropy.
        """
        from scipy import stats

        pd.set_option('display.max_columns', None)
        print(f"Dataset Shape: {self.data.shape}")
        summ = pd.DataFrame(self.data.dtypes, columns=['dtypes'])
        summ = summ.reset_index()
        summ['Name'] = summ['index']
        summ = summ[['Name', 'dtypes']]
        summ['Missing'] = self.data.isnull().sum().values
        summ['Uniques'] = self.data.nunique().values
        summ['First Value'] = self.data.loc[0].values
        summ['Second Value'] = self.data.loc[1].values
        summ['Third Value'] = self.data.loc[2].values

        for name in summ['Name'].value_counts().index:
            summ.loc[summ['Name'] == name, 'Entropy'] = round(
                stats.entropy(self.data[name].value_counts(normalize=True), base=2), 2)
        print(summ)

    def fill_na(self, method: str = 'mode'):
        """
        Fill the NA values.
        method: mode, nan
        """
        data = self.data
        if method == 'mode':
            for key, value in data.isnull().sum().items():
                if value:
                    data[key].fillna(self.data[key].mode()[0], inplace=True)
        elif method == 'nan':
            data = data.fillna(np.nan)
        return OneData(data)

    def append(self, others, ignore_index: bool = True):
        """
        Append two dataset.
        """
        if isinstance(others, pd.DataFrame):
            return OneData(self.data.append(others, ignore_index=ignore_index))
        if isinstance(others, OneData):
            return OneData(self.data.append(others.data, ignore_index=ignore_index))

    def drop(self, column: list = None, row: list = None):
        """
        The advanced function of DataFrame.drop, you can input a list of index to drop them.
        """
        if column is None:
            column = []
        data = self.data
        if column:
            data = data.drop(column, axis=1)
        if row:
            for n in row:
                data = data.drop(n)
        return OneData(data)

    def reduce_mem_usage(self, use_float16: bool = False):
        """
        Automatically distinguish the type of one single data and reset a suitable type.
        """
        df = self.data
        start_mem = df.memory_usage().sum() / 1024 ** 2
        print("Memory usage of DataFrame is {:.3f} MB".format(start_mem))

        for col in df.columns:
            if is_datetime(df[col]) or is_categorical_dtype(df[col]):
                continue
            col_type = df[col].dtype

            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == "int":
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            else:
                df[col] = df[col].astype("category")

        end_mem = df.memory_usage().sum() / 1024 ** 2
        print("Memory usage after optimization is: {:.3f} MB".format(end_mem))
        print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

        return OneData(self.data)


def data_input(*args):
    """
    Input data as DataFrame.
    """

    file_form = args[0].split('.')[-1]
    if file_form == 'csv':
        data = pd.read_csv(args[0])
    if file_form == 'xls' or file_form == 'xlsx':
        data = pd.read_excel(args[0])
    if file_form == 'json':
        data = pd.read_json(args[0])
    if file_form == 'sql':
        data = pd.read_sql(args[0])

    return data
