import pandas as pd
import numpy as np
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype

from .plot import Plot
from .data_pd import Pandas


class OneData(Plot, Pandas):

    def __init__(self, *args):
        super().__init__(*args)
        self.train_data = ''
        self.test_data = ''

        if isinstance(args[0], str):
            from ..io import auto_read
            filepath = args[0]
            self.data = auto_read(filepath)
        elif isinstance(args[0], pd.DataFrame):
            self.data = args[0]
        elif isinstance(args[0], OneData):
            self.data = args[0].data
            self.train_data = args[0].train_data
            self.test_data = args[0].test_data
        elif isinstance(args[0], list):
            if isinstance(args[0][0], int):
                self.data = pd.DataFrame({0: args[0]})
            else:
                temp = {}
                for n in range(len(args[0])):
                    temp[n] = args[0][n]
                self.data = pd.DataFrame(temp)

    # create dataset from the data, and the train_proportion is the proportion of the train dataset
    def make_dataset(self, train_proportion=0.0, save=False):
        index_num = int(self.shape()[0] * train_proportion)
        self.train_data = self.data.iloc[:index_num, :]
        self.test_data = self.data.iloc[index_num:, :]
        if save:
            self.train_data.to_csv(path_or_buf=self.filepath.split('.')[0] + '_train.csv', index=False)
            self.test_data.to_csv(path_or_buf=self.filepath.split('.')[0] + '_test.csv', index=False)
        else:
            return OneData(self.train_data), OneData(self.test_data)

    # create the summary of data
    def summary(self):
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

        return summ

    # fill the na values
    def fill_na(self):
        data = self.data
        for key, value in data.isnull().sum().items():
            if value:
                data[key].fillna(self.data[key].mode()[0], inplace=True)
        return OneData(data)

    # the advanced function of DataFrame.drop, you can input a list of index to drop them
    def drop(self, column=[], row=[]):
        data = self.data
        if column:
            data = data.drop(column, axis=1)
        if row:
            for n in row:
                data = data.drop(n)
        return OneData(data)

    def reduce_mem_usage(self, use_float16=False):
        df = self.data
        start_mem = df.memory_usage().sum() / 1024 ** 2
        print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

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
        print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
        print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

        return OneData(self.data)