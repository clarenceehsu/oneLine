from ..modules.data import *
from scipy import stats
from ..io import auto_read
from .plot import Plot


class OneData(Plot):
    def __init__(self, *args):
        super().__init__(*args)
        self.train_data = ''
        self.test_data = ''
        if isinstance(args[0], str):
            filepath = args[0]
            self.data = auto_read(filepath)
        elif isinstance(args[0], pd.DataFrame):
            self.data = args[0]
        elif isinstance(args[0], OneData):
            self.data = args[0].data
            self.train_data = args[0].train_data
            self.test_data = args[0].test_data
        elif isinstance(args[0], list):
            self.data = pd.DataFrame(args[0])

    # return the shape of the data
    def shape(self):
        return self.data.shape

    # return the head of the data
    def head(self, n=5):
        return OneData(self.data.head(n=n))

    # create dataset from the data, and the train_proportion is the proportion of the train dataset
    def make_dataset(self, train_proportion=0.0, save=False):
        index_num = int(self.shape()[0] * train_proportion)
        self.train_data = self.data.iloc[:index_num, :]
        self.test_data = self.data.iloc[index_num:, :]
        if save:
            self.train_data.to_csv(path_or_buf=self.filepath.split('.')[0] + '_train.csv', index=False)
            self.test_data.to_csv(path_or_buf=self.filepath.split('.')[0] + '_test.csv', index=False)

    # create the summary of data
    def summary(self):
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

    # convert the data to list
    def to_list(self):
        return np.array(self.data).tolist()

    # convert the data to array
    def to_array(self):
        return np.array(self.data)

    # convert the data to DataFrame
    def to_df(self):
        return self.data

    # save to a .csv file
    def save_csv(self, filepath):
        self.data.to_csv(path_or_buf=filepath, index=False)

    # save to a .xls file
    def save_excel(self, filepath):
        self.data.to_excel(excel_writer=filepath, index=False)

    # the advanced function of DataFrame.drop, you can input a list of index to drop them
    def drop(self, column=[], index=[]):
        data = self.data
        if column:
            data = data.drop(column, axis=1)
        if index:
            for n in index:
                data = data.drop(n)
        return OneData(data)
