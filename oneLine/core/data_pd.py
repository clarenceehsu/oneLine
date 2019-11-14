from ..modules.data import *


class Pandas:
    def __init__(self, *args):
        self.data = pd.DataFrame()

    # return the shape of the core
    @property
    def shape(self):
        return self.data.shape

    @property
    def length(self):
        return self.data.shape[0] * self.data.shape[1]

    # return the head of the data
    def head(self, n=5):
        return self.data.head(n=n)

    def info(self):
        return self.data.info()

    # convert the core to list
    def to_list(self):
        return np.array(self.data).tolist()

    # convert the core to array
    def to_array(self):
        return np.array(self.data)

    # convert the core to DataFrame
    def to_df(self):
        return self.data

    # save to a .csv file
    def save_csv(self, filepath):
        self.data.to_csv(path_or_buf=filepath, index=False)

    # save to a .xls file
    def save_excel(self, filepath):
        self.data.to_excel(excel_writer=filepath, index=False)