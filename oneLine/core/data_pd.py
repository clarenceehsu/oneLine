from ..modules.data import *


class Pandas:
    """
    That a module inherit some useful functions from Pandas.
    """
    def __init__(self, *args):
        self.data = pd.DataFrame()

    @property
    def shape(self):
        """
        Return the shape of the data.
        """
        return self.data.shape

    @property
    def length(self):
        """
        Return the length of the data.
        """
        return self.data.shape[0] * self.data.shape[1]

    @property
    def columns(self):
        return self.data.columns

    def head(self, n=5):
        """
        Return the head of the data.
        """
        return self.data.head(n=n)

    def info(self):
        """
        The info of the data, which is same as Pandas.
        """
        return self.data.info()

    def to_list(self):
        """
        Convert the data to list.
        """
        return np.array(self.data).tolist()

    def to_array(self):
        """
        Convert the data to array.
        """
        return np.array(self.data)

    def to_df(self):
        """
        Convert the data to DataFrame,
        """
        return self.data

    def save_csv(self, filepath):
        """
        Save to a .csv file.
        """
        self.data.to_csv(path_or_buf=filepath, index=False)

    def save_excel(self, filepath):
        """
        Save to a .xls or .xlsx file.
        """
        self.data.to_excel(excel_writer=filepath, index=False)