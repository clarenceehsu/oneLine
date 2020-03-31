from pandas import Series


class OneSeries(Series):
    def __init__(self, data):
        super().__init__(data=data)