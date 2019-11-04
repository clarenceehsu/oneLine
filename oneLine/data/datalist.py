import os


class OneDatalist:
    def __init__(self, data_list=[], filepath=''):
        if data_list:
            self.data_list = data_list
        else:
            self.data_list = os.listdir(filepath)

    def shape(self):
        return len(self.data_list)

    def make_dataset(self, train_proportion=0.0):
        index_num = int(self.shape() * train_proportion)
        train_list = self.data_list[:index_num]
        test_list = self.data_list[index_num:]
        return OneDatalist(data_list=train_list), OneDatalist(data_list=test_list)

    def show(self):
        print(self.data_list)