import os


class OneDatalist:
    def __init__(self, *args):
        if isinstance(args[0], list):
            self.data_list = args[0]
        elif isinstance(args[0], str):
            self.data_list = os.listdir(args[0])
        else:
            print('Input format error, please in put a valid dataset that satisfied OneData.')
            quit()

    def shape(self):
        return len(self.data_list)

    def make_dataset(self, train_proportion=0.0):
        index_num = int(self.shape() * train_proportion)
        train_list = self.data_list[:index_num]
        test_list = self.data_list[index_num:]
        return OneDatalist(data_list=train_list), OneDatalist(data_list=test_list)

    def head(self, n=5):
        return self.data_list[:n]

    def to_list(self):
        return self.data_list
