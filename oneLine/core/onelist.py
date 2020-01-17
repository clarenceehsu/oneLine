import os

from .onedata import OneData


class OneList:
    """
    OneList is a solution for manage and edit OneData and more.
    Now this object is only for OneData and manage them.
    """
    def __init__(self, *args):
        self.data_list = []

        if not args:
            pass

        elif isinstance(args[0], list) and isinstance(args[0][0], OneData):
            self.data_list = args[0]
        elif isinstance(args[0], str):
            temp_list = os.listdir(args[0])
            os.chdir(args[0])
            for n in temp_list:
                self.data_list.append(OneData(n))
        else:
            print('Input format error, please in put a valid dataset that satisfied OneList.')
            quit()

    def __getitem__(self, item):
        """
        Get OneData item from OneList. And it will return 'No item matched.' and quit if there is no data matched.
        Every OneData would have their file name as the self.name parameters, which is used to match the item.
        """
        for n in self.data_list:
            print(n.name, item)
            if n.name == item:
                return n

        print('No item matched.')
        quit()

    def __repr__(self):
        temp = ''
        for n in self.data_list:
            temp += f'{ n.name }\n'
        temp += f'Total: { len(self.data_list) }'
        return temp

    def append(self, *args):
        if isinstance(args[0], OneList):
            self.data_list += args[0].data_list
        elif isinstance(args[0], list) and isinstance(args[0][0], OneData):
            self.data_list += args[0]
        elif isinstance(args[0], OneData):
            self.data_list.append(args[0])

    def show(self, info: bool = False):
        """
        Show the information of the OneList.
        It will only print names of OneDatas is info=False, and will print the usage and other details if info=True
        """
        if not info:
            for n in self.data_list:
                print(n.name)
        elif info:
            all_usage = 0.0
            for n in self.data_list:
                usage = n.data.memory_usage().sum() / 1024 ** 2
                shape = n.shape
                columns = list(n.columns)
                all_usage += usage
                print(n.name + '\n - Shape: {}\n - Index:{}\n - Memory usage: {:.3f} MB\n'.format(shape, columns, usage))
            print('-----------------------------------\nTotal usage: {:.3f} MB'.format(all_usage))

    def shape(self):
        """
        Return the length of the list.
        """
        return len(self.data_list)

    def make_dataset(self, train_proportion: float = 0.0):
        """
        Create dataset from the list.
        """
        index_num = int(self.shape() * train_proportion)
        train_list = self.data_list[:index_num]
        test_list = self.data_list[index_num:]
        return OneList(train_list), OneList(test_list)

    def head(self, n: int = 5):
        """
        Return the head of the data_list.
        """
        return OneList(self.data_list[:n])
