from torch import nn

from .net import Net


class OneNet:
    def __init__(self):
        self.module = Net()
        self.batch_size = None
        self.epoch = None
        self.lr = None

        self.optimizer = None
        self.loss_func = None

    def __repr__(self):
        print(self.module.layers)
        return ''

    def __getitem__(self, item):
        return self.module.layers[item]

    def add(self, name: str, module: nn.Module):
        self.module.layers.add_module(name, module)

    def options(self, options: dict):
        for option, n in options.items():
            if option == 'optimizer':
                self.optimizer = n
            elif option == 'loss':
                self.loss_func = n
