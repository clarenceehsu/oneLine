from ..modules.dl_torch import *

class OneNet:
    def __init__(self, def_net='BP', input_size=0, hidden_size=[], output_size=0, activation='ReLU'):
        self.net = ''
        self.def_net = def_net
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.activation = activation

    def build(self):
        if self.def_net == 'BP':
            self.net = torch.nn.Sequential()
            for n in len(self.hidden_size) - 1:
                if n == 0:
                    self.net.add_module(f'Conv_{ n }', nn.Linear(self.input_size, self.hidden_size[n]))
                    self.net.add_module(f'ReLU_{ n }', nn.ReLU())
                elif n == len(self.hidden_size) - 1:
                    self.net.add_module(f'Conv_{n}', nn.Linear(self.input_size, self.hidden_size[n]))
                else:
                    self.net.add_module(f'Conv_{ n }', nn.Linear(self.hidden_size[n - 1], self.hidden_size[n]))
                    self.net.add_module(f'ReLU_{ n }', nn.ReLU())

    def forward(self, x):
        x = self.net(x)
        return x

