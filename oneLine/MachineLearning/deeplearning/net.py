from torch import nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential()

    def forward(self, x):
        x = self.layers(x)
        return x
