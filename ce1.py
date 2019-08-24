import torch.nn as nn
import torch


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.lstmlayer = nn.LSTM(28, 40, num_layers=2)
        self.linearlayer = nn.Linear(40, 10)

    def forward(self, x):
        # [ 28, 1, 28] => [28, 1, 28]
        output, (hidden, cell) = self.lstmlayer(x)
        output = hidden[-1, :, :]
        output = self.linearlayer(output)
        return output


if __name__ == '__main__':
    x = torch.randn(28, 1, 28)
    net = RNN()
    output = net(x)
    print(output.argmax())