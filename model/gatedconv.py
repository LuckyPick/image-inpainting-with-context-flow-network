import torch
from torch import nn


class GatedConv(nn.Module):
    def __init__(self, cin, cout, k_size,
                 stride=1, dilation=1, padding=1, activation='relu'):
        super(GatedConv, self).__init__()
        self.conv = nn.Conv2d(cin, 2 * cout, kernel_size=k_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.act1 = nn.ReLU(inplace=False)
        self.act2 = nn.Sigmoid()

    def forward(self, xin):
        x = self.conv(xin)
        x, y = torch.chunk(x, 2, dim=1)
        x = self.act1(x)
        y = self.act2(y)
        x_out = x * y
        return x_out


if __name__ == '__main__':
    x = torch.rand((10, 3, 20, 20))
    model = GatedConv(3, 4, 3,)
    y = model(x)
    print(y.shape)
