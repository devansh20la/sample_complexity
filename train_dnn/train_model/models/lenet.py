'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, filters):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, filters[0], 5)
        self.conv2 = nn.Conv2d(filters[0], filters[1], 5)
        self.fc1 = nn.Linear(filters[1]*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

if __name__ == "__main__":
    net = LeNet([1, 1])
    summary(net, input_size=(1, 28, 28))