import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os


class fcnet(nn.Module):
    def __init__(self, args):

        super(fcnet, self).__init__()

        if args.model_depth > 1:
            layers = [nn.Linear(28 * 28, args.model_width[0]),
                      nn.ReLU()]

            for layer_idx in range(1, args.model_depth - 1):
                layers.append(nn.Linear(args.model_width[layer_idx - 1],
                                        args.model_width[layer_idx]))
                layers.append(nn.ReLU())

            layers.append(nn.Linear(args.model_width[-2],
                                    args.model_width[-1]))
        else:
            layers = [nn.Linear(28 * 28, args.model_width[0])]

        self.layers = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.layers(x)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Sample Complexity Training')
    parser.add_argument('--lr', '--learning_rate', type=float,
                        default=0.1, help='initial learning rate')
    args = parser.parse_args()

    args.model_width = [20,10]
    args.model_depth = 2
    net = fcnet(args)
    print(net)
    net(torch.randn(1,1,28,28))