import torch
import torch.nn as nn
import torch.nn.functional as F

class bottleneck(nn.Module):

    def __init__(self, in_planes, dim, out_planes, num_classes):
        super(bottleneck, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_planes, dim),
            nn.ReLU(True),

            nn.Linear(dim, out_planes),
            nn.ReLU(True),

            nn.Linear(out_planes, num_classes)
        )

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
        x = self.fc(x)
        return x