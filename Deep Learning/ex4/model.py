import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.res_conv = True
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        if stride == 1 and in_channels == out_channels:
            self.res_conv = False

        self.bn3 = nn.BatchNorm2d(out_channels)
        self.identity = None

    def forward(self, x):
        self.identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.res_conv:
            self.identity = self.conv1x1(self.identity)

        # self.identity = self.bn3(self.identity)
        out += self.identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.layer1 = ResBlock(64, 64, 1)
        self.layer2 = ResBlock(64, 128, 2)
        self.layer3 = ResBlock(128, 256, 2)
        self.layer4 = ResBlock(256, 512, 2)

        self.avgpool = nn.AvgPool2d(kernel_size=10)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x

