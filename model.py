import torch
import torch.nn as nn
import torch.nn.functional as F

def upsample_like(x, target):
    return F.interpolate(x,size=target.shape[2:],mode="bilinear",align_corners=True)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace = True)
        )

    def forward(self, x):
        return self.block(x)

class FLnet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features = (64, 128, 256, 512, 1024)):
        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv0_0 = ConvBlock(in_channels=in_channels, out_channels=features[0])
        self.conv1_0 = ConvBlock(in_channels=features[0], out_channels=features[1])
        self.conv2_0 = ConvBlock(in_channels=features[1], out_channels=features[2])
        self.conv3_0 = ConvBlock(in_channels=features[2], out_channels=features[3])
        self.conv4_0 = ConvBlock(in_channels=features[3], out_channels=features[4])

        self.conv0_1 = ConvBlock(in_channels = features[0] + features[1], out_channels=features[0])
        self.conv0_2 = ConvBlock(in_channels = features[0] * 2 + features[1], out_channels=features[0])
        self.conv0_3 = ConvBlock(in_channels = features[0] * 3 + features[1], out_channels=features[0])
        self.conv0_4 = ConvBlock(in_channels = features[0] * 4 + features[1], out_channels=features[0])

        self.conv1_1 = ConvBlock(in_channels = features[1] + features[2], out_channels=features[1])
        self.conv1_2 = ConvBlock(in_channels = features[1] * 2 + features[2], out_channels=features[1])
        self.conv1_3 = ConvBlock(in_channels = features[1] * 3 + features[2], out_channels=features[1])

        self.conv2_1 = ConvBlock(in_channels = features[2] + features[3], out_channels=features[2])
        self.conv2_2 = ConvBlock(in_channels = features[2] * 2 + features[3], out_channels=features[2])

        self.conv3_1 = ConvBlock(in_channels = features[3] + features[4], out_channels=features[3])

        self.last_layer = nn.Conv2d(in_channels=features[0], out_channels=out_channels, kernel_size=1)

    def forward(self, x):

        x0_0 = self.conv0_0(x)

        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, upsample_like(x1_0, x0_0)], dim=1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, upsample_like(x2_0, x1_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, upsample_like(x1_1, x0_0)], dim=1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, upsample_like(x3_0, x2_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, upsample_like(x2_1, x1_0)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, upsample_like(x1_2, x0_0)], dim=1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, upsample_like(x4_0, x3_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, upsample_like(x3_1, x2_0)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, upsample_like(x2_2, x1_0)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, upsample_like(x1_3, x0_0)], dim=1))

        output = self.last_layer(x0_4)
        return output

def build_model(in_channels=3, out_channels=1):
  return FLnet(in_channels = in_channels, out_channels = out_channels)
