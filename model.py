import torch
import torch.nn as nn
import torch.nn.functional as F

def upsample_like(x, target):
    return F.interpolate(x,size=target.shape[2:],mode="bilinear",align_corners=True)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace = True),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),

            nn.LeakyReLU(0.1, inplace = True)
        )

    def forward(self, x):
        return self.block(x)

class FLnet(nn.Module):
    def __init__(self, out_channels=1, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.conv0_0 = nn.Sequential(resnet.conv1,  resnet.bn1,resnet.relu)
        self.pool0 = resnet.maxpool


        self.conv1_0 = resnet.layer1
        self.conv2_0 = resnet.layer2
        self.conv3_0 = resnet.layer3
        self.conv4_0 = resnet.layer4


        f0, f1, f2, f3, f4 = 64, 256, 512, 1024, 2048

        self.conv0_1 = ConvBlock(in_channels=f0 + f1, out_channels=f0)
        self.conv0_2 = ConvBlock(in_channels=f0 * 2 + f1, out_channels=f0)
        self.conv0_3 = ConvBlock(in_channels=f0 * 3 + f1, out_channels=f0)
        self.conv0_4 = ConvBlock(in_channels=f0 * 4 + f1, out_channels=f0)

        self.conv1_1 = ConvBlock(in_channels=f1 + f2, out_channels=f1)
        self.conv1_2 = ConvBlock(in_channels=f1 * 2 + f2, out_channels=f1)
        self.conv1_3 = ConvBlock(in_channels=f1 * 3 + f2, out_channels=f1)

        self.conv2_1 = ConvBlock(in_channels=f2 + f3, out_channels=f2)
        self.conv2_2 = ConvBlock(in_channels=f2 * 2 + f3, out_channels=f2)

        self.conv3_1 = ConvBlock(in_channels=f3 + f4, out_channels=f3)

        if self.deep_supervision:
            self.final1 = nn.Conv2d(f0, out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(f0, out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(f0, out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(f0, out_channels, kernel_size=1)
        else:
            self.last_layer = nn.Conv2d(f0, out_channels, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool0(x0_0))


        x0_1 = self.conv0_1(torch.cat([x0_0, upsample_like(x1_0, x0_0)], dim=1))

        x2_0 = self.conv2_0(x1_0)
        x1_1 = self.conv1_1(torch.cat([x1_0, upsample_like(x2_0, x1_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, upsample_like(x1_1, x0_0)], dim=1))

        x3_0 = self.conv3_0(x2_0)
        x2_1 = self.conv2_1(torch.cat([x2_0, upsample_like(x3_0, x2_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, upsample_like(x2_1, x1_0)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, upsample_like(x1_2, x0_0)], dim=1))

        x4_0 = self.conv4_0(x3_0)
        x3_1 = self.conv3_1(torch.cat([x3_0, upsample_like(x4_0, x3_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, upsample_like(x3_1, x2_0)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, upsample_like(x2_2, x1_0)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, upsample_like(x1_3, x0_0)], dim=1))

        if self.deep_supervision and self.training:
            return [self.final1(x0_1), self.final2(x0_2), self.final3(x0_3), self.final4(x0_4)]
        else:
            return self.final4(x0_4) if self.deep_supervision else self.last_layer(x0_4)
