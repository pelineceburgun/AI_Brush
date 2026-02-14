import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, ks, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv = ConvBNReLU(in_chan, out_chan)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_chan, out_chan, 1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.conv(x)
        atten = self.attention(feat)
        return feat * atten


class ContextPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18(weights=None)

        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)

        self.conv_avg = ConvBNReLU(512, 128, ks=1, padding=0)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        feat8 = self.resnet.layer1(x)
        feat16 = self.resnet.layer2(feat8)
        feat32 = self.resnet.layer3(feat16)
        feat32 = self.resnet.layer4(feat32)

        avg = self.avg_pool(feat32)
        avg = self.conv_avg(avg)
        avg = F.interpolate(avg, size=feat32.shape[2:], mode='bilinear', align_corners=True)

        feat32 = self.arm32(feat32) + avg
        feat16 = self.arm16(feat16)

        return feat16, feat32


class FeatureFusionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(256, 256, 1, bias=False)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.Sigmoid()
        )

    def forward(self, fsp, fcp):
        fcp = F.interpolate(fcp, size=fsp.shape[2:], mode='bilinear', align_corners=True)
        feat = torch.cat([fsp, fcp], dim=1)
        feat = self.relu(self.bn(self.conv1(feat)))
        atten = self.conv2(feat)
        return feat * atten


class BiSeNet(nn.Module):
    def __init__(self, n_classes=19):
        super().__init__()
        self.cp = ContextPath()
        self.ffm = FeatureFusionModule()

        self.conv_out = ConvBNReLU(256, 256)
        self.conv_out16 = ConvBNReLU(128, 256)
        self.conv_out32 = ConvBNReLU(128, 256)

        self.out = nn.Conv2d(256, n_classes, 1)

    def forward(self, x):
        feat16, feat32 = self.cp(x)
        feat_fuse = self.ffm(feat16, feat32)

        out = self.out(self.conv_out(feat_fuse))
        return out
