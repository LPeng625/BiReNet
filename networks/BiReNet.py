import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from functools import partial
from networks import ffm, edm


nonlinearity = partial(F.relu, inplace=True)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class BiReNet34(nn.Module):
    def __init__(self, out_channels=1):
        super(BiReNet34, self).__init__()
        self.is_Train = True
        self.has_FFM = True
        self.has_EDM = True
        self.has_AuxHead = True

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=False)
        # resnet.load_state_dict(torch.load('resnet34.pth'))
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        if self.has_FFM:
            self.FeatureFusionModule1 = ffm.FeatureFusionModule(
                filters[0],
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU'),
                init_cfg=None)

            self.FeatureFusionModule2 = ffm.FeatureFusionModule(
                filters[1],
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU'),
                init_cfg=None)

            self.FeatureFusionModule3 = ffm.FeatureFusionModule(
                filters[2],
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU'),
                init_cfg=None)

        if self.has_EDM:
            self.EDM = edm.init('carv4', is_ori=True, inplane=64)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, out_channels, 3, padding=1)

    def forward(self, x):
        # EDM
        if self.has_EDM:
            edge = self.EDM(x)

        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)

        # Encoder
        e1 = self.firstmaxpool(x)
        e1 = self.encoder1(e1)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder
        if self.has_FFM:
            d4 = self.stdc.FeatureFusionModule3(self.decoder4(e4), e3)
            d3 = self.stdc.FeatureFusionModule2(self.decoder3(d4), e2)
            d2 = self.stdc.FeatureFusionModule1(self.decoder2(d3), e1)
        else:
            d4 = self.decoder4(e4) + e3
            d3 = self.decoder3(d4) + e2
            d2 = self.decoder2(d3) + e1

        if self.has_EDM:
            d1 = self.decoder1(d2) + edge
        else:
            d1 = self.decoder1(d2)

        out = self.finalconv3(self.finalrelu2(self.finalconv2(self.finalrelu1(self.finaldeconv1(
            d1)))))
        out = F.sigmoid(out)

        if self.is_Train:
            if self.has_EDM and self.has_AuxHead:
                out_e = self.finalconv3(self.finalrelu2(self.finalconv2(self.finalrelu1(self.finaldeconv1(edge)))))
                out_e = F.sigmoid(out_e)
                outs = [out] + [out_e]
                outs = [outs[i] for i in (0, 1)]
            else:
                outs = [out]
                outs = [outs[i] for i in (0,)]
            return tuple(outs)
        else:
            return out

