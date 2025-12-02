import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision.models as models
from model.MobileNetV3 import mobilenetv3_large


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class DWBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(DWBasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=in_planes, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class DWConv3(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DWConv3, self).__init__()
        self.conv = nn.Sequential(
            DWBasicConv2d(in_planes, in_planes, 3 , padding=1),
            BasicConv2d(in_planes, out_planes, 1)
        )

    def forward(self, x):
        return self.conv(x)


class Reduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Reduction, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out

        return self.sigmoid(out)


class ChannelAttention_no_sig(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention_no_sig, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out

        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avg_out, max_out], 1)
        x2 = self.conv1(x1)

        return self.sigmoid(x2)


class SpatialAttention_no_sig(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention_no_sig, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avg_out, max_out], 1)
        x2 = self.conv1(x1)

        return x2


class SPEM(nn.Module):
    def __init__(self, channel):
        super(SPEM, self).__init__()
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_cat = DWConv3(channel * 2, channel)
        self.conv_out = nn.Sequential(
            DWBasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )

    def forward(self, s1, s2):
        x_si = self.conv_cat(torch.cat((s1, self.up2(s2)), dim=1))
        A_si = self.conv_out(x_si)

        return A_si


class SEEM(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEEM, self).__init__()
        self.down2 = nn.MaxPool2d(2, stride=2)
        self.conv_cat = DWBasicConv2d(channel * 2, channel * 2, 3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_out = nn.Sequential(
            nn.Conv2d(channel * 2, channel // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=True)
        )

    def forward(self, s4, s5):
        x_45 = self.conv_cat(torch.cat((self.down2(s4), s5), dim=1))
        A_ci = self.conv_out(self.avgpool(x_45))

        return A_ci


class SCorrM(nn.Module):
    def __init__(self):
        super(SCorrM, self).__init__()
        self.smooth = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        nn.init.constant_(self.smooth.weight, 1/9)
        self.smooth.weight.requires_grad = False

        self.conv_fuse = nn.Sequential(
            nn.Conv2d(2, 1, 3, padding=1, bias=False),
            nn.BatchNorm2d(1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, sa, A_si):
        A_si1 = F.interpolate(A_si, size=sa.size()[2:], mode='bilinear', align_corners=False)
        A_si1 = self.smooth(A_si1)

        W_sa = self.sigmoid(sa * A_si1)
        sa_f = W_sa * sa + (1 - W_sa) * A_si1

        sa_diff = torch.abs(sa - A_si1)

        A_scorr = self.conv_fuse(torch.cat([sa_f, sa_diff], dim=1))

        return self.sigmoid(A_scorr)


class CCorrM(nn.Module):
    def __init__(self, channel, reduction=4):
        super(CCorrM, self).__init__()
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(channel * 2, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, ca, A_ci):
        W_ca = self.sigmoid(ca * A_ci)
        ca_f = W_ca * ca + (1 - W_ca) * A_ci

        ca_diff = torch.abs(ca - A_ci)

        A_ccorr = self.conv_fuse(torch.cat([ca_f, ca_diff], dim=1))

        return self.sigmoid(A_ccorr)


class SCAFM(nn.Module):
    def __init__(self, channel):
        super(SCAFM, self).__init__()
        self.sa_no_sig = SpatialAttention_no_sig()
        self.ca_no_sig = ChannelAttention_no_sig(channel)

        self.scorr = SCorrM()
        self.ccorr = CCorrM(channel)

        self.refine = nn.Sequential(
            DWBasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.Sigmoid()
        )

    def forward(self, x_mix, A_si, A_ci):
        x_mix_sa = self.sa_no_sig(x_mix)
        x_mix_ca = self.ca_no_sig(x_mix)

        A_scorr = self.scorr(x_mix_sa, A_si)
        A_ccorr = self.ccorr(x_mix_ca, A_ci)

        A_s_g = torch.mean(A_scorr, dim=(2, 3), keepdim=True)
        A_ccorr = A_s_g * A_ccorr

        A_c_g = torch.mean(A_ccorr, dim=1, keepdim=True)
        A_scorr = A_c_g * A_scorr

        A_fuse = A_ccorr * A_scorr
        A_fuse = self.refine(A_fuse)

        x_scafm = A_fuse * x_mix

        return x_scafm


class SSIM1(nn.Module):
    def __init__(self, channel):
        super(SSIM1, self).__init__()
        channel_sp = channel // 4

        self.branch1 = nn.Sequential(
            BasicConv2d(channel_sp, channel_sp, 1),
            BasicConv2d(channel_sp, channel_sp, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(channel_sp, channel_sp, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(channel_sp, channel_sp, 3, padding=3, dilation=3)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(channel_sp, channel_sp, 1),
            BasicConv2d(channel_sp, channel_sp, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(channel_sp, channel_sp, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(channel_sp, channel_sp, 3, padding=5, dilation=5)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(channel_sp, channel_sp, 1),
            BasicConv2d(channel_sp, channel_sp, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(channel_sp, channel_sp, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(channel_sp, channel_sp, 3, padding=7, dilation=7)
        )

        self.branch4 = nn.Sequential(
            BasicConv2d(channel_sp, channel_sp, 1),
            BasicConv2d(channel_sp, channel_sp, kernel_size=(1, 9), padding=(0, 4)),
            BasicConv2d(channel_sp, channel_sp, kernel_size=(9, 1), padding=(4, 0)),
            BasicConv2d(channel_sp, channel_sp, 3, padding=9, dilation=9)
        )

        self.conv_cat = BasicConv2d(channel, channel, 1)

        self.scafm = SCAFM(channel)

    def forward(self, x_in, A_si, A_ci):
        x_s = torch.chunk(x_in, chunks=4, dim=1)
        x_1, x_2, x_3, x_4 = x_s

        x_m1 = self.branch1(x_1)
        x_m2 = self.branch2(x_2 + x_m1)
        x_m3 = self.branch3(x_3 + x_m1 + x_m2)
        x_m4 = self.branch4(x_4 + x_m1 + x_m2 + x_m3)

        x_mix = self.conv_cat(torch.cat([x_m1, x_m2, x_m3, x_m4], dim=1))

        x_scafm = self.scafm(x_mix, A_si, A_ci)

        x_ssim = x_scafm + x_in

        return x_ssim


class SSIM(nn.Module):
    def __init__(self, channel):
        super(SSIM, self).__init__()
        self.down2 = nn.MaxPool2d(2, stride=2)
        self.sa_pre = SpatialAttention()

        channel_sp = channel // 4

        self.branch1 = nn.Sequential(
            BasicConv2d(channel_sp, channel_sp, 1),
            BasicConv2d(channel_sp, channel_sp, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(channel_sp, channel_sp, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(channel_sp, channel_sp, 3, padding=3, dilation=3)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(channel_sp, channel_sp, 1),
            BasicConv2d(channel_sp, channel_sp, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(channel_sp, channel_sp, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(channel_sp, channel_sp, 3, padding=5, dilation=5)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(channel_sp, channel_sp, 1),
            BasicConv2d(channel_sp, channel_sp, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(channel_sp, channel_sp, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(channel_sp, channel_sp, 3, padding=7, dilation=7)
        )

        self.branch4 = nn.Sequential(
            BasicConv2d(channel_sp, channel_sp, 1),
            BasicConv2d(channel_sp, channel_sp, kernel_size=(1, 9), padding=(0, 4)),
            BasicConv2d(channel_sp, channel_sp, kernel_size=(9, 1), padding=(4, 0)),
            BasicConv2d(channel_sp, channel_sp, 3, padding=9, dilation=9)
        )

        self.conv_cat = BasicConv2d(channel, channel, 1)

        self.scafm = SCAFM(channel)

    def forward(self, x_pre, x_in, A_si, A_ci):
        x_sa = (self.sa_pre(self.down2(x_pre))) * x_in + x_in

        x_s = torch.chunk(x_sa, chunks=4, dim=1)
        x_1, x_2, x_3, x_4 = x_s

        x_m1 = self.branch1(x_1)
        x_m2 = self.branch2(x_2 + x_m1)
        x_m3 = self.branch3(x_3 + x_m1 + x_m2)
        x_m4 = self.branch4(x_4 + x_m1 + x_m2 + x_m3)

        x_mix = self.conv_cat(torch.cat([x_m1, x_m2, x_m3, x_m4], dim=1))

        x_scafm = self.scafm(x_mix, A_si, A_ci)

        x_ssim = x_scafm + x_sa

        return x_ssim


class PD(nn.Module):
    def __init__(self, channel):
        super(PD, self).__init__()
        self.ca = ChannelAttention(channel)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_out = DWConv3(channel * 2, channel)

    def forward(self, x_in, x_lat, pd_lat=None):
        x1 = self.ca(x_lat) * x_in + x_in
        if pd_lat is not None:
            x_pd = self.conv_out(torch.cat((x1, self.up2(pd_lat)), dim=1))
        else:
            x_pd = self.conv_out(torch.cat((x1, self.up2(x_lat)), dim=1))

        return x_pd


class SSINet(nn.Module):
    def __init__(self, channel=32):
        super(SSINet, self).__init__()
        # Backbone model
        self.encoder = mobilenetv3_large()
        self.encoder.load_state_dict(torch.load('./model/mobilenetv3-large-1cd25616.pth'), strict=False)

        self.Translayer1 = Reduction(16, channel)
        self.Translayer2 = Reduction(24, channel)
        self.Translayer3 = Reduction(40, channel)
        self.Translayer4 = Reduction(112, channel)
        self.Translayer5 = Reduction(960, channel)

        self.spem = SPEM(channel)
        self.seem = SEEM(channel)

        self.ssim1 = SSIM1(channel)
        self.ssim2 = SSIM(channel)
        self.ssim3 = SSIM(channel)
        self.ssim4 = SSIM(channel)
        self.ssim5 = SSIM(channel)

        self.pd1 = PD(channel)
        self.pd2 = PD(channel)
        self.pd3 = PD(channel)
        self.pd4 = PD(channel)

        self.s_conv1 = nn.Conv2d(channel, 1, 3, padding=1)
        self.s_conv2 = nn.Conv2d(channel, 1, 3, padding=1)
        self.s_conv3 = nn.Conv2d(channel, 1, 3, padding=1)
        self.s_conv4 = nn.Conv2d(channel, 1, 3, padding=1)
        self.s_conv5 = nn.Conv2d(channel, 1, 3, padding=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        size = x.size()[2:]
        f1, f2, f3, f4, f5 = self.encoder(x)

        s1 = self.Translayer1(f1)
        s2 = self.Translayer2(f2)
        s3 = self.Translayer3(f3)
        s4 = self.Translayer4(f4)
        s5 = self.Translayer5(f5)

        A_si = self.spem(s1, s2)
        A_ci = self.seem(s4, s5)

        x_ssim5 = self.ssim5(s4, s5, A_si, A_ci)
        x_ssim4 = self.ssim4(s3, s4, A_si, A_ci)
        x_ssim3 = self.ssim3(s2, s3, A_si, A_ci)
        x_ssim2 = self.ssim2(s1, s2, A_si, A_ci)
        x_ssim1 = self.ssim1(s1, A_si, A_ci)

        x_pd4 = self.pd4(x_ssim4, x_ssim5)
        x_pd3 = self.pd3(x_ssim3, x_ssim4, x_pd4)
        x_pd2 = self.pd2(x_ssim2, x_ssim3, x_pd3)
        x_pd1 = self.pd1(x_ssim1, x_ssim2, x_pd2)

        x5_out = self.s_conv5(x_ssim5)
        x4_out = self.s_conv4(x_pd4)
        x3_out = self.s_conv3(x_pd3)
        x2_out = self.s_conv2(x_pd2)
        sal_out = self.s_conv1(x_pd1)

        x5_out = F.interpolate(x5_out, size=size, mode='bilinear', align_corners=True)
        x4_out = F.interpolate(x4_out, size=size, mode='bilinear', align_corners=True)
        x3_out = F.interpolate(x3_out, size=size, mode='bilinear', align_corners=True)
        x2_out = F.interpolate(x2_out, size=size, mode='bilinear', align_corners=True)
        sal_out = F.interpolate(sal_out, size=size, mode='bilinear', align_corners=True)
        edg1 = F.interpolate(A_si, size=size, mode='bilinear', align_corners=True)

        return sal_out, self.sigmoid(sal_out), x2_out, self.sigmoid(x2_out), x3_out, self.sigmoid(x3_out), x4_out, self.sigmoid(x4_out), x5_out, self.sigmoid(x5_out), edg1
