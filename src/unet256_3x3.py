
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# we can play around with dilation variable with is currently set to 1
def make_conv_layer_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return [
        # the bias term is not need because batch normalisation already takes care of that
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        ]

def make_conv_layer_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True),
        nn.ReLu(inplace=True)
        ]


class UNet256_3x3(nn.Module):

    def __init__(self, in_shape, num_classes):
        super(UNet256_3x3, self).__init__()
        in_channels, height, width = in_shape

        self.down1 = nn.Sequential(
                *make_conv_layer_bn_relu(in_channels, 16, kernel_size=3, stride=1, padding=1),
                *make_conv_layer_bn_relu(16, 32, kernel_size=3, stride=2, padding=1)
                )

        self.down2 = nn.Sequential(
                *make_conv_layer_bn_relu(32, 64, kernel_size=3, stride=1, padding=1),
                *make_conv_layer_bn_relu(64, 128, kernel_size=3, stride=1, padding=1)
                )

        self.down3 = nn.Sequential(
                *make_conv_layer_bn_relu(128, 256, kernel_size=3, stride=1, padding=1),
                *make_conv_layer_bn_relu(256, 512, kernel_size=3, stride=1, padding=1)
                )

        self.down4 = nn.Sequential(
                *make_conv_layer_bn_relu(512, 512, kernel_size=3, stride=1, padding=1),
                *make_conv_layer_bn_relu(512, 512, kernel_size=3, stride=1, padding=1)
                )

        self.same = nn.Sequential(
                *make_conv_layer_bn_relu(512, 512, kernel_size=3, stride=1, padding=1)
                )

        self.up4 = nn.Sequential(
                *make_conv_layer_bn_relu(1024, 512, kernel_size=3, stride=1, padding=1),
                *make_conv_layer_bn_relu(512, 512, kernel_size=3, stride=1, padding=1)
                )

        self.up3 = nn.Sequential(
                *make_conv_layer_bn_relu(1024, 512, kernel_size=3, stride=1, padding=1),
                *make_conv_layer_bn_relu(512, 128, kernel_size=3, stride=1, padding=1)
                )

        self.up2 = nn.Sequential(
                *make_conv_layer_bn_relu(256, 128, kernel_size=3, stride=1, padding=1),
                *make_conv_layer_bn_relu(128, 32, kernel_size=3, stride=1, padding=1)
                )

        self.up1 = nn.Sequential(
                *make_conv_layer_bn_relu(64, 64, kernel_size=3, stride=1, padding=1),
                *make_conv_layer_bn_relu(64, 32, kernel_size=3, stride=1, padding=1)
                )

        self.up0 = nn.Sequential(
                *make_conv_layer_bn_relu(32, 32, kernel_size=3, stride=1, padding=1)
                )

        self.classify = nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0)



    def forward(self, x):
        # x is (3, 256, 256)
        down1 = self.down1(x) # down1 is (32, 128, 128)
        out = F.max_pool2d(down1, kernel_size=2, stride=2) # out is (32, 64, 64)

        down2 = self.down2(out) # down2 is (128, 64, 64)
        out = F.max_pool2d(down2, kernel_size=2, stride=2) # out is (128, 32, 32)

        down3 = self.down3(out) # down3 is (512, 32, 32)
        out = F.max_pool2d(down3, kernel_size=2, stride=2) # out is (512, 16, 16)

        down4 = self.down4(out) # down4 is (512, 16, 16)
        out = F.max_pool2d(down4, kernel_size=2, stride=2) # out is (512, 8, 8)

        out = self.same(out) # down4 is (512, 8, 8)

        out = F.upsample(out, scale_factor=2, mode='bilinear') # out is (512, 16, 16)
        out = torch.cat([down4, out], 1) # out is (1024, 16, 16)
        out = self.up4(out) # out is (512, 16, 16)

        out = F.upsample(out, scale_factor=2, mode='bilinear') # out is (512, 32, 32)
        out = torch.cat([down3, out], 1) # out is (1024, 32, 32)
        out = self.up3(out) # out is (128, 32, 32)

        out = F.upsample(out, scale_factor=2, mode='bilinear') # out is (128, 64, 64)
        out = torch.cat([down2, out], 1)# out is (256, 64, 64)
        out = self.up2(out)# out is (32, 64, 64)

        out = F.upsample(out, scale_factor=2, mode='bilinear') # out is (32, 128, 128)
        out = torch.cat([down1, out], 1) # out is (64, 128, 128)
        out = self.up1(out) # out is (32, 128, 128)

        out = F.upsample(out, scale_factor=2, mode='bilinear') # out is (32, 256, 256)
        out = self.up0(out) # out is (32, 128, 128)

        out = self.classify(out)
        return out

if __name__ == "__main__":
    a = UNet256_3x3((3,256,256), 3)
    a.train()
    a.eval()
