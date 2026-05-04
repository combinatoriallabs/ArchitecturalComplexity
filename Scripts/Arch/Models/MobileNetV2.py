#Taken and modified from: https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py
#This version is cleaned up and has a linearizable model structure

import torch
import os
import math

#Path config
if os.environ["USER"] == "nickubuntu" or os.environ["USER"] == "nickubuntuworkstation":
    strModelDir = "/home/" + os.environ["USER"] + "/ImagenetPretrainedModels/"
elif os.environ["USER"] == "nicklaptop":
    strModelDir = "/home/nicklaptop/ImagenetPretrainedModels/"
else:
    strModelDir = "NOT SET UP YET!"
    print("Double check path config!")

def conv_bn(inp, oup, stride):
    return torch.nn.Sequential(
        torch.nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        torch.nn.BatchNorm2d(oup),
        torch.nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return torch.nn.Sequential(
        torch.nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        torch.nn.BatchNorm2d(oup),
        torch.nn.ReLU6(inplace=True)
    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(torch.nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = torch.nn.Sequential(
                # dw
                torch.nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                torch.nn.BatchNorm2d(hidden_dim),
                torch.nn.ReLU6(inplace=True),
                # pw-linear
                torch.nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(oup),
            )
        else:
            self.conv = torch.nn.Sequential(
                # pw
                torch.nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(hidden_dim),
                torch.nn.ReLU6(inplace=True),
                # dw
                torch.nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                torch.nn.BatchNorm2d(hidden_dim),
                torch.nn.ReLU6(inplace=True),
                # pw-linear
                torch.nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(torch.nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual

        self.bCheckLinear = True #this tells IModel to automatically add flattens in the right places
        
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2 if input_size > 32 else 1)] #don't "pool" down right off the bat if dealing with small images
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = torch.nn.Sequential(*self.features)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        # building classifier
        self.classifier = torch.nn.Linear(self.last_channel, n_class)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenet_v2(bImNPreTrained: bool = False, iSz: int = 224, iCls: int = 1000):
    model = MobileNetV2(width_mult=1, input_size = iSz, n_class = iCls,)

    if bImNPreTrained:
        with open(strModelDir + "mobilenet_v2.pth", "rb") as f:
            sd = torch.load(f, weights_only = False)

        #manually setup the final linear weights as n_class is not 1000 for our purposes
        sd["classifier.weight"] = torch.randn((iCls, 1280))
        torch.nn.init.normal_(sd["classifier.weight"], 0, 0.01)
        sd["classifier.bias"] = torch.zeros((iCls))

        model.load_state_dict(sd, strict = False)

    return model

if __name__=="__main__":
    # model check
    model = mobilenet_v2(iCls = 10, iSz = 32)