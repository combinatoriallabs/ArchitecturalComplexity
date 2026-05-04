import torch
import torch.nn as nn
import os

try:
    from Arch.Models.ModelUtils import *
    from Arch.Models.Misc import *
except:
    from ModelUtils import *
    from Misc import *

#original codes taken from: https://github.com/huyvnphan/PyTorch_CIFAR10/tree/master and https://github.com/chenyaofo/pytorch-cifar-models/blob/master/pytorch_cifar_models/resnet.py
#Modifications are done to facilitate integration with other projects and to ensure ease of configuration and linearizable model structures
'''
models taken from: https://raw.githubusercontent.com/pytorch/vision/v0.9.1/torchvision/models/resnet.py
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
'''

#Path config
if os.environ["USER"] == "nickubuntu" or os.environ["USER"] == "nickubuntuworkstation":
    strModelDir = "/home/" + os.environ["USER"] + "/ImagenetPretrainedModels/"
elif os.environ["USER"] == "nicklaptop":
    strModelDir = "/home/nicklaptop/ImagenetPretrainedModels/"
else:
    strModelDir = "NOT SET UP YET!"
    print("Double check path config!")

__all__ = [
    "ResNet",
    "resnet9",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnet9H",
    "resnet18H",
    "resnet34H",
    "resnet50H",
    "resnet101H",
    "resnet152H",
    "resnet7C",
    "resnet20C",
    "resnet32C",
    "resnet44C",
    "resnet56C",
    "MiniResNet",
    "MiniResNetX2"
]

dUrls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def MakeResidualSection(block, inchannels, channels, blocks, stride=1, groups=1, width=64, batch_norm: bool = True, relu: torch.nn.Module = nn.ReLU(inplace=True)):
    norm_layer = nn.BatchNorm2d if batch_norm else None
    downsample = None
    if stride != 1 or inchannels != channels * block.expansion:
        if norm_layer: 
            downsample = nn.Sequential(
                conv1x1(inchannels, channels * block.expansion, stride),
                norm_layer(channels * block.expansion),
            )
        else:
            downsample = nn.Sequential(
                conv1x1(inchannels, channels * block.expansion, stride),
            )

    layers = []
    layers.append(
        block(
            inchannels,
            channels,
            stride,
            downsample,
            groups,
            width,
            norm_layer,
            relu,
        )
    )
    inchannels = channels * block.expansion
    for _ in range(1, blocks):
        layers.append(
            block(
                inchannels,
                channels,
                groups=groups,
                base_width=width,
                norm_layer=norm_layer,
            )
        )

    return nn.Sequential(*layers)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        norm_layer=None,
        relu = nn.ReLU(inplace=True),
    ):
        super(BasicBlock, self).__init__()
        #if norm_layer is None:
        #    norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes) if norm_layer else None
        self.relu = relu
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes) if norm_layer else None
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.bn1: out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.bn2: out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

#TODO: add switch-able batch norm for this guy as well
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        norm_layer=None,
        relu = nn.ReLU(inplace=True),
    ):
        super(Bottleneck, self).__init__()
        # if norm_layer is None:
        #     norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width) if norm_layer else None
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = norm_layer(width) if norm_layer else None
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion) if norm_layer else None
        self.relu = relu
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.bn1: out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.bn2: out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.bn3: out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        channels,
        num_classes=10,
        groups=1,
        batch_norm: bool = True,
        bInitialMaxPool: bool = True,
        bInitWeights: bool = True,
        bImNet: bool = False,
        **kwargs,
    ):
        super(ResNet, self).__init__()
        
        if len(layers) not in  [3, 4] or len(channels) not in [3, 4] or len(layers) != len(channels):
            print("Error! ResNet() called with incorrect amounts of channels and layers!")
        
        self._norm_layer = nn.BatchNorm2d if batch_norm else None

        self.inplanes = 64
        self.groups = groups
        self.base_width = 64

        self.bCheckLinear = True #this tells IModel to automatically add flattens in the right places

        if bImNet: self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        else: self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = self._norm_layer(self.inplanes) if batch_norm else None

        self.relu = nn.ReLU(inplace=True)
        if "kn" in kwargs.keys() and kwargs["kn"] is not None: self.relu = GenReLU(kwargs["kn"], kwargs["kp"])
        
        self.maxpool = None
        if bInitialMaxPool: self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = MakeResidualSection(block, self.inplanes, channels[0], layers[0], 1, 1, 64, batch_norm, self.relu)
        self.inplanes = channels[0] * block.expansion

        self.layer2 = MakeResidualSection(block, self.inplanes, channels[1], layers[1], 2, 1, 64, batch_norm, self.relu)
        self.inplanes = channels[1] * block.expansion

        self.layer3 = MakeResidualSection(block, self.inplanes, channels[2], layers[2], 2, 1, 64, batch_norm, self.relu)
        self.inplanes = channels[2] * block.expansion

        if len(layers) == 4:
            self.layer4 = MakeResidualSection(block, self.inplanes, channels[3], layers[3], 2, 1, 64, batch_norm, self.relu)
            self.inplanes = channels[3] * block.expansion
        else:
            self.layer4 = None
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(channels[-1] * block.expansion, num_classes)

        if bInitWeights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            # Zero-initialize the last BN in each residual branch,
            # so that the residual branch starts with zeros, and each residual block behaves like an identity.
            # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
            if batch_norm:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        nn.init.constant_(m.bn3.weight, 0)
                    elif isinstance(m, BasicBlock):
                        nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        if self.bn1: x = self.bn1(x)
        x = self.relu(x)
        if self.maxpool is not None: x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.layer4 is not None: x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        return x
    
    def classify(self, x):
        x = self.fc(x)
        return x
    
    def GetHeadGrad(self) -> torch.Tensor:
        '''
        Used for computing DPS between CE and FKD losses
        '''
        with torch.no_grad():
            return torch.matmul(self.fc.bias.grad.unsqueeze(0), self.fc.weight)


def _resnet(block, layers, channels, batch_norm: bool = True, strModel: str = None, num_classes: int = 10, bImNet: bool = False, **kwargs):
    model = ResNet(block, layers, channels, batch_norm=batch_norm, num_classes=num_classes, bInitWeights = True if strModel is None else False, 
                   bImNet = bImNet, **kwargs)
    
    if strModel is not None:
        if not os.path.exists(strModelDir + strModel):
            print("WARNING! Unable to load weights! Double check model path: {}".format(strModelDir + strModel))
            return model
        with open(strModelDir + strModel, "rb") as f:
            sd = torch.load(f, weights_only = False)
            if not bImNet:
                #overwrite a few weights because the architectures changed slightly from imagenet version
                sd["conv1.weight"] = torch.randn((64, 3, 3, 3))
                nn.init.kaiming_normal_(sd["conv1.weight"], mode="fan_out", nonlinearity="relu")
                sd["fc.weight"] = torch.randn((num_classes, channels[-1] * block.expansion))
                nn.init.normal_(sd["fc.weight"], 0, 0.01)
                sd["fc.bias"] = torch.zeros((num_classes))
            model.load_state_dict(sd)
    
    return model


def resnet9(batch_norm: bool = True, bImNet: bool = False, **kwargs):
    return _resnet(BasicBlock, [1, 1, 1, 1], [64, 128, 256, 512], batch_norm=batch_norm, bImNet = bImNet, **kwargs)

def resnet18(batch_norm: bool = True, bImNPreTrained: bool = False, **kwargs):
    strM = "resnet18.pth" if bImNPreTrained else None
    return _resnet(BasicBlock, [2, 2, 2, 2], [64, 128, 256, 512], batch_norm=batch_norm, strModel = strM, **kwargs)

def resnet34(batch_norm: bool = True, bImNPreTrained: bool = False, bImNet: bool = False, **kwargs):
    strM = "resnet34.pth" if bImNPreTrained else None
    return _resnet(BasicBlock, [3, 4, 6, 3], [64, 128, 256, 512], batch_norm=batch_norm, strModel = strM, bImNet = bImNet, **kwargs)

def resnet50(batch_norm: bool = True, bImNPreTrained: bool = False, bImNet: bool = False, **kwargs):
    strM = "resnet50.pth" if bImNPreTrained else None
    return _resnet(Bottleneck, [3, 4, 6, 3], [64, 128, 256, 512], batch_norm=batch_norm, strModel = strM, bImNet = bImNet, **kwargs)

def resnet101(batch_norm: bool = True, bImNPreTrained: bool = False, bImNet: bool = False, **kwargs):
    strM = "resnet101.pth" if bImNPreTrained else None
    return _resnet(Bottleneck, [3, 4, 23, 3], [64, 128, 256, 512], batch_norm=batch_norm, strModel = strM, bImNet = bImNet, **kwargs)

def resnet152(batch_norm: bool = True, bImNPreTrained: bool = False, bImNet: bool = False, **kwargs):
    strM = "resnet152.pth" if bImNPreTrained else None
    return _resnet(Bottleneck, [3, 8, 36, 3], [64, 128, 256, 512], batch_norm=batch_norm, strModel = strM, bImNet = bImNet, **kwargs)


#These are similar versions of the above models, with fewer channels
def resnet9H(batch_norm: bool = True, bImNet: bool = False, **kwargs):
    return _resnet(BasicBlock, [1, 1, 1, 1], [32, 64, 128, 256], batch_norm=batch_norm, bImNet = bImNet, **kwargs)

def resnet18H(batch_norm: bool = True, bImNet: bool = False, **kwargs):
    return _resnet(BasicBlock, [2, 2, 2, 2], [32, 64, 128, 256], batch_norm=batch_norm, bImNet = bImNet, **kwargs)

def resnet34H(batch_norm: bool = True, bImNet: bool = False, **kwargs):
    return _resnet(BasicBlock, [3, 4, 6, 3], [32, 64, 128, 256], batch_norm=batch_norm, bImNet = bImNet, **kwargs)
def resnet34Q(batch_norm: bool = True, bImNet: bool = False, **kwargs):
    return _resnet(BasicBlock, [3, 4, 6, 3], [16, 32, 64, 128], batch_norm=batch_norm, bImNet = bImNet, **kwargs)
def resnet34S(batch_norm: bool = True, bImNet: bool = False, **kwargs):
    return _resnet(BasicBlock, [3, 4, 6, 3], [12, 24, 48, 96], batch_norm=batch_norm, bImNet = bImNet, **kwargs)
def resnet34S2(batch_norm: bool = True, bImNet: bool = False, **kwargs):
    return _resnet(BasicBlock, [3, 4, 6, 3], [10, 20, 40, 80], batch_norm=batch_norm, bImNet = bImNet, **kwargs)
def resnet34QQ(batch_norm: bool = True, bImNet: bool = False, **kwargs):
    return _resnet(BasicBlock, [3, 4, 6, 3], [8, 16, 32, 64], batch_norm=batch_norm, bImNet = bImNet, **kwargs)
def resnet34SS(batch_norm: bool = True, bImNet: bool = False, **kwargs):
    return _resnet(BasicBlock, [3, 4, 6, 3], [6, 12, 24, 48], batch_norm=batch_norm, bImNet = bImNet, **kwargs)
def resnet34QQQ(batch_norm: bool = True, bImNet: bool = False, **kwargs):
    return _resnet(BasicBlock, [3, 4, 6, 3], [4, 8, 16, 32], batch_norm=batch_norm, bImNet = bImNet, **kwargs)
def resnet34QQQQ(batch_norm: bool = True, bImNet: bool = False, **kwargs):
    return _resnet(BasicBlock, [3, 4, 6, 3], [2, 4, 8, 16], batch_norm=batch_norm, bImNet = bImNet, **kwargs)

def resnet50H(batch_norm: bool = True, bImNet: bool = False, **kwargs):
    return _resnet(Bottleneck, [3, 4, 6, 3], [32, 64, 128, 256], batch_norm=batch_norm, bImNet = bImNet, **kwargs)

def resnet101H(batch_norm: bool = True, bImNet: bool = False, **kwargs):
    return _resnet(Bottleneck, [3, 4, 23, 3], [32, 64, 128, 256], batch_norm=batch_norm, bImNet = bImNet, **kwargs)

def resnet152H(batch_norm: bool = True, bImNet: bool = False, **kwargs):
    return _resnet(Bottleneck, [3, 8, 36, 3], [32, 64, 128, 256], batch_norm=batch_norm, bImNet = bImNet, **kwargs)


#These are similar versions of the above models, with 3 sections and fewer channels
#20C, 32C, 44C, and 56C are reimplemented from: https://github.com/chenyaofo/pytorch-cifar-models/blob/master/pytorch_cifar_models/resnet.py
#7C is inspired by the above, and provides a super small architecture for distillation experiments
def resnet7C(batch_norm: bool = True, **kwargs):
    return _resnet(BasicBlock, [1, 1, 1], [16, 32, 64], batch_norm=batch_norm, **kwargs)

def resnet20C(batch_norm: bool = True, **kwargs):
    return _resnet(BasicBlock, [3, 3, 3], [16, 32, 64], batch_norm=batch_norm, **kwargs)

def resnet32C(batch_norm: bool = True, **kwargs):
    return _resnet(BasicBlock, [5, 5, 5], [16, 32, 64], batch_norm=batch_norm, **kwargs)

def resnet44C(batch_norm: bool = True, **kwargs):
    return _resnet(BasicBlock, [7, 7, 7], [16, 32, 64], batch_norm=batch_norm, **kwargs)

def resnet56C(batch_norm: bool = True, **kwargs):
    return _resnet(Bottleneck, [9, 9, 9], [16, 32, 64], batch_norm=batch_norm, **kwargs)



class MiniResNet(torch.nn.Module):
    def __init__(self, iC: int = 10):
        super(MiniResNet, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, 1, 1),
            torch.nn.LeakyReLU(),
            BasicBlock(16, 16),
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 5, 2, 2),
            torch.nn.LeakyReLU(),
            BasicBlock(32, 32),
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 5, 2, 2),
            torch.nn.LeakyReLU(),
            BasicBlock(64, 64),
        )

        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 96, 5, 2, 2),
            torch.nn.LeakyReLU(),
            BasicBlock(96, 96),
        )

        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(96, 128, 5, 2, 2),
            torch.nn.LeakyReLU(),
            BasicBlock(128, 128),
            torch.nn.Conv2d(128, 128, 2, 2, 0)
        )

        self.head = torch.nn.Sequential(
            torch.nn.Linear(128, iC),
            torch.nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x.view(x.shape[0], -1)
    
    def classify(self, x):
        return self.head(x)
    
    
class MiniResNetX2(torch.nn.Module):
    def __init__(self, iC: int = 10):
        super(MiniResNetX2, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1),
            torch.nn.LeakyReLU(),
            BasicBlock(32, 32),
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 5, 2, 2),
            torch.nn.LeakyReLU(),
            BasicBlock(64, 64),
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 5, 2, 2),
            torch.nn.LeakyReLU(),
            BasicBlock(128, 128),
        )

        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 192, 5, 2, 2),
            torch.nn.LeakyReLU(),
            BasicBlock(192, 192),
        )

        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(192, 256, 5, 2, 2),
            torch.nn.LeakyReLU(),
            BasicBlock(256, 256),
            torch.nn.Conv2d(256, 256, 2, 2, 0)
        )

        self.head = torch.nn.Sequential(
            torch.nn.Linear(256, iC),
            torch.nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x.view(x.shape[0], -1)
    
    def classify(self, x):
        return self.head(x)
    


def main():
    #model = resnet34(bInitialMaxPool = True, bImNPreTrained = False)
    #x = torch.zeros((1, 3, 224, 224))

    model = resnet34H(bInitialMaxPool = True, num_classes = 200)
    x = torch.randn((2, 3, 64, 64))
    PrintModelSummary(model, x)

    from IModel import IModel
    iModel = IModel(model)

    # f = iModel.features(x)
    # c = iModel.classify(f)

if __name__ == "__main__":
    main()