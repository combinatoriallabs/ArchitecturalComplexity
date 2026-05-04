import torch
import torch.nn as nn

import os

#original code taken from: https://github.com/huyvnphan/PyTorch_CIFAR10/tree/master
#Modifications are done to facilitate integration with other projects

#weights taken from: https://raw.githubusercontent.com/pytorch/vision/v0.9.1/torchvision/models/vgg.py
'''
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
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
    "VGG",
    "vgg4",
    "vgg7",
    "vgg11",
    "vgg13",
    "vgg16",
    "vgg19",
]


class VGG(nn.Module):
    def __init__(self, features, iFinalChannels: int, num_classes=10, bFullSize: bool = False, init_weights=True):
        super(VGG, self).__init__()
        self.features = features

        self.bCheckLinear = True #this tells IModel to automatically add flattens in the right places

        # CIFAR 10 (7, 7) to (1, 1)
        if bFullSize:
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            iSz = 7
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            iSz = 1

        self.classifier = nn.Sequential(
            nn.Linear(iFinalChannels * iSz * iSz, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    "ET": [64, "M", 96, "M", 128, "M", 192, "M", 256, "M"],
    "T": [64, "M", 128, "M", 192, 192, "M", 256, 256, "M", 256, 256, "M"],
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M",512, 512, 512, 512, "M"],
}


def _vgg(cfg, batch_norm: bool = True, strModel: str = None, num_classes: int = 10, **kwargs):
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), cfgs[cfg][-2], num_classes=num_classes, init_weights = True if strModel is None else False, **kwargs)
    
    if strModel is not None:
        if not os.path.exists(strModelDir + strModel):
            print("WARNING! Unable to load weights! Double check model path: {}".format(strModelDir + strModel))
            return model
        
        with open(strModelDir + strModel, "rb") as f:
            sd = torch.load(f, weights_only = False)
        
        #overwrite a few weights because the architectures changed slightly from imagenet version
        sd["classifier.0.weight"] = torch.randn((4096, cfgs[cfg][-2]))
        nn.init.normal_(sd["classifier.0.weight"], 0, 0.01)
        sd["classifier.6.weight"] = torch.randn((num_classes, 4096))
        nn.init.normal_(sd["classifier.6.weight"], 0, 0.01)
        sd["classifier.6.bias"] = torch.zeros((num_classes))
        model.load_state_dict(sd)
    
    return model

def vgg4(batch_norm: bool = True, **kwargs):
    return _vgg("ET", batch_norm=batch_norm, **kwargs)

def vgg7(batch_norm: bool = True, **kwargs):
    return _vgg("T", batch_norm=batch_norm, **kwargs)

def vgg11(batch_norm: bool = True, bImNPreTrained: bool = False, **kwargs):
    strM = "vgg11.pth" if bImNPreTrained else None
    return _vgg("A", batch_norm=batch_norm, strModel = strM, **kwargs)

def vgg13(batch_norm: bool = True, bImNPreTrained: bool = False, **kwargs):
    strM = "vgg13.pth" if bImNPreTrained else None
    return _vgg("B", batch_norm=batch_norm, strModel = strM, **kwargs)

def vgg16(batch_norm: bool = True, bImNPreTrained: bool = False, **kwargs):
    strM = "vgg16.pth" if bImNPreTrained else None
    return _vgg("D", batch_norm=batch_norm, strModel = strM, **kwargs)

def vgg19(batch_norm: bool = True, bImNPreTrained: bool = False, **kwargs):
    strM = "vgg19.pth" if bImNPreTrained else None
    return _vgg("E", batch_norm=batch_norm, strModel = strM, **kwargs)

if __name__ == "__main__":
    tModel = vgg4(bImNPreTrained = True)
    x = torch.randn(1, 3, 32, 32)
    #PrintMod