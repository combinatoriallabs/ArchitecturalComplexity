'''
Code written by Nicholas J. Cooper.
Released under the MIT license, see the GitHub page for the full legal boilerplate.
tldr: you freely can do whatever you like with this code so long as this message is retained and you cite the GitHub: https://github.com/combinatoriallabs/ArchitecturalComplexity
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
'''



import torch

from OperationLayers import *

from typing import Callable

class TOMLP(torch.nn.Module):
    def __init__(self, vecOps: list[TOLayer], iCls: int, tmActivation: torch.nn.Module = None, iBatchAwareModes: int = 0,
                 tcFunc: Callable[[torch.tensor], torch.tensor] = None, bResidual: bool = False) -> None:
        super().__init__()
        
        self.vecLayers = torch.nn.ModuleList(vecOps)
        self.tAct = tmActivation() if tmActivation is not None else None
        self.tLin = torch.nn.Linear(torch.prod(torch.tensor(vecOps[-1].OutputShape())[iBatchAwareModes:]), iCls)
        self.tcFunc = tcFunc
        self.bRes = bResidual

    def zero_grad(self, set_to_none = True):
        for toL in self.vecLayers: toL.zero_grad(set_to_none = set_to_none)
        self.tLin.zero_grad()
        return

    def forward(self, X: torch.tensor) -> torch.tensor:
        for idtaL in self.vecLayers:
            if self.tcFunc is not None: T = self.tcFunc(X)
            else: T = X
            
            T = idtaL(T)
            
            if self.tAct is not None: T = self.tAct(T)

            if self.bRes: X = T + X
            else: X = T

        return self.tLin(X.reshape(X.shape[0], -1))
    

def BuildTOMLP(idtaLayer: TOLayer, iLayers: int, iCls: int, tmActivation: torch.nn.Module = None, iBatchAwareModes: int = 0, tcFunc = None,
                bResidual: bool = False) -> TOMLP:
    vecL = []
    for _ in range(iLayers):
        vecL.append(copy.deepcopy(idtaLayer))
        vecL[-1].SetupParameterTensors()

    return TOMLP(vecL, iCls = iCls, tmActivation = tmActivation, iBatchAwareModes = iBatchAwareModes, tcFunc = tcFunc, bResidual = bResidual)


class TOBasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(
        self,
        sIn: list[int],
        sOut: list[int],
        tomOp: TOMatrix,
        stride=1,
        downsample=None,
        norm_layer=None,
        relu = torch.nn.ReLU(inplace=True),
    ) -> None:
        super(TOBasicBlock, self).__init__()

        self.iK = 3
        self.iP = 1
        self.iS = stride

        sConv = list(Conv2dUnfold(torch.rand([1] + sIn), iK = self.iK, iS = self.iS, iP = self.iP).shape)[1:]
        #print(f"First Layer: sIn {sConv}, sOut {sOut}")
        self.toL1 = BuildTOLayer(copy.deepcopy(tomOp), sConv, sOut)

        self.bn1 = norm_layer(sOut[0]) if norm_layer else None
        self.relu = relu

        sConv2 = list(Conv2dUnfold(torch.rand([1] + sOut), iK = self.iK, iS = 1, iP = self.iP).shape)[1:]
        #print(f"Second Layer: sIn {sConv2}, sOut {sOut}")
        self.toL2 = BuildTOLayer(copy.deepcopy(tomOp), sConv2, sOut)

        self.bn2 = norm_layer(sOut[0]) if norm_layer else None
        self.downsample = downsample
        self.stride = stride

        self.bGeneralizedLayer = True

        return

    def forward(self, x):
        identity = x

        out = Conv2dUnfold(x, iK = self.iK, iS = self.iS, iP = self.iP)
        out = self.toL1(out)

        if self.bn1: out = self.bn1(out)
        out = self.relu(out)

        out = Conv2dUnfold(out, iK = self.iK, iS = 1, iP = self.iP)
        out = self.toL2(out)

        if self.bn2: out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def MakeTOResidualSection(block, iSz, tomOp, inchannels, channels, blocks, stride=1, batch_norm: bool = True, relu: torch.nn.Module = torch.nn.ReLU(inplace=True)):
    from Arch.Models.Resnet import conv1x1
    
    norm_layer = torch.nn.BatchNorm2d if batch_norm else None
    downsample = None
    if stride != 1 or inchannels != channels * block.expansion:
        if norm_layer: 
            downsample = torch.nn.Sequential(
                conv1x1(inchannels, channels * block.expansion, stride),
                norm_layer(channels * block.expansion),
            )
        else:
            downsample = torch.nn.Sequential(
                conv1x1(inchannels, channels * block.expansion, stride),
            )

    iSzOut = iSz // stride

    layers = []
    layers.append(
        block(
            [inchannels, iSz, iSz],
            [channels, iSzOut, iSzOut],
            tomOp,
            stride,
            downsample,
            norm_layer,
            relu,
        )
    )
    inchannels = channels * block.expansion
    for _ in range(1, blocks):
        layers.append(
            block(
                [inchannels, iSzOut, iSzOut],
                [channels, iSzOut, iSzOut],
                tomOp,
                norm_layer=norm_layer,
            )
        )

    return torch.nn.Sequential(*layers)


class TOResNet(torch.nn.Module):
    def __init__(
        self,
        block,
        iSz: int,
        tomOp: TOMatrix,
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
        super(TOResNet, self).__init__()
        
        if len(layers) not in  [3, 4] or len(channels) not in [3, 4] or len(layers) != len(channels):
            print("Error! TOResNet() called with incorrect amounts of channels and layers!")
        
        self._norm_layer = torch.nn.BatchNorm2d if batch_norm else None

        self.inplanes = 64
        self.groups = groups
        self.base_width = 64

        self.bCheckLinear = True #this tells IModel to automatically add flattens in the right places

        if bImNet:
            self.conv1 = torch.nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            iSz //= 2
        else: self.conv1 = torch.nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = self._norm_layer(self.inplanes) if batch_norm else None

        self.relu = torch.nn.ReLU(inplace=True)
        
        self.maxpool = None
        if bInitialMaxPool:
            self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            iSz //= 2

        self.layer1 = MakeTOResidualSection(block, iSz, tomOp, self.inplanes, channels[0], layers[0], 1, batch_norm, self.relu)
        self.inplanes = channels[0] * block.expansion

        self.layer2 = MakeTOResidualSection(block, iSz, tomOp, self.inplanes, channels[1], layers[1], 2, batch_norm, self.relu)
        self.inplanes = channels[1] * block.expansion
        iSz //= 2

        self.layer3 = MakeTOResidualSection(block, iSz, tomOp, self.inplanes, channels[2], layers[2], 2, batch_norm, self.relu)
        self.inplanes = channels[2] * block.expansion
        iSz //= 2

        if len(layers) == 4:
            self.layer4 = MakeTOResidualSection(block, iSz, tomOp, self.inplanes, channels[3], layers[3], 2, batch_norm, self.relu)
            self.inplanes = channels[3] * block.expansion
        else:
            self.layer4 = None
        
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.fc = torch.nn.Linear(channels[-1] * block.expansion, num_classes)

        if bInitWeights:
            for m in self.modules():
                if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                    torch.nn.init.constant_(m.weight, 1)
                    torch.nn.init.constant_(m.bias, 0)

            # Zero-initialize the last BN in each residual branch,
            # so that the residual branch starts with zeros, and each residual block behaves like an identity.
            # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
            if batch_norm:
                for m in self.modules():
                    if isinstance(m, TOBasicBlock):
                        torch.nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x = self.conv1(x)
        if self.bn1: x = self.bn1(x)
        x = self.relu(x)
        if self.maxpool is not None: x = self.maxpool(x)
        x = self.layer1(x)
        #print(x.shape)
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
        

def _TOresnet(block, iSz, tomOp, layers, channels, batch_norm: bool = True, num_classes: int = 10, bImNet: bool = False, **kwargs):
    model = TOResNet(block, iSz, tomOp, layers, channels, batch_norm=batch_norm, num_classes=num_classes, bInitWeights = True, 
                   bImNet = bImNet, **kwargs)
    
    return model


def TOresnet9(iSz, tomOp, batch_norm: bool = True, bImNet: bool = False, **kwargs):
    return _TOresnet(TOBasicBlock, iSz, tomOp, [1, 1, 1, 1], [64, 128, 256, 512], batch_norm=batch_norm, bImNet = bImNet, **kwargs)

def TOresnet18(iSz, tomOp, batch_norm: bool = True, bImNPreTrained: bool = False, **kwargs):
    strM = "resnet18.pth" if bImNPreTrained else None
    return _TOresnet(TOBasicBlock, iSz, tomOp, [2, 2, 2, 2], [64, 128, 256, 512], batch_norm=batch_norm, strModel = strM, **kwargs)

def TOresnet18Inv(iSz, tomOp, batch_norm: bool = True, bImNPreTrained: bool = False, **kwargs):
    strM = "resnet18.pth" if bImNPreTrained else None
    return _TOresnet(TOBasicBlock, iSz, tomOp, [4, 3, 2, 1], [64, 128, 256, 512], batch_norm=batch_norm, strModel = strM, **kwargs)

def TOresnet34(iSz, tomOp, batch_norm: bool = True, bImNPreTrained: bool = False, bImNet: bool = False, **kwargs):
    strM = "resnet34.pth" if bImNPreTrained else None
    return _TOresnet(TOBasicBlock, iSz, tomOp, [3, 4, 6, 3], [64, 128, 256, 512], batch_norm=batch_norm, strModel = strM, bImNet = bImNet, **kwargs)



if __name__ == "__main__":
    iSz = 32

    tStruct = torch.tensor([
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 1],

        [0, 0, 1, 1, 1, 0]
    ])

    tomOp = TOMatrix(tStruct)

    tModel = TOresnet9(iSz, tomOp, bInitialMaxPool = False).to(device)
