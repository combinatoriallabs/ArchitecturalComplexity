import torch

import functools
from typing import Union, Callable

#this class taken and modified from: https://github.com/Hao840/vanillaKD/blob/main/losses/kd.py
class KLDiv(torch.nn.Module):
    def __init__(self):
        super(KLDiv, self).__init__()

    def forward(self, z_s, z_t, fTemp, bNormalize: bool = False, fEpsilon: float = 1e-6):
        if bNormalize:
            tSAvg = torch.mean(z_s, dim = 1, keepdim = True)
            tSStd = torch.std(z_s, dim = 1, keepdim = True)
            tTAvg = torch.mean(z_t, dim = 1, keepdim = True)
            tTStd = torch.std(z_t, dim = 1, keepdim = True)

            z_s = (z_s - tSAvg) / (tSStd + fEpsilon)
            z_t = (z_t - tTAvg) / (tTStd + fEpsilon)

        log_pred_student = torch.nn.functional.log_softmax(z_s / fTemp, dim=1)
        pred_teacher = torch.nn.functional.softmax(z_t / fTemp, dim=1)
        kd_loss = torch.nn.functional.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
        #kd_loss *= fTemp ** 2 #this is handled explicitly elsewhere instead
        return kd_loss

class GenReLU(torch.nn.Module):
    def __init__(self, kn: float = 0.1, kp: float = 1.0):
        super(GenReLU, self).__init__()
        self.kn = kn
        self.kp = kp

    def forward(self, x: torch.tensor):
        return torch.where(x > 0, self.kp * x, self.kn * x)

class LinearHead(torch.nn.Module):
    def __init__(self, dim: int, num_classes: int = 10):
        super(LinearHead, self).__init__()
        self.dim = dim
        self.layer = torch.nn.Linear(self.dim, num_classes)
        
    def forward(self, x):
        return torch.nn.functional.leaky_relu(self.layer(x))
    
class NormalizeLayer(torch.nn.Module):
    """normalization layer
    taken from: https://github.com/DefangChen/SemCKD/blob/master/models/util.py
    """
    def __init__(self, power = 2):
        super(NormalizeLayer, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class NLayerMLP(torch.nn.Module):
    def __init__(self, iDimInput: int, vecHiddenDims: list[int], iC: int = None, bNormalizeOutput: bool = True,
                 tmActivation: torch.nn.Module = torch.nn.LeakyReLU, bFlatten: bool = True, bResidual: bool = False) -> None:
        '''
        bFlatten: adds flatten() layer to beginning
        '''
        super(NLayerMLP, self).__init__()

        if bResidual:
            assert iDimInput == vecHiddenDims[-1], "NLayerMLP Requires that the final output size match the input size for Residual mode"
            self.bGeneralizedLayer = True
        self.bRes = bResidual

        self.vecLayers = []
        if bFlatten: self.vecLayers.append(torch.nn.Flatten())

        if len(vecHiddenDims) > 0:
            self.vecLayers += [torch.nn.Linear(iDimInput, vecHiddenDims[0])]
            if tmActivation is not None: self.vecLayers += [tmActivation()]
            for i in range(1, len(vecHiddenDims)):
                self.vecLayers.append(torch.nn.Linear(vecHiddenDims[i - 1], vecHiddenDims[i]))
                if tmActivation is not None: self.vecLayers.append(tmActivation())
        
            if bNormalizeOutput: self.vecLayers.append(NormalizeLayer(2))
            if iC is not None: self.vecLayers.append(torch.nn.Linear(vecHiddenDims[-1], iC))
        else:
            self.vecLayers.append(torch.nn.Linear(iDimInput, iC))
        
        self.mList = torch.nn.ModuleList(self.vecLayers) #making torch happy
        
        return
        
    def forward(self, X: torch.tensor) -> torch.tensor:
        if self.bRes: I = X
        for tL in self.vecLayers: X = tL(X)
        if self.bRes: X += I
        return X

class SingleLayerConv2D(torch.nn.Module):
    def __init__(self, iChanIn: int, iChanOut: int, bLearned: bool, iKernelSize: int = 1) -> None:
        super(SingleLayerConv2D, self).__init__()
        self.conv = torch.nn.Conv2d(iChanIn, iChanOut, kernel_size=iKernelSize, padding=iKernelSize - 1, stride=1, bias=False)
        self.bL = bLearned

    def forward(self, x):
        if self.bL:
            return self.conv(x)
        with torch.no_grad():
            return self.conv(x)
    
class SingleLayerConv1D(torch.nn.Module):
    def __init__(self, iChanIn: int, iChanOut: int, bLearned: bool, iKernelSize: int = 1) -> None:
        super(SingleLayerConv1D, self).__init__()
        self.conv = torch.nn.Conv1d(iChanIn, iChanOut, kernel_size=iKernelSize, padding=iKernelSize - 1, stride=1, bias=False)
        self.bL = bLearned
        #torch.nn.init.kaiming_normal_(self.conv.weight, mode="fan_out")

    def forward(self, x):
        if self.bL:
            return self.conv(x)
        with torch.no_grad():
            return self.conv(x)

class ThreeLayerConv2D(torch.nn.Module):
    """non-linear embed by MLP
    taken and modified from: https://github.com/DefangChen/SemCKD/blob/master/models/util.py
    """
    def __init__(self, num_input_channels: int, num_target_channels: int, bLearned: bool) -> None:
        super(ThreeLayerConv2D, self).__init__()
        self.num_mid_channel = 2 * num_target_channels
        
        def conv1x1(in_channels, out_channels, stride=1):
            return torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)
        def conv3x3(in_channels, out_channels, stride=1):
            return torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        
        self.conv1 = conv1x1(num_input_channels, self.num_mid_channel)
        self.norm1 = torch.nn.BatchNorm2d(self.num_mid_channel)
        self.conv2 = conv3x3(self.num_mid_channel, self.num_mid_channel)
        self.norm2 = torch.nn.BatchNorm2d(self.num_mid_channel)
        self.conv3 = conv1x1(self.num_mid_channel, num_target_channels)

        self.bL = bLearned

        return

    def forward(self, x):
        if self.bL:
            x = self.conv1(x)
            x = self.norm1(x)
            x = torch.nn.functional.relu(x)
            x = self.conv2(x)
            x = self.norm2(x)
            x = torch.nn.functional.relu(x)
            x = self.conv3(x)
        else: 
            with torch.no_grad():
                x = self.conv1(x)
                x = self.norm1(x)
                x = torch.nn.functional.relu(x)
                x = self.conv2(x)
                x = self.norm2(x)
                x = torch.nn.functional.relu(x)
                x = self.conv3(x)
        return x
    
class ThreeLayerConv1D(torch.nn.Module):
    """non-linear embed by MLP
    taken and modified from: https://github.com/DefangChen/SemCKD/blob/master/models/util.py
    """
    def __init__(self, num_input_channels: int, num_target_channels: int, bLearned: bool) -> None:
        super(ThreeLayerConv1D, self).__init__()
        self.num_mid_channel = 2 * num_target_channels
        
        def conv1x1(in_channels, out_channels, stride=1):
            return torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)
        def conv3x3(in_channels, out_channels, stride=1):
            return torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        
        self.conv1 = conv1x1(num_input_channels, self.num_mid_channel)
        self.norm1 = torch.nn.BatchNorm1d(self.num_mid_channel)
        self.conv2 = conv3x3(self.num_mid_channel, self.num_mid_channel)
        self.norm2 = torch.nn.BatchNorm1d(self.num_mid_channel)
        self.conv3 = conv1x1(self.num_mid_channel, num_target_channels)

        self.bL = bLearned

        return

    def forward(self, x):
        if self.bL:
            x = self.conv1(x)
            x = self.norm1(x)
            x = torch.nn.functional.relu(x)
            x = self.conv2(x)
            x = self.norm2(x)
            x = torch.nn.functional.relu(x)
            x = self.conv3(x)
        else: 
            with torch.no_grad():
                x = self.conv1(x)
                x = self.norm1(x)
                x = torch.nn.functional.relu(x)
                x = self.conv2(x)
                x = self.norm2(x)
                x = torch.nn.functional.relu(x)
                x = self.conv3(x)
        return x
    

class UnaryOperation(torch.nn.Module):
    def __init__(self, tOp: Union[torch.nn.Module, Callable], iDim: Union[int, list[int]] = None):
        '''
        Utility class for converting function calls into Modules. Useful for hooking things into the IModel interface
        '''
        super().__init__()
        self.tOp = tOp
        self.iDim = iDim

        return

    def __str__(self) -> str:
        if isinstance(self.tOp, torch.nn.Module): return self.tOp.__class__.__name__
        elif isinstance(self.tOp, functools.partial): return self.tOp.func.__name__
        elif callable(self.tOp): return self.tOp.__name__

    def forward(self, X: torch.tensor) -> torch.tensor:
        if callable(self.tOp):
            return self.tOp(X) if self.iDim is None else self.tOp(X, dim = self.iDim)
        return self.tOp(X)

    
if __name__ == "__main__":
    tModel = NLayerMLP(3072, [256, 128])
    print(tModel(torch.randn(1, 3072)).shape)