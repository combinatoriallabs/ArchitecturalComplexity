'''
Code written by Nicholas J. Cooper.
Released under the MIT license, see the GitHub page for the full legal boilerplate.
tldr: you freely can do whatever you like with this code so long as this message is retained and you cite the GitHub: https://github.com/combinatoriallabs/ArchitecturalComplexity
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
'''



import torch

from Arch.Models.ViT import *
from Arch.Models.IModel import *

from OperationLayers import Conv2dUnfold, UnaryOperation


class ViTInput1D(torch.nn.Module):
    def __init__(self, iImSize: int, iPatchSize: int, iChannels: int, iModelDim: int):
        super().__init__()
        
        self.tInputLayer = ViTInputLayer(iImSize, iPatchSize, iChannels, iModelDim)
        self.tEmbedLayer = PositionEncodingLayer(iImSize, iPatchSize, iModelDim, iDim = 1)

        return
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #Conv
        x = self.tInputLayer(x)
        #+pos embeddings
        x = self.tEmbedLayer(x)

        return x
    

class PatchUnfold(torch.nn.Module):
    def __init__(self, iPatchSize: int, iChannels: int = -1):
        super().__init__()

        self.tUnfold = UnaryOperation(partial(Conv2dUnfold, iK = iPatchSize, iS = iPatchSize, iP = 0))

        if iChannels > 1:
            self.tConv = torch.nn.Conv2d(3, iChannels, kernel_size = 3, stride = 1, padding = 1)
            self.tOp = torch.nn.Sequential(self.tConv, self.tUnfold)
        else: self.tOp = self.tUnfold

        self.bGeneralizedLayer = True

        return
    
    def forward(self, X: torch.tensor) -> torch.tensor:
        return self.tOp(X)
    
class OverlapPatchUnfold(torch.nn.Module):
    def __init__(self, iPatchSize: int, iChannels: int = -1):
        super().__init__()

        self.tUnfold = UnaryOperation(partial(Conv2dUnfold, iK = iPatchSize, iS = iPatchSize // 2, iP = iPatchSize // 2))

        if iChannels > 1:
            self.tConv = torch.nn.Conv2d(3, iChannels, kernel_size = 3, stride = 1, padding = 1)
            self.tOp = torch.nn.Sequential(self.tConv, self.tUnfold)
        else: self.tOp = self.tUnfold

        return
    
    def forward(self, X: torch.tensor) -> torch.tensor:
        return self.tOp(X)
    

class Conv2Downsample(torch.nn.Module):
    def __init__(self, vecChannels: list[int]):
        super().__init__()

        self.vecLayers = []
        for i in range(len(vecChannels) - 1):
            self.vecLayers.append(torch.nn.Conv2d(vecChannels[i], vecChannels[i+1], kernel_size = 3, stride = 1, padding = 1))
            self.vecLayers.append(torch.nn.AvgPool2d(2, 2))

        self.vecLayers = torch.nn.ModuleList(self.vecLayers)

    def forward(self, X: torch.tensor) -> torch.tensor:
        for tm in self.vecLayers: X = tm(X)
        return X