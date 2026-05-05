'''
Code written by Nicholas J. Cooper.
Released under the MIT license, see the GitHub page for the full legal boilerplate.
tldr: you freely can do whatever you like with this code so long as this message is retained and you cite the GitHub: https://github.com/combinatoriallabs/ArchitecturalComplexity
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
'''


import torch

from typing import List, Optional, Iterable
import copy

try:
    from Arch.Models.ModelUtils import *
except:
    from ModelUtils import *

class ISeq(torch.nn.Module):
    '''
    Wrapper for module lists
    '''
    def __init__(self, tModuleList: Iterable[torch.nn.Module]) -> None:
        super(ISeq, self).__init__()
        self.vecLayers = copy.deepcopy(tModuleList)

    def forward(self, x: torch.tensor) -> torch.tensor:
        for tM in self.vecLayers:
            x = tM(x)
        return x

class IModel(torch.nn.Module):
    '''
    Generic interface class for linearizable models. Provides easy per-layer feature access and metric tracking tools
    '''
    def __init__(self, tModel: torch.nn.Module) -> None:
        super(IModel, self).__init__()
        self.tModel: torch.nn.Module = tModel

        self.bCheckLinear = self.tModel.bCheckLinear if hasattr(self.tModel, "bCheckLinear") else False
        self.bSkipFirstPool = False

        self.bHasMetaModules = False
        self.vecMetaModules = []

        self.bGeneralizedLayer = True
        
        if IsGeneralizedLayer(self.tModel): self.vecLayers = [self.tModel]
        else: self.vecLayers: List[torch.nn.Module] = LinearizeModel(self.tModel.children(), bCheckLinear = self.bCheckLinear)

        for tL in self.vecLayers:
            if hasattr(tL, "bMetaModule"):
                self.bHasMetaModules = True
                self.vecMetaModules.append(tL)
        #print(self.vecLayers)
        self.iGenLayers: int = CountGenLayers(self.vecLayers)
        self.iHeadIdxLiteral, self.iHeadIdx = FindHead(self.vecLayers)
        self.mapGenIdxToLiteralIdx = GenLayerIdxMap(self.vecLayers)

        self.dParamGroups = torch.nn.ModuleDict({
            "backbone": torch.nn.ModuleList([tL for tL in self.vecLayers[:self.iHeadIdxLiteral]]),
            "classification_head": torch.nn.ModuleList([tL for tL in self.vecLayers[self.iHeadIdxLiteral:]]),
        })
        
        #if "Linear" not in self.vecLayers[-1].__class__.__name__:
        #    print("Warning! Final linearizable layer of model passed to IModel is not Linear!")

        return
    
    def HasMetaModules(self) -> bool:
        return self.bHasMetaModules
    
    def UpdateLoss(self, fLoss: float) -> None:
        for tL in self.vecMetaModules: tL.UpdateLoss(fLoss)
        return

    def UpdateStep(self,) -> None:
        for tL in self.vecMetaModules: tL.UpdateStep()
        return

    def DisableFirstPool(self) -> None:
        self.bSkipFirstPool = True
        return
    
    def SetEndLayer(self, iLidx: int, bUseLiteralIndices: bool = False) -> None:
        '''
        Truncates the model to stop at the iLidx-th layer.
        WARNING: This overwrites self.tModel. Do not use if you need to retain access to the complete model.
        '''
        if not bUseLiteralIndices: iLidx = self.mapGenIdxToLiteralIdx[iLidx]

        iR = len(self.vecLayers) - iLidx
        for _ in range(iR):
            tL = self.vecLayers.pop()
            del tL

        self.tModel = torch.nn.Sequential(*self.vecLayers) #TODO: warning!

        self.dParamGroups = torch.nn.ModuleDict({
            "backbone": torch.nn.ModuleList([tL for tL in self.vecLayers]),
        })

        return

    def GetClassifierParams(self) -> list[torch.tensor, Optional[torch.tensor]]:
        tLayer = self.vecLayers[-1]
        vecRet = [copy.deepcopy(tLayer.weight.data)]
        if hasattr(tLayer, "bias"):
            vecRet.append(copy.deepcopy(tLayer.bias.data))

        return vecRet
        
    def forward(self, x, vecFeatureLayers: List[int] = [], iStartIdxGen: int = -1, 
                bUseLiteralIndices: bool = False, bEarlyQuit: bool = False, bStopAtHead: bool = False,
                bNaNCheck: bool = False, vecSkipLayers: list[int] = []) -> tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        '''
        Wrapper around vecLayers which allows mulitple features to be returned. Also prevents unnecessary compute.        
        '''
        
        if iStartIdxGen >= 0:
            iStartIdxLiteral = self.mapGenIdxToLiteralIdx[iStartIdxGen] + 1
        else:
            iStartIdxLiteral = 0

        if len(vecFeatureLayers) == 0:
            for i in range(iStartIdxLiteral, len(self.vecLayers)):
                if i in vecSkipLayers: continue
                x = self.vecLayers[i](x)
                #print(x.shape)
            return x
        
        for i in range(len(vecFeatureLayers)):
            if vecFeatureLayers[i] < 0:
                if bUseLiteralIndices:
                    vecFeatureLayers[i] += len(self.vecLayers)
                else:
                    vecFeatureLayers[i] += self.iGenLayers

        vecRet = []
        iL = 0
        bFirstPool = True

        for i in range(iStartIdxLiteral, len(self.vecLayers)):
            if bStopAtHead and i >= self.iHeadIdxLiteral: break

            #grab the next layer
            tLayer = self.vecLayers[i]
            
            if self.bSkipFirstPool and "pool" in tLayer.__class__.__name__.lower() and bFirstPool:
                bFirstPool = False
                continue
            
            if i in vecSkipLayers: continue

            x = tLayer(x)
            #print(x.shape)

            if bNaNCheck:
                tCheck = torch.sum(x)
                if tCheck != tCheck:
                    print("NaNs found after layer", i)
                    print(tLayer.__class__.__name__)
                    print(x)
                    input()
            
            #add these features to the return list if required
            if (bUseLiteralIndices and i in vecFeatureLayers):
                vecRet.append(x)
            
            if IsGeneralizedLayer(tLayer):
                if (not bUseLiteralIndices and iL in vecFeatureLayers):
                    vecRet.append(x)
                iL += 1 #increment gen. layer counter
                
            if bEarlyQuit:
                if (bUseLiteralIndices and i > max(vecFeatureLayers)) or (not bUseLiteralIndices and iL > max(vecFeatureLayers)):
                    return x, vecRet
        
        return x, vecRet
    
    def features(self, x, bNaNCheck: bool = False):
        if bNaNCheck:
            for tLayer in self.vecLayers[:self.iHeadIdxLiteral]:
                tCheck = torch.sum(x)
                if tCheck != tCheck:
                    print("NaNs found!")
                    print(tLayer.__class__.__name__)
                    print(x)
                    input()

                x = tLayer(x)
                
        else:
            for tLayer in self.vecLayers[:self.iHeadIdxLiteral]:
                x = tLayer(x)


        return x
    
    def classify(self, x, bNaNCheck: bool = False):
        if bNaNCheck:
            for tLayer in self.vecLayers[self.iHeadIdxLiteral:]:
                
                tCheck = torch.sum(x)
                if tCheck != tCheck:
                    print("NaNs found!")
                    print(tLayer.__class__.__name__)
                    print(x)
                    input()
                
                x = tLayer(x)

        else:
            for tLayer in self.vecLayers[self.iHeadIdxLiteral:]:            
                x = tLayer(x)

        return x
    
    def GetHeadGrad(self):
        '''
        generic impl of feature extractor grad computation
        '''
        with torch.no_grad():
            return torch.matmul(self.vecLayers[self.iHeadIdxLiteral].bias.grad.unsqueeze(0), self.vecLayers[self.iHeadIdxLiteral].weight)
            

if __name__ == "__main__":
    #from Resnet import *
    # from VGG import *
    # from ViT import *
    # from MobileNetV2 import *
    # from ShuffleNetV2 import *
    #from Misc import NLayerMLP
    #tModel = VisionTransformer(32, 4, 3, 8, 6, 192, 576, num_classes = 10, bAvgTokens = True, representation_size = None)
    #tModel = resnet34(bImNPreTrained = False)
    #tModel = vgg19(bFullSize = False)
    #tModel = mobilenet_v2(iCls = 200, iSz = 32)
    #tModel = ShuffleNetV2(n_class = 10, input_size = 224)

    #tModel = NLayerMLP(768, [768, 384, 192], 10)

    tModel = torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.Linear(10, 5),
        torch.nn.Linear(5, 2),
    )

    #for vecL in tModel.children(): print(vecL.__class__.__name__)

    tModel = IModel(tModel)

    #print(tModel.iHeadIdx, tModel.iGenLayers, CountParams(tModel))

    tModel = tModel.to("cuda")

    #x = torch.randn((1, 3, 32, 32)).to("cuda")

    PrintModelSummary(tModel)
    print(CountParams(tModel))

    tModel.SetEndLayer(1, bUseLiteralIndices = True)
    #del tModel[2]

    PrintModelSummary(tModel)
    print(CountParams(tModel))

    #print(iModel.GetClassifierParams())

    #print([l.__class__.__name__ for l in iModel.dParamGroups["transition_layers"]])

    #iSeq = ISeq(iModel.dParamGroups["classification_head"])
    #iSeq(x)

    #print(iModel.mapGenIdxToLiteralIdx)

    #iModel.classify(iModel.features(x))

    #f, vecF = iModel(x, vecFeatureLayers = [-1], bEarlyQuit = True)

    #print(vecF[0].shape)

    # y = iModel(f, iStartIdxGen = 7)

    # print(y.shape)

    # _, f1 = iModel(x, vecFeatureLayers = [15])
    # iModel.DisableFirstPool()
    # x = torch.randn((1, 3, 32, 32)).to("cuda")
    # _, f2 = iModel(x, vecFeatureLayers = [15])

    # print(f1[0].shape, f2[0].shape)
    
    # print("---------------------")

    # x, f = iModel(x, vecFeatureLayers = [-1], bUseLiteralIndices = False)

    # print(f[0].shape, torch.min(f[0]))
