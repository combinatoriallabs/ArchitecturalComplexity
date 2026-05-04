'''
Code written by Nicholas J. Cooper.
Released under the MIT license, see the GitHub page for the full legal boilerplate.
tldr: you freely can do whatever you like with this code so long as this message is retained and you cite the GitHub: https://github.com/Thegolfingocto/ArchitecturalComplexity
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
'''



import torch
import torchvision

from typing import List

def IsGeneralizedLayer(tLayer: torch.nn.Module) -> bool:
    '''
    Simple helper to control which layers count as layers for cross-architecture experiments.
    '''
    return (hasattr(tLayer, "bGeneralizedLayer") and tLayer.bGeneralizedLayer) or (isinstance(tLayer, torch.nn.modules.activation.ReLU) or isinstance(tLayer, torch.nn.modules.activation.LeakyReLU) or 
            isinstance(tLayer, torch.nn.modules.activation.ReLU6) or tLayer.__class__.__name__ == "BasicBlock" or 
            tLayer.__class__.__name__ == "Bottleneck" or tLayer.__class__.__name__ == "InvertedBlock" or
            tLayer.__class__.__name__ == "InvertedResidual" or isinstance(tLayer, torchvision.models.vision_transformer.EncoderBlock) or 
            tLayer.__class__.__name__ == "EncoderBlock" or tLayer.__class__.__name__ == "Flatten" or 
            tLayer.__class__.__name__ == "BasicFishBlock")
    # or tLayer.__class__.__name__ == "FishConv2d"
def CountGenLayers(vecLayers: List[torch.nn.Module]) -> int:
    cnt = 0
    for tL in vecLayers:
        if IsGeneralizedLayer(tL):
            cnt += 1
            
    return cnt

def GenLayerIdxMap(vecLayers: List[torch.nn.Module]) -> List[int]:
    mapIdx = []
    for i in range(len(vecLayers)):
        tL = vecLayers[i]
        if IsGeneralizedLayer(tL):
            mapIdx.append(i)

    return mapIdx

def FindHead(vecLayers: list[torch.nn.Module]) -> tuple[int, int]:
    idx = 0
    idx2 = 0
    for tL in vecLayers:
        if "Linear" in tL.__class__.__name__:
            return idx, idx2
        idx += 1
        if IsGeneralizedLayer(tL): idx2 += 1
        
    return idx, idx2

def LinearizeModel(vecLayers: List[torch.nn.Module], bCheckLinear: bool = False, bFirstLinear: bool = False) -> List[torch.nn.Module]:
    '''
    Helper function which returns a linearized list of all layers. Used in the IModel generic feature-friendly model class. 
    '''
    vecRet = []
    for layer in vecLayers:
        #print(layer.__class__.__name__, len(list(layer.children())))
        if layer.__class__.__name__ in ["Sequential", "ModuleList"]:
            for l in LinearizeModel(layer, bCheckLinear, bFirstLinear): vecRet.append(l)
        elif "torch" not in str(layer.__class__) and len(list(layer.children())) > 0 and not IsGeneralizedLayer(layer):
            for l in LinearizeModel(layer.children(), bCheckLinear, bFirstLinear): vecRet.append(l)
        else:
            if bCheckLinear and "Linear" in layer.__class__.__name__ and not bFirstLinear:
                vecRet.append(torch.nn.Flatten(1, -1))
                vecRet.append(layer)
                bFirstLinear = True
            else:
                vecRet.append(layer)
            #vecRet.append(layer)
    return vecRet

def PrintModelSummary(tModel: any, x = None):
    if hasattr(tModel, "vecLayers"):
        vecLayers = tModel.vecLayers
    else:
        vecLayers = LinearizeModel(tModel.children())
    cnt = 0
    cnt2 = 0
    if x is not None: print(x.shape)
    for layer in vecLayers:
        if x is not None:
            with torch.no_grad():
                x = layer(x)
        if "Linear" in layer.__class__.__name__ and hasattr(layer, "weight"):
            print(cnt, cnt2, type(layer), layer.weight.shape)
        else:
            print(cnt, cnt2, type(layer))
        if x is not None:
            print(cnt, cnt2, x.shape, x.view(x.shape[0], -1).shape[1])
        cnt += 1
        if IsGeneralizedLayer(layer):
            print("***Gen. Layer***")
            cnt2 += 1
    return

def CountParams(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)