import torch

'''
Taken and modified from: https://github.com/kzkadc/poly-nets/blob/master/train.py
Implements PolyNet-V2 from figure 1 of: https://openaccess.thecvf.com/content_CVPR_2020/papers/Chrysos_P-nets_Deep_Polynomial_Neural_Networks_CVPR_2020_paper.pdf
with borrowed channels/stages from ResNet
'''

class PolyNetV2Block(torch.nn.Module):
    def __init__(self, iChIn: int, iChOut: int):
        super().__init__()

        self.tC1 = torch.nn.Conv2d(iChIn, iChOut, kernel_size = 3, stride = 1, padding = 1)
        self.tC2 = torch.nn.Conv2d(iChIn, iChOut, kernel_size = 3, stride = 1, padding = 1)

        self.bGeneralizedLayer = True

    def forward(self, X: torch.tensor) -> torch.tensor:
        return self.tC1(X) * self.tC2(X)
    
class PolyNetV2DownsampleBlock(torch.nn.Module):
    def __init__(self, iChIn: int, iChOut: int):
        super().__init__()

        self.tC1 = torch.nn.Conv2d(iChIn, iChOut, kernel_size = 4, stride = 2, padding = 1)
        self.tC2 = torch.nn.Conv2d(iChIn, iChOut, kernel_size = 4, stride = 2, padding = 1)

        self.bGeneralizedLayer = True

    def forward(self, X: torch.tensor) -> torch.tensor:
        return self.tC1(X) * self.tC2(X)

class PolyNetV2(torch.nn.Module):
    def __init__(self, vecStages: list[list[int]], iSz: int, iChIn = 3, iCls = 10) -> None:
        super().__init__()

        self.vecLayers = []
        for vecStage in vecStages:
            for iC in vecStage:
                self.vecLayers.append(PolyNetV2Block(iChIn, iC))
                iChIn = iC
            self.vecLayers.append(PolyNetV2DownsampleBlock(iChIn, iChIn*2))
            iChIn *= 2
            iSz //= 2

        self.vecLayers.append(torch.nn.Flatten())
        self.vecLayers.append(torch.nn.Linear(iSz**2 * iChIn, iCls))

        self.vecLayers = torch.nn.ModuleList(self.vecLayers)

        return

    def forward(self, x) -> torch.tensor:
        for tL in self.vecLayers: x = tL(x)
        return x
    

def PolyNetV2_T(iChIn: int, iSz: int, iCls: int) -> torch.nn.Module:
    vecStages = [
        [16],
        [],
        [128],
    ]
    return PolyNetV2(vecStages, iSz, iChIn, iCls)

def PolyNetV2_S(iChIn: int, iSz: int, iCls: int) -> torch.nn.Module:
    vecStages = [
        [16, 32],
        [64],
        [128],
    ]
    return PolyNetV2(vecStages, iSz, iChIn, iCls)


if __name__ == "__main__":
    X = torch.randn((128, 3, 64, 64)).to("cuda")
    vecStages = [
        [16, 32],
        [32, 64],
        [64, 128],
        [128, 256]
    ]
    tModel = PolyNetV2(vecStages = vecStages, iSz = 64).to("cuda")

    Y = tModel(X)
    print(Y.shape)
