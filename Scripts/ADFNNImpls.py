'''
Code written by Nicholas J. Cooper.
Released under the MIT license, see the GitHub page for the full legal boilerplate.
tldr: you freely can do whatever you like with this code so long as this message is retained and you cite the GitHub: https://github.com/combinatoriallabs/ArchitecturalComplexity
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
'''



import torch

from functools import partial

from ADFNN import *
from InputLayers import *


def BuildMLP(iInputSize: int, iHiddenSize: int, iLayers: int, iCls: int) -> GenADFNN:
    tStructMLPMM = torch.tensor([
        [1, 0],
        [1, 1],

        [1, 0]
        ])
    
    tTEM = torch.zeros((iLayers, 2*iLayers + 1))
    for i in range(iLayers): tTEM[i, 2*i:2*i + 3] = 1

    vecTOMs = [TOMatrix(tStructMLPMM) for _ in range(iLayers)]

    vecSOut = [[iHiddenSize] for _ in range(iLayers - 1)] + [[iCls]]

    vecTOMData = [TOMOpData(vecUnaryOps = [UnaryOperation(torch.nn.functional.leaky_relu)] if i < iLayers - 1 else [], vecBiasShape = vecSOut[i]) for i in range(iLayers)]

    adfnnModel = ADFNN(temArch = TEMatrix(tTEM), vecTOMs = vecTOMs, vecRowData = vecTOMData,
                       sIn = [iInputSize], vecSOut = vecSOut)
    
    tModel = GenADFNN(adfnnModel, iNumStacks = 1, vecInitModules = [torch.nn.Flatten()], bResidual = False)
    return tModel


def BuildViT(iNumTokens: int, iHeads: int, iModelDim: int, iMLPDim: int,
             iLayers: int, iImSize: int, iChannels: int, iPatchSize: int) -> GenADFNN:

    tStructMLPMM = torch.tensor([
        [1, 1, 0],
        [0, 1, 1],

        [0, 1, 0]
        ])
    
    tStructQKVMM = torch.tensor([
        [1, 1, 0, 0],
        [0, 1, 1, 1],

        [0, 1, 0, 0]
        ])
    
    tStructAMM = torch.tensor([
        [1, 1, 0, 1],
        [0, 1, 1, 1],

        [0, 1, 0, 0]
        ])
    
    tStructSC = torch.tensor([
        [1, 1],
        [1, 1],

        [0, 0]
    ])

    tTEM = torch.tensor([
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

        [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],

        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],

        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1]
    ])

    vecTOMs = [
        TOMatrix(tStructQKVMM),
        TOMatrix(tStructQKVMM),
        TOMatrix(tStructQKVMM),

        TOMatrix(tStructAMM),
        TOMatrix(tStructAMM, tShape = torch.tensor([32, 65, 65, 6])),
        TOMatrix(tStructMLPMM),

        TOMatrix(tStructSC),

        TOMatrix(tStructMLPMM),
        TOMatrix(tStructMLPMM),
        TOMatrix(tStructSC)
    ]

    vecSOut = [
        [iNumTokens, iModelDim // iHeads, iHeads],
        [iNumTokens, iModelDim // iHeads, iHeads],
        [iNumTokens, iModelDim // iHeads, iHeads],

        [iNumTokens, iNumTokens, iHeads],
        [iNumTokens, iModelDim // iHeads, iHeads],
        [iNumTokens, iModelDim],

        [iNumTokens, iModelDim],

        [iNumTokens, iMLPDim],
        [iNumTokens, iModelDim],
        [iNumTokens, iModelDim]
    ]

    vecTOMData = [
        TOMOpData(vecBiasShape = [1, iModelDim // iHeads, iHeads]),
        TOMOpData(vecBiasShape = [1, iModelDim // iHeads, iHeads]),
        TOMOpData(vecBiasShape = [1, iModelDim // iHeads, iHeads]),
    
        TOMOpData(vecUnaryOps = [UnaryOperation(partial(ScalarMult, fScalar = 1 / math.sqrt(iModelDim // iHeads))), UnaryOperation(torch.nn.Softmax(dim = -3))]),
        TOMOpData(),
        TOMOpData(vecBiasShape = [1, iModelDim]),

        TOMOpData(strRowSpaceOp = "add", vecUnaryOps = [UnaryOperation(torch.nn.LayerNorm(iModelDim, eps = 1e-6))]),

        TOMOpData(vecUnaryOps = [UnaryOperation(torch.nn.functional.gelu)], vecBiasShape = [1, iMLPDim]),
        TOMOpData(vecUnaryOps = [UnaryOperation(torch.nn.functional.gelu)], vecBiasShape = [1, iModelDim]),
        TOMOpData(strRowSpaceOp = "add", vecUnaryOps = [UnaryOperation(torch.nn.LayerNorm(iModelDim, eps = 1e-6))])
    ]

    mmFlat = ModeMap([Mode(iNumTokens), Mode(iModelDim // iHeads), Mode(iHeads)], [Mode(iNumTokens), Mode(iModelDim)],
                     tStruct = torch.tensor([
                                            [1, 0, 0],
                                            [0, 1, 1]
                                            ]))
    vecModeMaps = [
        TEModeMap(4, 5, mmFlat)
    ]

    adfnnModel = ADFNN(temArch = TEMatrix(tTEM), vecTOMs = vecTOMs, vecModeMaps = vecModeMaps, vecRowData = vecTOMData,
                       sIn = [iNumTokens, iModelDim], vecSOut = vecSOut)    
    #adfnnModel.Setup()

    tInput = ViTInput1D(iImSize, iPatchSize, iChannels, iModelDim)
    tModel = GenADFNN(adfnnModel, iNumStacks = iLayers, vecInitModules = [tInput, torch.nn.LayerNorm(iModelDim, eps = 1e-6)], bResidual = False)

    return tModel



if __name__ == "__main__":

    def TestViT():
        tModel = BuildViT(65, 6, 192, 576, iLayers = 8, iImSize = 32, iChannels = 3, iPatchSize = 4).to("cuda")
        X = torch.randn((128, 3, 32, 32)).to("cuda")
        Y = tModel(X)
        print(Y.shape)

    def TestMLP():
        from Arch.Models.Misc import NLayerMLP
        tModel = BuildMLP(3072, 3072, 10, 10).to("cuda")
        tModel2 = NLayerMLP(3072, [3072]*9, 10, bNormalizeOutput = False, bFlatten = True).to("cuda")
        for i in range(10):
            #print(tModel.vecLayers[0].vecTOLayers[i].vecWeights[0].shape)
            #print(tModel.vecLayers[0].vecTOLayers[i].tBias.shape)
            #print(tModel2.vecLayers[2*i+1].weight.shape)
            tModel2.vecLayers[2*i+1].weight = torch.nn.Parameter(torch.transpose(torch.clone(tModel.vecLayers[0].vecTOLayers[i].vecWeights[0]), 0, 1)).to("cuda")
            tModel2.vecLayers[2*i+1].bias = torch.nn.Parameter(torch.clone(tModel.vecLayers[0].vecTOLayers[i].tBias))

        print(sum(p.numel() for p in tModel.parameters()))
        print(sum(p.numel() for p in tModel2.parameters()))

        tOpt1 = torch.optim.SGD(tModel.parameters(), lr = 1)
        tOpt2 = torch.optim.SGD(tModel2.parameters(), lr = 1)
        #tOpt1 = torch.optim.AdamW(tModel.parameters(), lr = 1)
        #tOpt2 = torch.optim.AdamW(tModel2.parameters(), lr = 1)

        tLossFunc = torch.nn.CrossEntropyLoss()

        X = torch.randn((2, 3, 32, 32)).to("cuda")
        Y = tModel(X)
        Y2 = tModel2(X)

        print(Y)
        print(Y2)
        print("L-inf distance Y->Y2:", torch.max(torch.abs(Y - Y2)))

        for i in range(10):
            print("L-inf distance weights before step:", torch.max(torch.abs(tModel.vecLayers[0].vecTOLayers[i].vecWeights[0] - tModel2.vecLayers[2*i+1].weight.transpose(0,1))))

        T = torch.zeros((2, 10)).to("cuda")
        T[0, 0] = 1
        T[1, 1] = 1
        # T = torch.zeros((10)).to("cuda")
        # T[0] = 1

        l = tLossFunc(Y, T)
        l2 = tLossFunc(Y2, T)

        tOpt1.zero_grad()
        tOpt2.zero_grad()

        l.backward()
        l2.backward()

        #print(tModel.vecLayers[0].vecTOLayers[0].vecWeights[0].grad)
        #print(tModel2.vecLayers[1].weight.grad)
        for i in range(10):
            print("L-inf distance grad:", torch.max(torch.abs(tModel.vecLayers[0].vecTOLayers[i].vecWeights[0].grad - tModel2.vecLayers[2*i+1].weight.grad.transpose(0,1))))

        tOpt1.step()
        tOpt2.step()

        # with torch.no_grad():
        #     for i in range(10):
        #         tModel.vecLayers[0].vecTOLayers[i].vecWeights[0] -= tModel.vecLayers[0].vecTOLayers[i].vecWeights[0].grad
        #         tModel2.vecLayers[2*i+1].weight -= tModel2.vecLayers[2*i+1].weight.grad


        for i in range(10):
            print("L-inf distance weights after step:", torch.max(torch.abs(tModel.vecLayers[0].vecTOLayers[i].vecWeights[0] - tModel2.vecLayers[2*i+1].weight.transpose(0,1))))

    TestMLP()