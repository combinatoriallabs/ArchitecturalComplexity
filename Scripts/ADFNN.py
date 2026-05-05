'''
Code written by Nicholas J. Cooper.
Released under the MIT license, see the GitHub page for the full legal boilerplate.
tldr: you freely can do whatever you like with this code so long as this message is retained and you cite the GitHub: https://github.com/combinatoriallabs/ArchitecturalComplexity
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
'''



import torch

from OperationLayers import *
from DTAMatrices import *

from Arch.Utils.Utils import *


device = "cpu" if not torch.cuda.is_available() else "cuda"


@dataclass
class TEModeMap:
    iSource: int
    iDest: int
    mmMap: ModeMap
    iSourceIdx: int = -1


class ADFNN(torch.nn.Module):
    def __init__(self, temArch: TEMatrix, vecTOMs: list[TOMatrix], vecModeMaps: list[TEModeMap] = None, 
                 sIn: list[int] = None, vecSOut: list[list[int]] = None,
                 vecRowData: list[TOMOpData] = None,
                 bLearned: bool = True, device = device) -> None:
        '''
        Main top-level 2-SM trainable module.
        If the TOMs are pre-setup, sIn and vecSOut are not necessary. Otherwise, they must be specified.
        If the TOMs are not pre-setup, AND mode maps are provided, the post mode-map input shapes will be used to initialize the TOMs.
        If ModeMaps are not specified in the TEMatrix, empty ones will be created.
        If no ModeMap is present for a TOM connection of constant order, a copy map will be created and the output shape passed directly to the next TOM as input. Explicity pass a pre-created ModeMap to override this behaviour. 
        '''
        
        super().__init__()

        assert temArch.IsSetup(), "ADFNN expects a Setup() TEM"
        assert temArch.iE == len(vecTOMs), "ADFNN got a TEM with {} rows, but got {} TOMs".format(temArch.iE, len(vecTOMs))

        self.temArch = temArch
        self.vecTOMs = vecTOMs
        self.vecRowData = [None for _ in range(self.temArch.iE)]
        if vecRowData is not None:
            for i in range(len(vecRowData)): self.vecRowData[i] = vecRowData[i]
        for i in range(self.temArch.iE):
            if self.vecRowData[i] is None: self.vecRowData[i] = TOMOpData()
        
        self.vecTOLayers: list[TOLayer] = []

        self.bSetup = False

        self.bLearned = bLearned
        self.device = device

        self.bGeneralizedLayer = True

        #Grab shapes from the TOMs if not explicity passed at construction time
        if sIn is None:
            assert self.vecTOMs[0].IsSetup(), "ADFNN expects pre-initialized TOMs if shapes are not provided"
            self.sIn = self.vecTOMs[0].InputShape()
        else: self.sIn = sIn

        if vecSOut is None:
            self.vecSOut = []
            for i in range(self.temArch.iE):
                self.vecSOut.append(self.vecTOMs[i].OutputShape())
        else: self.vecSOut = vecSOut

        #internal tensorial storage for MMs. Source indices are offset by +1
        self.tensModeMaps = [[[None for _ in range(self.temArch.vecA[i - 1])] for _ in range(self.temArch.iE)] for i in range(self.temArch.iE)]
        self.tensModeMaps[0] = [[None] for _ in range(self.temArch.iE)]
        
        #Pull in the mode maps, if provided
        if vecModeMaps is not None:
            for mm in vecModeMaps:
                if mm.iDest not in self.temArch.vecDestinations[mm.iSource + 1][mm.iSourceIdx]:
                    print("Error: ADFNN got a mode map for TOMs {}->{} but TOM {} has destinations {} for index {}".format(mm.iSource, mm.iDest, mm.iSource, self.temArch.vecDestinations[mm.iSource + 1], mm.iSourceIdx))
                    continue
                self.tensModeMaps[mm.iSource + 1][mm.iDest][mm.iSourceIdx] = mm.mmMap

        return
    

    def __str__(self) -> str:
        strRet = "TEM:\n"
        strRet += str(self.temArch.tStruct.float().numpy()) + "\n"
        for i in range(len(self.vecTOMs)):
            strRet += "------------------------------\n"
            strRet += "TOM {}/{}\n".format(i + 1, len(self.vecTOMs))
            strRet += self.vecTOMs[i].GetOperationString() + "\n"

        return strRet

    def Setup(self) -> bool:

        bSetup = True

        for i in range(self.temArch.iE):

            #process the deps
            vecTOIn = []
            for j in range(len(self.temArch.vecDependencies[i])):
                iDep = self.temArch.vecDependencies[i][j].iDep
                iSrcIdx = self.temArch.vecDependencies[i][j].iDepIdx
                
                #collect the input shape
                if self.tensModeMaps[iDep + 1][i][iSrcIdx] is not None:
                    sIn = self.tensModeMaps[iDep + 1][i][iSrcIdx].OutputShape()
                else:
                    if iDep == -1: sIn = self.sIn
                    elif iSrcIdx == self.temArch.vecA[iDep] - 1: sIn = self.vecSOut[iDep]
                    else:
                        sIn = self.vecTOMs[iDep].GetShape(iSrcIdx, bRemoveZeros = True)

                    inO = self.vecTOMs[i].GetOrder(self.temArch.vecDependencies[i][j].vecInputIdx[0]) #for now, assume the orders are all correct
                    if inO == len(sIn):
                        #if this "looks like" a copy mode-map, just create it
                        if self.vecTOMs[i].IsSetup(): sOut = self.vecTOMs[i].GetShape(self.temArch.vecDependencies[i][j].vecInputIdx[0], bRemoveZeros = True)
                        else: sOut = sIn

                        self.tensModeMaps[iDep + 1][i][iSrcIdx] = ModeMap([Mode(iSz) for iSz in sIn], [Mode(iSz) for iSz in sOut])

                        if ListEquals(sIn, sOut):
                            vecPerm = ListPermuataion(sIn, sOut)
                            for k in range(len(sIn)):
                                if not self.tensModeMaps[iDep + 1][i][iSrcIdx].AddCopy(vecPerm[k], k):
                                    print("Error encountered adding copy btwn modes {} and {} of TOMs {} and {}".format(vecPerm[k], k, iDep, i))
                                    bSetup = False
                        else:
                            print("Warning: initialized modemap for Dep. btwn TOMs {}->{}: {}, but still require a tStruct".format(iDep, i, self.temArch.vecDependencies[i][j]))
                            bSetup = False
                    else:
                        print("Warning: require a ModeMap between TOMs {} and {}".format(iDep, i))
                        bSetup = False

                #print("After MM: {}".format(sIn))
                toIn = TOInput(sIn, self.temArch.vecDependencies[i][j].vecInputIdx)

                #if the TOM is already setup, make sure the shapes line up
                if self.vecTOMs[i].IsSetup():
                    for idx in toIn.vecInputIdx:
                        sChk = self.vecTOMs[i].GetShape(idx, bRemoveZeros = True)
                        if not ListEquals(toIn.sIn, sChk):
                            print("Error: found shape mismatch between TOM {} and input {} at index {}".format(i, toIn, idx))

                            print("TEM:")
                            print(self.temArch.tStruct)
                            for idxP in range(i+1):
                                print("TOM {}: ".format(idxP), self.vecTOMs[idxP].GetOperationString())
                                #print(f"tShape:\n{self.vecTOMs[idxP].tShape}")
                            print("Input row {}, shape {}, sChk {}".format(idx, toIn.sIn, sChk))

                            #input()
                            bSetup = False

                vecTOIn.append(toIn)
                    

            #finally, create the TOMs if need be
            if not self.vecTOMs[i].IsSetup():
                print("Setting up TOM {} for inputs {} -> {}".format(i, vecTOIn, self.vecSOut[i]))
                if not SetupTOM(self.vecTOMs[i], vecTOIn, self.vecSOut[i]):
                    print("Shape error encountered while setting up TOM {}".format(i))


            #create the corresponding TOLayer
            tolLayer = TOLayer(self.vecTOMs[i], vecTOIn, self.vecSOut[i],
                     strNormalization = "uniform", bLearned = self.bLearned, bResidual = False, 
                     opData = self.vecRowData[i], device = device)

            self.vecTOLayers.append(tolLayer)
            print("Created TOLayer {} w/ output shape {} and unary ops {}".format(i, self.vecTOLayers[-1].OutputShape(), [str(o) for o in self.vecTOLayers[-1].opData.vecUnaryOps]))
            #print(self.vecTOLayers[-1].tomOp.tShape)
            #print(self.vecTOLayers[-1].vecOutputPerm, self.vecSOut[i])

        #keeping torch happy
        if self.bLearned: self.vecTMTOLayers = torch.nn.ModuleList(self.vecTOLayers)

        
        self.bSetup = bSetup
        return self.bSetup
    

    #--------------Accessors--------------#

    def IsSetup(self) -> bool: return self.bSetup
    
    def GetTEM(self) -> torch.tensor: return self.temArch.tStruct

    def GetAD(self, idx: int = -1) -> int:
        vecS = self.vecTOMs[idx].OutputShape()
        iSz = 1
        for s in vecS: iSz *=s
        return iSz



    def zero_grad(self, set_to_none = True):
        for toL in self.vecTOLayers: toL.zero_grad(set_to_none = set_to_none)
        return

    def forward(self, x: torch.tensor) -> torch.tensor:
        if not self.bSetup:
            print("Error: Tried to call ADFNN.forward() but model is not setup")
            return None
        
        vecT = [x]

        for i in range(self.temArch.iE):
            vecX = []
            for j in range(len(self.temArch.vecDependencies[i])):
                iDep = self.temArch.vecDependencies[i][j].iDep
                iSrcIdx = self.temArch.vecDependencies[i][j].iDepIdx

                if iDep == -1 or iSrcIdx == self.temArch.vecA[iDep] - 1: 
                    vecX.append(self.tensModeMaps[iDep + 1][i][iSrcIdx](vecT[iDep + 1]))
                else:
                    vecX.append(self.tensModeMaps[iDep + 1][i][iSrcIdx](self.vecTOLayers[iDep].GetParam(iSrcIdx).squeeze()))
            tR = self.vecTOLayers[i](vecX) #main TOM calculation            
            vecT.append(tR)

            # print("Shape after processing TOM {}: {}".format(i, vecT[-1].shape))
            # input()


        return vecT[-1]



class GenADFNN(torch.nn.Module):
    def __init__(self, adfnnModel: ADFNN, iNumStacks: int, vecInitModules: list[torch.nn.Module] = [], iResidualFq: int = -1, vecOutModules: list[torch.nn.Module] = []) -> None:
        '''
        bResidual adds simple skip connections around each copy of the input ADFNN. No residual connections are added to InitModules.
        '''
        super().__init__()

        self.vecPerm = None
        if iNumStacks > 1 or iResidualFq > 0:
            assert ListEquals(adfnnModel.sIn, adfnnModel.vecSOut[-1]), "Stacked ADFNN mode and skip connections require shape preserving ADFNNs"
            self.vecPerm = ListPermuataion(adfnnModel.vecSOut[-1], adfnnModel.sIn)

        self.vecInputModules = vecInitModules

        self.vecLayers: list[ADFNN] = []
        for _ in range(iNumStacks):
            tL = copy.deepcopy(adfnnModel)
            tL.Setup()
            self.vecLayers.append(tL)

        self.vecOutputModules = vecOutModules

        self.vecTMLayers = torch.nn.ModuleList(self.vecInputModules + self.vecLayers + self.vecOutputModules) #keeping torch happy

        self.iResFq = iResidualFq

        self.bGeneralizedLayer = True

        return
    
    def GetAD(self, idx: int) -> int: return self.vecLayers[-1].GetAD(idx)
    

    def zero_grad(self, set_to_none = True):
        for tm in self.vecTMLayers: tm.zero_grad(set_to_none = set_to_none)
        return
    
    def InputModules(self, X: torch.tensor) -> torch.tensor:
        for tM in self.vecInputModules: X = tM(X)
        return X

    def forward(self, X: torch.tensor) -> torch.tensor:
        for tM in self.vecInputModules: X = tM(X)

        if self.vecPerm is not None:
            iBMs = len(X.shape) - len(self.vecPerm)
            vecPerm = [i for i in range(iBMs)] + [iBMs + idx for idx in self.vecPerm]

        if self.iResFq > 0: I = X

        for i in range(len(self.vecLayers)):
            Y = self.vecLayers[i](X)
            if self.vecPerm is not None: Y = torch.permute(Y, vecPerm)
            if self.iResFq > 0 and (i + 1) % self.iResFq == 0:
                X = Y + I
                I = X
            else: X = Y
        
        #print(X.shape)
        #input()

        for tM in self.vecOutputModules: X = tM(X)

        return X



if __name__ == "__main__":

    def TestResMLP():
        tTOStructMM = torch.tensor([
            [1, 1, 0],
            [0, 1, 1],

            [0, 1, 0]
        ])

        tTOStructSC = torch.tensor([
            [1, 1],
            [1, 1],

            [0, 0]
        ])

        tom1 = TOMatrix(tTOStructMM)
        tom2 = TOMatrix(tTOStructSC)

        tom3 = TOMatrix(tTOStructMM)
        tom4 = TOMatrix(tTOStructSC)


        tTEStruct = torch.tensor([
            [1, 1, 1, 0, 0, 0, 0],
            [1, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 1, 1]
        ])

        temArch = TEMatrix(tTEStruct)
        print(temArch.vecDestinations)
        print(temArch.vecDependencies)

        vecTOMData = [
            TOMOpData(),
            TOMOpData(strRowSpaceOp = "add", vecUnaryOps = [UnaryOperation(torch.nn.functional.leaky_relu, iDim = None)]),
            TOMOpData(),
            TOMOpData(strRowSpaceOp = "add", vecUnaryOps = [UnaryOperation(torch.nn.functional.leaky_relu, iDim = None)])
        ]

        adfnnModel = ADFNN(temArch = temArch, vecTOMs = [tom1, tom2, tom3, tom4], sIn = [65, 192], vecSOut = [[65, 192], [65, 192], [65, 192], [65, 192]],
                           vecRowData = vecTOMData)
        
        adfnnModel.Setup()

        stackedModel = GenADFNN(adfnnModel, iNumStacks = 4)

        X = torch.randn((128, 65, 192)).to(device)
        Y = adfnnModel(X)
        Y2 = stackedModel(X)
        print(Y.shape, CountParams(adfnnModel))
        print(Y2.shape, CountParams(stackedModel))


    def TestResConv2():
        tTOStructMM = torch.tensor([
            [1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1],

            [0, 0, 1, 1, 1, 0]
        ])

        tTOStructSC = torch.tensor([
            [1, 1, 1],
            [1, 1, 1]
        ])

        tom1 = TOMatrix(tTOStructMM)
        tom2 = TOMatrix(tTOStructSC, tContract = torch.tensor([0, 0, 0]))

        tom3 = TOMatrix(tTOStructMM)
        tom4 = TOMatrix(tTOStructSC, tContract = torch.tensor([0, 0, 0]))


        tTEStruct = torch.tensor([
            [1, 1, 1, 0, 0, 0, 0],
            [1, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 1, 1]
        ])

        temArch = TEMatrix(tTEStruct)
        print(temArch.vecDestinations)
        print(temArch.vecDependencies)

        mmConv2 = ModeMap([3, 32, 32], [32, 32, 3, 5, 5])
        mmConv2.AddUnfold(1, [0, 3])
        mmConv2.AddUnfold(2, [1, 4])

        vecMMs = [TEModeMap(-1, 0, mmConv2), TEModeMap(1, 2, mmConv2)]

        adfnnModel = ADFNN(temArch = temArch, vecTOMs = [tom1, tom2, tom3, tom4], sIn = [3, 32, 32], vecSOut = [[3, 32, 32], [3, 32, 32], [3, 32, 32], [3, 32, 32]],
                           vecTOMOpTypes = [["mult", "add"], ["add", "add"], ["mult", "add"], ["add", "add"]], vecModeMaps = vecMMs)
        
        adfnnModel.Setup()

        X = torch.randn((128, 3, 32, 32)).to(device)
        Y = adfnnModel(X)
        print(Y.shape)


    def TestSomethingWeird():
        tTOStruct1 = torch.tensor([
            [1, 1, 1, 0, 1, 0],
            [0, 0, 1, 1, 0, 1],
            [1, 0, 1, 0, 1, 0],

            [1, 0, 1, 0, 1, 0]
        ])

        tTOStruct2 = torch.tensor([
            [1, 1, 1, 1],
            [0, 1, 0, 1],
            [1, 0, 1, 1],

            [0, 0, 1, 1]
        ])

        tTOStructSC03 = torch.tensor([
            [1, 1, 1],
            [1, 1, 1],

            [0, 0, 0]
        ])

        tTOStructSC04 = torch.tensor([
            [1, 1, 1, 1],
            [1, 1, 1, 1],

            [0, 0, 0, 0]
        ])

        tom1 = TOMatrix(tTOStruct1)
        tom2 = TOMatrix(tTOStruct2)

        tomSC03 = TOMatrix(tTOStructSC03)
        tomSC04 = TOMatrix(tTOStructSC04)


        tTEStruct = torch.tensor([
            [1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 1]
        ])

        temArch = TEMatrix(tTEStruct)
        print(temArch.vecDestinations)
        print(temArch.vecDependencies)

        mm3To4 = ModeMap([6, 8, 8], [4, 6, 8, 8])
        mm3To4.AddUnfold(1, [0, 2])

        mm2To3 = ModeMap([4, 6], [6, 8, 8])
        mm2To3.AddUnfold(0, [1, 2])

        vecMMs = [TEModeMap(0, 1, mm3To4), TEModeMap(2, 3, mm2To3)]

        adfnnModel = ADFNN(temArch = temArch, vecTOMs = [tom1, tomSC04, tom2, tomSC03], sIn = [4, 6, 8, 8], vecSOut = [[6, 8, 8], [4, 6, 8, 8], [4, 6], [6, 8, 8]],
                           vecTOMOpTypes = [["mult", "add"], ["add", "add"], ["mult", "add"], ["add", "add"]], vecModeMaps = vecMMs)
        
        adfnnModel.Setup()

        X = torch.randn((16, 16, 4, 6, 8, 8)).to(device)
        Y = adfnnModel(X)
        print(Y.shape)


    def TestParamSkip():
        tTOStructMM = torch.tensor([
            [1, 1, 0],
            [0, 1, 1]
        ])

        tTOStructSC = torch.tensor([
            [1, 1],
            [1, 1]
        ])

        tom1 = TOMatrix(tTOStructMM, tContract = torch.tensor([0, 1, 0]))
        tom2 = TOMatrix(tTOStructSC, tContract = torch.tensor([0, 0]))

        tom3 = TOMatrix(tTOStructMM, tContract = torch.tensor([0, 1, 0]))
        tom4 = TOMatrix(tTOStructSC, tContract = torch.tensor([0, 0]))


        tTEStruct = torch.tensor([
            [1, 1, 1, 0, 0, 0],
            [1, 0, 1, 1, 0, 0],
            [0, 1, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 1]
        ])

        temArch = TEMatrix(tTEStruct)
        print(temArch.vecDestinations)
        print(temArch.vecDependencies)

        adfnnModel = ADFNN(temArch = temArch, vecTOMs = [tom1, tom2, tom3, tom4], sIn = [65, 192], vecSOut = [[65, 192], [65, 192], [65, 192], [65, 192]],
                           vecTOMOpTypes = [["mult", "add"], ["add", "add"], ["mult", "add"], ["add", "add"]])
        
        adfnnModel.Setup()

        X = torch.randn((128, 65, 192)).to(device)
        Y = adfnnModel(X)
        print(Y.shape)

        
    TestResMLP()
    # TestResConv2()
    # TestSomethingWeird()
    # TestParamSkip()