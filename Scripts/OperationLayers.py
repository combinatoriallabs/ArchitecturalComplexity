'''
Code written by Nicholas J. Cooper.
Released under the MIT license, see the GitHub page for the full legal boilerplate.
tldr: you freely can do whatever you like with this code so long as this message is retained and you cite the GitHub: https://github.com/combinatoriallabs/ArchitecturalComplexity
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
'''



import torch
import copy

from DTAMatrices import *

from Arch.Utils.Utils import *
from Arch.Models.Misc import UnaryOperation

device = "cpu" if not torch.cuda.is_available() else "cuda"

def Conv1dUnfold(F: torch.tensor, iK: int = 5, iS: int = 1, iP: int = 2) -> torch.tensor:
    '''
    Returns a tensor of shape (Ch, H', iK)
    '''
    R = torch.nn.functional.pad(F, (iP, iP))
    return R.unfold(2, iK, iS)

def Conv2dUnfold(F: torch.tensor, iK: int = 5, iS: int = 1, iP: int = 2) -> torch.tensor:
    '''
    Returns tensor of shape (Ch, H', W', iK, iK)
    '''
    R = torch.nn.functional.pad(F, (iP, iP, iP, iP))
    return R.unfold(2, iK, iS).unfold(3, iK, iS)#.permute(0, 2, 3, 1, 4, 5) #uncomment this to switch to (H', W', Ch, iK, iK) format

def ScalarMult(X: torch.tensor, fScalar: float) -> torch.tensor: return fScalar * X


class TOMOpData:
    def __init__(self, strRowSpaceOp: str = "mult", strContractOp: str = "add", vecUnaryOps: list[UnaryOperation] = [], vecBiasShape: list[int] = None):
        self.strRowSpaceOp = strRowSpaceOp
        self.strContractOp = strContractOp
        self.vecUnaryOps: list[UnaryOperation] = vecUnaryOps
        self.vecBiasShape = vecBiasShape



class TOLayer(torch.nn.Module):
    def __init__(self, tomOp: TOMatrix, vecInputs: list[TOInput], sOut: list[int], strNormalization: str = "uniform", 
                 bLearned: bool = True, bResidual: bool = False, 
                 opData: TOMOpData = None, device = device) -> None:
        '''
        Expects a fully setup TOMatrix as input.
        Supported operations: {add, mult, min, max}
        '''
        super().__init__()

        assert strNormalization.lower() in ["uniform", "normal"], "Unsupported normalization mode {}".format(strNormalization)
        assert tomOp.IsSetup(), "TOLayer requires a fully setup DTAMatrix"

        #operation data
        self.tomOp = tomOp
        self.opData = opData if opData is not None else TOMOpData()

        #parameter storage
        self.vecWeights = []
        self.tBias = None

        #input mapping
        self.vecInputs = vecInputs
        self.vecInputIndices = [idx for inp in self.vecInputs for idx in inp.vecInputIdx]
        self.mapRowIdxToInput = [None for _ in range(self.tomOp.iA)]
        for i in range(len(self.vecInputs)):
            inp = self.vecInputs[i]
            for idx in inp.vecInputIdx:
                self.mapRowIdxToInput[idx] = i
        self.mapRowIdxToWeight = [None for _ in range(self.tomOp.iA)]

        #shape data
        self.sOut = sOut
        self.dInputReshapes = {}
        self.vecOutputPerm = None
        self.iBatchModes = 0
        self.vecContractionTiming = None
        self.vecContracts = None
        self.vecInputReshapes = []
        self.vecRowPerm = [i for i in range(self.tomOp.iA)]

        #FAA - frequently accessed attributes
        self.device = device
        self.strNorm = strNormalization.lower()
        self.bLearned = bLearned
        self.bRes = bResidual

        self.bGeneralizedLayer = True

        #setup
        self.ComputePermutations()
        self.SetupParameterTensors()
        self.ComputeContractionTiming()

        return
    
    def __str__(self) -> str:
        strRet = "Shape:  " + str(torch.max(self.tomOp.tShape, dim = 0)[0].numpy()) + "\n"
        strRet += "Struct: " + str(self.tomOp.tStruct[0, :].float().numpy())
        if 0 in self.vecInputIndices: strRet += " <--"
        strRet += "\n"
        for i in range(1, self.tomOp.iA):
            strRet += "        " + str(self.tomOp.tStruct[i, :].float().numpy()) 
            if i in self.vecInputIndices: strRet += " <--"        
            strRet += "\n"
        strRet += "        " + str(self.tomOp.tContract.float().numpy()).replace("1.", "||")
        
        return strRet
    
    def OutputShape(self) -> list[int]:
        return self.tomOp.OutputShape()
    
    def GetSM(self) -> torch.tensor:
        return self.tomOp.GetSM(self.iBatchModes)
    
    def GetShape(self) -> torch.tensor:
        return self.tomOp.GetShape()
    
    def ComputePermutations(self) -> None:
        #Compute input permutations
        for inp in self.vecInputs:
            for idx in inp.vecInputIdx:
                tS = torch.clone(self.tomOp.tShape[idx, :])
                #tS = tS[tS != 0]
                tS[tS == 0] = 1 #avoid doing unsqueezing later
                vecS = list(tS.int())
                self.dInputReshapes[idx] = vecS

        #compute output permutation
        vecOS = self.tomOp.OutputShape()
        self.vecOutputPerm = ListPermuataion(vecOS, self.sOut)
        if not self.vecOutputPerm:
            print("Error: TOLayer got requested output shape {} but TOM output shape {}".format(self.sOut, vecOS))

        return
    
    def SetBatchModes(self, iBatchModes: int, vecInputBMs: list[int]) -> None:
        '''
        Updates permutations and structure matrix to accept iBatchModes "dummy" modes as input.
        This method will be automatically called if batched input is passed in forward().
        Assumes the batch modes are the leftmost modes.
        In the case when some inputs have more batch modes than others, it is assumed the inputs agree on the leftmost batch modes.
        Assumes vecInputBMs is ordered based on the given input ordering at construction time.
        '''

        if self.iBatchModes == iBatchModes: return

        assert iBatchModes > 0, "Non-positive batch dimensions are not supported, got {}".format(iBatchModes)
        #print(iBatchModes, vecInputBMs)
        iOldBM = self.iBatchModes

        self.iBatchModes = iBatchModes
        self.vecOutputPerm = [i for i in range(self.iBatchModes)] + [i + self.iBatchModes - iOldBM for i in self.vecOutputPerm[iOldBM:]]

        tBatchStruct = torch.zeros((self.tomOp.iA, self.iBatchModes))
        for i in range(len(vecInputBMs)):
            for idx in self.vecInputs[i].vecInputIdx:
                j = self.vecRowPerm[idx]
                tBatchStruct[j, :vecInputBMs[i]] = 1

        self.tomOp.tStruct = self.tomOp.tStruct[:, iOldBM:]
        self.tomOp.tStruct = torch.cat((tBatchStruct, self.tomOp.tStruct), dim = 1)

        tBatchContract = torch.zeros((self.iBatchModes))
        self.tomOp.tContract = self.tomOp.tContract[iOldBM:]
        self.tomOp.tContract = torch.cat((tBatchContract, self.tomOp.tContract), dim = 0)

        if self.vecContractionTiming is not None: self.ComputeContractionTiming() #recompute this to incorporate batch modes

        return

    #parameter setup
    def SetupParameterTensors(self) -> None:
        vecWeights = []
        for i in range(self.tomOp.iA):
            if i in self.vecInputIndices: continue
            tS = torch.clone(self.tomOp.tShape[i, :])
            tS[tS == 0] = 1 #avoid doing unsqueezing later
            vecS = [1 for _ in range(self.iBatchModes)] + list(tS.int().numpy())

            tSVar = self.tomOp.tShape[i, :] * self.tomOp.tContract[self.iBatchModes:]
            
            if self.strNorm == "uniform":
                tW = (2 * torch.rand(vecS, device = self.device)) - 1
                
                fVar = torch.sqrt(6 / (torch.sum(tSVar[tSVar > 0])))
                if fVar == torch.inf: fVar = torch.sqrt(6 / torch.sum(tS[tS > 1])) #if we ran into division by zero, switch back

                #fVar *= 0.1
            elif self.strNorm == "normal":
                tW = torch.randn(vecS, device = self.device)
                fVar = torch.sqrt(12 / torch.sum(tS[tS > 1]))

            self.mapRowIdxToWeight[i] = len(vecWeights)

            if self.bLearned: vecWeights.append(torch.nn.Parameter(fVar * tW))
            else: vecWeights.append(fVar * tW)

        if self.bLearned: self.vecWeights = torch.nn.ParameterList(vecWeights)
        else: self.vecWeights = vecWeights

        if self.opData.vecBiasShape is not None and len(self.opData.vecBiasShape) > 0:
            fVar = math.sqrt(12 / sum(self.opData.vecBiasShape))
            if self.bLearned: self.tBias = torch.nn.Parameter(fVar * torch.randn(self.opData.vecBiasShape, device = self.device))
            else: self.tBias = fVar * torch.randn(self.opData.vecBiasShape, device = self.device)

        #check for unary ops which need grad tracking
        vecModules = []
        for i in range(len(self.opData.vecUnaryOps)):
            if isinstance(self.opData.vecUnaryOps[i], torch.nn.Module): vecModules.append(self.opData.vecUnaryOps[i])
        self.vecModules = torch.nn.ModuleList(vecModules)

        return
    
    def OptimizeColPermutation(self) -> None:
        #permute the columns of the 1-SM so the first input does not have to be .reshape()'d
        vecS = self.tomOp.GetShape(self.vecInputIndices[0])
        vecInS = self.vecInputs[0].sIn + [min(vecS) for _ in range(len(vecS) - len(self.vecInputs[0].sIn))]

        vecPerm = ListPermuataion(vecS, vecInS)

        self.tomOp.PermuteCols(vecPerm)
        self.ComputePermutations()
        self.SetupParameterTensors()

        return

    def OptimizeRowPermutation(self) -> None:
        '''
        Permutes the rows of the structure matrix such that the highest order (non-full) contraction is processed first
        '''
        #TODO: extend this to the arbitrary arity case. Need to recursively perform this permutation finding until iA is consumed. For low arity the current impl is good
        tBestCol = None
        iBestSize = 0
        for idx in list(torch.where(self.tomOp.tContract == 1)[0]):
            tCol = self.tomOp.tStruct[:, idx:idx+1]
            if torch.sum(tCol) == self.tomOp.iA: continue #skip full contractions

            tQ = torch.sum(torch.logical_xor(tCol, self.tomOp.tStruct), dim = 0)
            tIdx = torch.where(tQ == 0)[0] #mode indices

            iSz = 1
            for i in range(tIdx.shape[0]): iSz *= self.tomOp.Shape(tIdx[i])

            if iSz > iBestSize:
                iBestSize = iSz
                tBestCol = tCol

        if tBestCol is None: return #quit out if no optimization is possible

        tBestCol = tBestCol[:, 0] #shed the extra mode
        vecRowPerm = ListPermuataion(list(tBestCol), list(torch.sort(tBestCol, descending = True)[0]))

        self.tomOp.PermuteRows(vecRowPerm)

        dNewInputReshapes = {}
        for i in range(len(self.vecInputIndices)):
            idxOld = self.vecInputIndices[i]
            idxNew = vecRowPerm.index(idxOld)
            dNewInputReshapes[idxNew] = self.dInputReshapes[idxOld]
        self.dInputReshapes = dNewInputReshapes
        self.vecInputIndices = [vecRowPerm.index(self.vecInputIndices[j]) for j in range(len(self.vecInputIndices))]
        self.vecRowPerm = [vecRowPerm.index(self.vecRowPerm[j]) for j in range(len(self.vecRowPerm))]
        self.mapRowIdxToInput = [self.mapRowIdxToInput[vecRowPerm[j]] for j in range(self.tomOp.iA)]
        self.mapRowIdxToWeight = [self.mapRowIdxToWeight[vecRowPerm[j]] for j in range(self.tomOp.iA)]

        self.ComputeContractionTiming()

        return

    def ComputeContractionTiming(self) -> None:
        self.vecContractionTiming = []
        for i in range(1, self.tomOp.iA):
            tProcessableC = torch.where(torch.where(torch.sum(self.tomOp.tStruct[i+1:, :], dim = 0) == 0, 1, 0) * self.tomOp.tContract == 1)[0]
        
            self.vecContractionTiming.append(tProcessableC)

            self.vecContracts = tuple(torch.where(self.tomOp.tContract == 1)[0])

        return
    
    def ComputeCost(self) -> dict:
        return self.tomOp.ComputeCost()
    

    def GetParam(self, idx: int) -> torch.tensor:
        return self.vecWeights[self.mapRowIdxToWeight[idx]]

        
    def RowSpaceTensorOp(self, X: torch.tensor, Y: torch.tensor) -> torch.tensor:
        #return X * Y
    
        if self.opData.strRowSpaceOp == "mult": return X * Y
        elif self.opData.strRowSpaceOp == "add": return X + Y
        elif self.opData.strRowSpaceOp == "min": return torch.min(X, Y)[0]
        elif self.opData.strRowSpaceOp == "max": return torch.max(X, Y)[0]

    def ContractionTensorOp(self, X: torch.tensor, dim: tuple[int]) -> torch.tensor:
        if self.opData.strContractOp == "add": return torch.sum(X, dim = dim, keepdim = True)
        else:
            for iD in dim:
                if self.opData.strContractOp == "mult": X = torch.prod(X, dim = iD, keepdim = True)
                elif self.opData.strContractOp == "min": X = torch.min(X, dim = iD, keepdim = True)[0]
                elif self.opData.strContractOp == "max": X = torch.max(X, dim = iD, keepdim = True)[0]

            return X



    def zero_grad(self, set_to_none = True) -> None:
        for i in range(len(self.vecWeights)): self.vecWeights[i].grad = None
        for tm in self.vecModules: tm.zero_grad(set_to_none = set_to_none)
        if self.tBias is not None: self.tBias.grad = None

        return

    def forward(self, vecX: list[torch.tensor], bSample: bool = False) -> torch.tensor:
        if not isinstance(vecX, list): vecX = [vecX]
        if bSample: self.SetupParameterTensors()

        assert len(vecX) == len(self.vecInputs), "Expected {} inputs but got {}".format(len(self.vecInputs), len(vecX))

        #TODO: there are bugs in the non-homogenous batch mode case!
        iMaxBM = len(vecX[0].shape) - len(self.vecInputs[0].sIn)
        vecBatchShapes = [list(vecX[0].shape)[:iMaxBM]]
        vecBMs = [iMaxBM]
        for i in range(1, len(vecX)):
            iBM = len(vecX[i].shape) - len(self.vecInputs[i].sIn)
            vecBMs.append(iBM)
            vecBatchShapes.append(list(vecX[i].shape)[:iBM])
            if iBM > iMaxBM:
                iMaxBM = iBM

        self.SetBatchModes(iMaxBM, vecBMs)

        if 0 in self.vecInputIndices:
            idxIn = self.mapRowIdxToInput[0]
            R = vecX[idxIn].reshape(vecBatchShapes[idxIn] + [1 for _ in range(iMaxBM - vecBMs[idxIn])] + self.dInputReshapes[0])
            idx = 0
        else:
            R = self.vecWeights[0]
            idx = 1
        R = R

        for i in range(1, self.tomOp.iA):
            if i in self.vecInputIndices:
                idxIn = self.mapRowIdxToInput[i]
                R = self.RowSpaceTensorOp(R, vecX[idxIn].reshape(vecBatchShapes[idxIn] + [1 for _ in range(iMaxBM - vecBMs[idxIn])] + self.dInputReshapes[i]))#.contiguous()
            else:
                tW = self.vecWeights[idx]
                for i in range(self.iBatchModes): tW = tW.unsqueeze(0)
                R = self.RowSpaceTensorOp(R, tW)
                idx += 1
            R = R
            if self.vecContractionTiming[i - 1].shape[0] > 0: R = self.ContractionTensorOp(R, dim = tuple(self.vecContractionTiming[i - 1]))

        vecSqzDims = [idx for idx in range(len(R.shape)) if idx >= iMaxBM and R.shape[idx] == 1]
        T = torch.squeeze(R, dim = vecSqzDims)
        T = T.permute(self.vecOutputPerm)#.contiguous()
        if self.tBias is not None: T += self.tBias

        if self.bRes: Y = T + X
        else: Y = T

        for op in self.opData.vecUnaryOps: Y = op(Y)

        return Y


def SetupTOM(tomOp: TOMatrix, vecInputs: list[TOInput], sOut: list[int], vecFreeSizes: list[int] = [2]) -> bool:
    
    tomOp.ResetShape()
    
    for inp in vecInputs:
        if not tomOp.PropagateShape(inp.sIn, inp.vecInputIdx): return False

    #Next, process the requested output shape
    vecMissing = copy.deepcopy(sOut)
    vecOS = tomOp.OutputShape()
    vecOM = tomOp.OpenModes()
    for j in range(len(vecOS)):
        if vecOS[j] in vecMissing: vecMissing.remove(vecOS[j])

    if len(vecMissing) > len(vecOM):
        print("Need {} free modes for output shape {}, but operation only has {} open modes".format(len(vecMissing), sOut, len(vecOM)))
        return False
    
    for i in range(len(vecMissing)): tomOp.SetOpenMode(vecOM[i], vecMissing[i])

    #check for leftovers
    vecOM = tomOp.OpenModes()
    if len(vecOM) > 0: print("{} Extra open output modes detected".format(len(vecOM)))
    vecCOM = tomOp.OpenModes(bContracted = True)
    if len(vecCOM) > 0: print("{} Extra open contracted modes detected".format(len(vecCOM)))
    
    vecOM += vecCOM
    while len(vecOM) > 0:
        idxF = vecOM[0]
        iVal = vecFreeSizes[0]
        tomOp.SetOpenMode(idxF, iVal)

        vecOM = tomOp.OpenModes() + tomOp.OpenModes(bContracted = True)
        if len(vecFreeSizes) > 1: vecFreeSizes = vecFreeSizes[1:]
    
    return True


def BuildTOLayer(tomOp: TOMatrix, sIn: list[int], sOut: list[int], vecInputIdx: list[int] = [0], vecFreeSizes: list[int] = [2],
                  strNormalization: str = "uniform", bBias: bool = False, bLearned: bool = True, bResidual: bool = False, 
                  strRowSpaceOp: str = "mult", strContractionOp: str = "add", device = device) -> TOLayer:
    
    vecInputs = [TOInput(sIn, vecInputIdx)]

    if not SetupTOM(tomOp, vecInputs, sOut, vecFreeSizes = vecFreeSizes): return None

    opData = TOMOpData(strRowSpaceOp = strRowSpaceOp, strContractOp = strContractionOp, vecBiasShape = tomOp.OutputShape() if bBias else None)

    return TOLayer(tomOp, vecInputs, sOut,
                     strNormalization = strNormalization, bLearned = bLearned, bResidual = bResidual, 
                     opData = opData, device = device)


def BuildMITOLayer(tomOp: TOMatrix, vecInputs: list[TOInput], sOut: list[int], vecFreeSizes: list[int] = [2],
                  strNormalization: str = "uniform", bBias: bool = False, bLearned: bool = True, bResidual: bool = False, 
                  strRowSpaceOp: str = "mult", strContractionOp: str = "add", device = device) -> TOLayer:

    if not SetupTOM(tomOp, vecInputs, sOut, vecFreeSizes = vecFreeSizes): return False

    opData = TOMOpData(strRowSpaceOp = strRowSpaceOp, strContractOp = strContractionOp, vecBiasShape = tomOp.OutputShape() if bBias else None)

    return TOLayer(tomOp, vecInputs, sOut,
                     strNormalization = strNormalization, bLearned = bLearned, bResidual = bResidual, 
                     opData = opData, device = device)



def main():
    def TestCone():
        #Test for Cone Product
        X = torch.randn((128, 3, 32, 32)).to(device)

        tConeStruct = torch.tensor([
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 1],

            [0, 0, 1, 0]
        ])

        sIn = [3, 32, 32]
        sOut = [3, 32, 32]

        tomOp = TOMatrix(tConeStruct)

        tConeLayer = BuildTOLayer(tomOp, sIn, sOut, vecInputIdx = [0], vecFreeSizes = [2], bBias = True)
        tConeLayer.OptimizeRowPermutation()
        tConeLayer.OptimizeColPermutation()
        
        print(tConeLayer)
    
        Y = tConeLayer(X)
        print(Y.shape)

    def TestMultiBatchSize():
        X1 = torch.randn((3, 32)).to(device)
        X2 = torch.randn((128, 65, 32, 32, 3, 3)).to(device)

        inp1 = TOInput([3, 32], [0])
        inp2 = TOInput([32, 32, 3, 3], [1])

        tStruct = torch.tensor([
            [0, 0, 1, 0, 0, 1,],
            [1, 0, 0, 1, 1, 1,],
            [0, 1, 1, 0, 0, 1,],

            [0, 0, 1, 0, 0, 0]
        ])

        sOut = [32, 32, 3, 3, 2]

        toLayer = BuildMITOLayer(TOMatrix(tStruct), [inp1, inp2], sOut)

        Y = toLayer([X1, X2])
        print(Y.shape)

    TestMultiBatchSize()
    #TestCone()

if __name__ == "__main__":
    main()