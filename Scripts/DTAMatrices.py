'''
Code written by Nicholas J. Cooper.
Released under the MIT license, see the GitHub page for the full legal boilerplate.
tldr: you freely can do whatever you like with this code so long as this message is retained and you cite the GitHub: https://github.com/combinatoriallabs/ArchitecturalComplexity
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
'''



import torch

from dataclasses import dataclass
import math

from Arch.Utils.Utils import *

#--------------Level 1 Structure Matrix Stuff--------------#

#-------Utils-------#

@dataclass
class TOInput:
    sIn: list[int]
    vecInputIdx: tuple[int] = (0)

@dataclass
class TEDep:
    iDep: int
    vecInputIdx: tuple[int]
    iDepIdx: int = -1


class Mode:
    def __init__(self, iSize: int = -1, bContract: bool = None, strName: str = None) -> None:
        self.iSize = iSize
        self.bContract = bContract
        self.strName = strName

        return

    def __eq__(self, mCheck) -> bool:
        return self.iSize == mCheck.iSize and self.bContract == mCheck.bContract and self.strName == mCheck.strName


class CouplingPattern:
    def __init__(self, vecModes: list[Mode], vecTypes: list[int]) -> None:
        self.iOrder: int = len(vecModes)
        self.vecModes = vecModes
        self.vecTypes: list[int] = vecTypes

        return
    
    def __eq__(self, cpCheck) -> bool:
        return ListEquals(self.vecTypes, cpCheck.vecTypes) and ListEquals(self.vecModes, cpCheck.vecModes)


class OperationSignature:
    def __init__(self, vecCouplingPatterns: list[CouplingPattern]) -> None:
        self.vecCouplingPatterns = vecCouplingPatterns
        
    def __eq__(self, osCheck) -> bool:
        return ListEquals(self.vecCouplingPatterns, osCheck.vecCouplingPatterns)


#-------Main Lvl 1 SM Class-------#

class TOMatrix():
    def __init__(self, tStruct: torch.tensor, tContract: torch.tensor = None, tShape: torch.tensor = None) -> None:
        '''
        Utility class for working with operation structure/shape/type matrices. 
        Supports both explicit and implicit contraction data. Assumes contractions are specified in the bottom row of tStruct unless tContract is set.
        '''

        assert len(tStruct.shape) == 2, "tStruct must be a matrix"

        if tContract is not None:
            assert len(tContract.shape) == 1, "tContract must be a vector"
            assert tContract.shape[0] == tStruct.shape[1], "Shape mis-match between tContract and tStruct"
            
            self.tStruct = torch.clone(tStruct)
            self.tContract = torch.clone(tContract)
        else:
            self.tStruct = torch.clone(tStruct[:-1, :])
            self.tContract = torch.clone(tStruct[-1, :])

        self.iA, self.iOC = self.tStruct.shape[0], self.tStruct.shape[1]
        
        #if shapes were provided at construction time, pull them in
        self.tShape = -1 * torch.clone(self.tStruct).float()
        if tShape is not None:

            if len(tShape.shape) == 2: tShape = torch.max(tShape, dim = 0)[0]
            
            assert tShape.shape[0] == self.iOC, "Shape mis-match between tContract and tShape"
            for i in range(tShape.shape[0]): self.SetOpenMode(i, tShape[i])
            
        return
    
    def __str__(self) -> str:
        strRet = "Shape:  " + str(torch.max(self.tShape, dim = 0)[0].numpy()) + "\n"
        strRet += "Struct: " + str(self.tStruct[0, :].float().numpy()) + "\n"
        for i in range(1, self.iA): strRet += "        " + str(self.tStruct[i, :].float().numpy()) + "\n"
        strRet += "        " + str(self.tContract.float().numpy()).replace("1.", "| ")
        
        return strRet
    
    def GetOperationString(self, vecInputIdx: list[int] = []) -> str:
        strRet = "Shape:  " + str(torch.max(self.tShape, dim = 0)[0].numpy()) + "\n"
        strRet += "Struct: " + str(self.tStruct[0, :].float().numpy())
        if 0 in vecInputIdx: strRet += " <--"
        strRet += "\n"
        for i in range(1, self.iA):
            strRet += "        " + str(self.tStruct[i, :].float().numpy()) 
            if i in vecInputIdx: strRet += " <--"        
            strRet += "\n"
        strRet += "Cntrct: " + str(self.tContract.float().numpy()).replace("1.", "| ")
        
        return strRet
    
    def ResetShape(self) -> None:
        self.tShape[self.tShape != 0] = -1
        return
    
    def IsSetup(self) -> bool:
        return torch.min(self.tShape) >= 0
    
    def GetSM(self, iBatchCols: int = 0) -> torch.tensor:
        tRet = torch.zeros((self.iA + 1, self.iOC), dtype = torch.int8)
        tRet[:self.iA, :] = self.tStruct[:, iBatchCols:]
        tRet[self.iA, :] = self.tContract[iBatchCols:]

        return tRet

    def Shape(self, iColIdx: int) -> int:
        tCol = self.tShape[:, iColIdx]
        return torch.max(tCol).item()
    
    def HasFreeModes(self, vecInputIdx: list[int]) -> bool:
        '''
        Checks for any modes with interactions that do NOT involve vecInputIdx
        '''
        for i in range(self.iOC):
            if not self.tContract[i]: continue #only consider contracted out modes
            tCol = self.tStruct[:, i]
            if torch.max(tCol[vecInputIdx]) == 0: return True

        return False
    
    def GetFreeModes(self, vecInputIdx: list[int]) -> list[int]:
        '''
        Returns all modes without interactions that involve vecInputIdx
        '''
        vecRet = []
        for i in range(self.iOC):
            if not self.tContract[i]: continue #only consider contracted out modes
            tCol = self.tStruct[:, i]
            if torch.max(tCol[vecInputIdx]) == 0: vecRet.append(i)

        return vecRet
    
    def GetArity(self,) -> list[int]:
        return self.tStruct.shape[0]
    
    def GetOrderComplexity(self,) -> list[int]:
        return self.tStruct.shape[1]
    
    def GetCouplingArity(self,) -> list[int]:
        return torch.max(torch.sum(self.tStruct, dim = 0)).item()
    
    def RemoveFreeModes(self, vecInputIdx: list[int]) -> None:
        vecCols = [idx for idx in range(self.iOC)]
        for i in range(self.iOC):
            if not self.tContract[i]: continue #only consider contracted out modes
            tCol = self.tStruct[:, i]
            if torch.max(tCol[vecInputIdx]) == 0: vecCols.remove(i)

        self.tStruct = self.tStruct[:, vecCols]
        self.tContract = self.tContract[vecCols]
        self.tShape = self.tShape[:, vecCols]
        self.iOC = len(vecCols)

        return
    
    def RemoveCols(self, vecColIdx: list[int]) -> None:
        vecIdx = [i for i in range(self.iOC) if i not in vecColIdx]
        self.tStruct = self.tStruct[:, vecIdx]
        self.tShape = self.tShape[:, vecIdx]
        self.tContract = self.tContract[vecIdx]
        self.iOC = len(vecIdx)

        return

    def RemoveEmptyCols(self) -> None:
        tIdx = torch.where(torch.max(self.tStruct, dim = 0)[0] > 0)[0]

        self.tStruct = self.tStruct[:, tIdx]
        self.tShape = self.tShape[:, tIdx]
        self.tContract = self.tContract[tIdx]
        self.iOC = tIdx.shape[0]

        return


    def RelativeFreeModes(self, vecInputIdx: list[int]) -> list[int]:
        '''
        Returns a list of free mode indices which do not belong to any of the rows in vecInputIdx
        '''
        vecRet = []
        for iColIdx in range(self.iOC):
            tCol = self.tStruct[:, iColIdx]
            if tCol[tCol > 0].shape[0] == 1 and torch.max(self.tStruct[vecInputIdx, iColIdx]) == 0: vecRet.append(iColIdx)

        return vecRet

    def FreeModes(self) -> list[int]:
        vecFM = []
        for i in range(self.iOC):
            tCol = self.tStruct[:, i]
            if tCol[tCol > 0].shape[0] == 1: vecFM.append(i)

        return vecFM
    
    def CoupledModes(self) -> list[int]:
        vecCM = []
        for i in range(self.iOC):
            tCol = self.tStruct[:, i]
            if tCol[tCol > 0].shape[0] > 1: vecCM.append(i)

        return vecCM
    
    def NumOpenModes(self) -> int:
        '''
        Returns the number of 'set-able' columns
        '''
        return torch.sum(torch.where(torch.max(self.tShape, dim = 0) <= 0, 1, 0))
    
    def GetShapeIntersection(self, vecS: list[int]) -> list[int]:
        vecU = copy.deepcopy(vecS)
        vecRet = []
        tSh = torch.max(self.tShape, dim = 0)[0]
        for i in range(self.iOC):
            if tSh[i] in vecU:
                vecRet.append(i)
                vecU.remove(tSh[i])

        return vecRet
    
    def GetOpenModes(self) -> list[int]:
        '''
        Returns a list of mode indices which do not have shapes assigned
        '''
        return list(torch.where(torch.max(self.tShape, dim = 0)[0] <= 0)[0])

    def UpdateRow(self, iRowIdx: int, tRow: torch.tensor) -> None:
        '''
        Expects a 0/1 valued row.
        '''
        self.tStruct[iRowIdx, :] = tRow
        self.tShape[iRowIdx, :] = -1 * tRow
        self.UpdateShape()

        return
    
    def AddOne(self, iRowIdx: int, iColIdx: int) -> None:
        '''
        Adds a 1 to tStruct/tShape
        '''
        self.tStruct[iRowIdx, iColIdx] = 1
        self.tShape[iRowIdx, iColIdx] = -1
        self.UpdateShape()

        return

    def OpenModes(self, bContracted: bool = False) -> list[int]:
        vecOM = []
        for i in range(self.iOC):
            if self.tContract[i] != bContracted: continue #skip modes of the wrong contraction status
            tCol = self.tShape[:, i]
            if tCol[tCol > 0].shape[0] == 0: vecOM.append(i)

        return vecOM
    
    def GetShape(self, iRowIdx: int = None, bRemoveZeros: bool = False) -> list[int]:
        self.UpdateShape()

        if iRowIdx is not None:
            tRow = self.tShape[iRowIdx, :]
            if bRemoveZeros: tRow = tRow[tRow > 0]
            return [s.int().item() for s in list(tRow)]
        
        tSh = torch.max(self.tShape, dim = 0)[0]
        if bRemoveZeros: tSh = tSh[tSh > 0]
        return tSh
    
    def GetOrder(self, iRowIdx: int) -> int:
        return torch.sum(self.tStruct[iRowIdx, :])
    
    def InputShape(self) -> list[int]:
        tS = torch.clone(self.tShape[0, :])
        return list(tS[tS > 0])
    
    def OutputShape(self) -> list[int]:
        vecOM = self.OpenModes(bContracted = False)
        vecShape = []
        for i in range(self.iOC):
            if self.tContract[i]: continue
            if i in vecOM:
                vecShape.append(-1)
                continue
            tCol = self.tShape[:, i]
            tCol = tCol[tCol > 0]

            vecShape.append(int(tCol[0]))

        return vecShape
    
     #structure/shape update functions
    def UpdateShape(self) -> None:

        self.tShape = torch.maximum(torch.max(self.tShape, dim = 0, keepdim = True)[0], torch.ones(self.iA, 1)) * self.tStruct
        self.tShape[self.tShape == 1] = -1

        return

    def SetOpenMode(self, iCol, iVal) -> None:
        for i in range(self.iA):
            if not self.tShape[i, iCol]: continue
            self.tShape[i, iCol] = iVal

        return
    
    def SetOpenModes(self, vecShapes: list[int]) -> None:
        assert len(vecShapes) == self.tShape.shape[1], "Expected shapes for {} modes, but got {}".format(self.tShape.shape[1], len(vecShapes))

        for i in range(len(vecShapes)): self.SetOpenMode(i, vecShapes[i])

        return
    
    def PropagateShape(self, vecSIn: list[list[int]], vecIdx: list[int], bDebug: bool = False) -> bool:
        if not isinstance(vecSIn[0], list):
            vecSIn = [vecSIn for _ in range(len(vecIdx))]

        for i in range(len(vecIdx)):
            idxIn = vecIdx[i]
            vecS = copy.deepcopy(vecSIn[i])
                
            vecMissing = copy.deepcopy(vecS)
            for j in range(self.iOC):
                if self.tShape[idxIn, j] in vecMissing: vecMissing.remove(self.tShape[idxIn, j])

            #copy any remaining sizes
            if len(vecMissing) == 0: continue

            idxM = torch.argwhere(self.tShape[idxIn, :] == -1)[:, 0]
            if idxM.shape[0] < len(vecMissing):
                if bDebug: 
                    print("Input shape {} is incompatible with row {} of structure matrix:\n{}".format(vecS, idxIn, self.tShape))
                    input()
                return False

            self.tShape[idxIn, idxM[:len(vecMissing)]] = torch.tensor(vecMissing).float()

            self.UpdateShape()

        return True
    
    def PermuteRows(self, vecPerm: list[int]) -> None:
        self.tStruct = self.tStruct[vecPerm, :]
        self.tShape = self.tShape[vecPerm, :]

        return
    
    def PermuteCols(self, vecPerm: list[int]) -> None:
        self.tStruct = self.tStruct[:, vecPerm]
        self.tShape = self.tShape[:, vecPerm]
        self.tContract = self.tContract[vecPerm]

        return
    
    def NumContracts(self) -> int: return torch.sum(self.tContract).item()

    def SetContracts(self, vecContracts: list[int]) -> None:
        self.tContract[vecContracts] = 1
        
        return
    
    def ClearContracts(self) -> None:
        self.tContract[:] = 0

        return
    
    def ParamCount(self, vecIdx: list[int]) -> int:
        iParams = 0
        for idx in vecIdx:
            tS = torch.clone(self.tShape[idx, :])
            tS[tS <= 0] = 1
            iParams += torch.prod(tS)

        return iParams
    
    def ComputeCost(self) -> dict:
        iMults = 0
        iMaxMult = 0
        iAdds = 0
        
        tS = torch.clone(self.tShape[0, :])
        tS[tS <= 0] = 1
        for i in range(1, self.iA):
            tNextS = torch.clone(self.tShape[i, :])
            tNextS[tNextS <= 0] = 1
            tS = torch.maximum(tS, tNextS)
            iM = torch.prod(tS)
            if iM > iMaxMult: iMaxMult = iM
            iMults += iM

            #measure the contractions
            for idx in range(self.iOC):
                if not self.tContract[idx]: continue
                if tS[idx] == 1: continue
                if (i == self.iA - 1) or torch.max(self.tShape[i + 1:, idx]) <= 1:
                    iAdds += (tS[idx] - 1) * (torch.prod(tS) / tS[idx])
                    tS[idx] = 1

        return {
            "Multiplies": iMults,
            "MaxSingleMult": iMaxMult,
            "Adds": iAdds,
        }
    



#-------Mode Map Class and Util Functions-------#

def FindUnfoldStride(iSzIn: int, vecSzOut: list[int]) -> tuple[int, int, int]:
    '''
    Assumes vecSzOut is of the form [#Out, k1, k2, ..., kn]
    '''

    #TODO: figure out how to extend this to > 2 case
    assert len(vecSzOut) == 2, "Unfoldings must contain exactly 2 output shapes"

    for i in range(len(vecSzOut) - 1):
        iMinSubsets = iSzIn - vecSzOut[i + 1] + 1
        iMaxSubsets = iSzIn + vecSzOut[i + 1] - 1

        #print(iMinSubsets, iMaxSubsets)

    #TODO: fix this, it is not quite right. Should work for the o2 case
        for n in range(iMinSubsets, iMaxSubsets + 1, 1):
            if (n - 1) // vecSzOut[i] == (n - 1) / vecSzOut[i] and (n - 1) // vecSzOut[i] > 0: return ((n - 1) // vecSzOut[i], math.ceil((n - iMinSubsets) // 2), (n - iMinSubsets + 1) % 2)

    return (-1, -1, -1)



class ModeMap(torch.nn.Module):
    def __init__(self, vecInputModes: list[Mode], vecOutputModes: list[Mode], tStruct: torch.tensor = None) -> None:
        '''
        Expects tStruct to map input modes to output modes from Cols-->Rows, if provided.
        '''

        super().__init__()

        if tStruct is not None: assert len(vecInputModes) == tStruct.shape[1] and len(vecOutputModes) == tStruct.shape[0], "Shape mis-match w/ I/O modes and tStruct"

        vecInputModes = [Mode(m) if isinstance(m, int) else m for m in vecInputModes]
        vecOutputModes = [Mode(m) if isinstance(m, int) else m for m in vecOutputModes]

        self.vecInputModes = vecInputModes
        self.vecSzIn = [m.iSize for m in self.vecInputModes]
        self.vecOutputModes = vecOutputModes
        self.vecSzOut = [m.iSize for m in self.vecOutputModes]

        self.vecCopies, self.vecUnfolds, self.vecFlattens = [], [], []
        #copies store (idx, idx), flattens store (list[idx], idx), unfolds store (idx, list[idx], stride, pad, pad_parity)

        if tStruct is None: self.tStruct = torch.zeros((len(vecOutputModes), len(vecInputModes))).bool()
        else: self.tStruct = tStruct.bool()
        
        self.bSetup = False
        self.Setup()

        if min([m.iSize for m in self.vecOutputModes]) < 0: self.ComputeOutputShapes()        

        return
    

    def Setup(self,) -> bool:
        '''
        Checks if output shapes are compatible with the specified semi-HG of modes.
        Computes and stores the 3 types of mode operations
        '''

        if torch.sum(self.tStruct) == 0: return False #vacuous/unitialized case

        self.vecCopies, self.vecUnfolds, self.vecFlattens = [], [], []

        bErr = False

        vecSkipCols = []

        for i in range(len(self.vecInputModes)):
            if i in vecSkipCols: continue

            tCol = self.tStruct[:, i]

            iColSum = torch.sum(tCol)
            if iColSum == 0: bErr = True #every input mode must map somewhere
            elif iColSum == 1:
                #check for copy/flatten
                idxR = torch.where(tCol == 1)[0][0]
                tRow = self.tStruct[idxR, :]
                iRowSum = torch.sum(tRow)

                if iRowSum == 0: bErr = True

                elif iRowSum == 1:
                    #check for valid copy
                    self.vecCopies.append([i, idxR])
                    bErr = bErr or (self.vecInputModes[i].iSize != self.vecOutputModes[idxR].iSize)

                else:
                    #check for valid flatten
                    vecFidx = list(torch.where(tRow == 1)[0])
                    bValid = True
                    for idx in vecFidx:
                        bValid = bValid and (torch.sum(self.tStruct[:, idx]) == 1)
                        vecSkipCols.append(idx)
                    bErr = bErr or not bValid
                    
                    self.vecFlattens.append([vecFidx, idxR])
                    
                    iSz = 1
                    for k in vecFidx: iSz *= self.vecInputModes[k].iSize
                    bErr = bErr or (self.vecOutputModes[idxR].iSize != iSz)

            else:
                #TODO: check for valid unfold
                tRidx = torch.where(tCol == 1)[0]
                bValid = True
                for k in range(tRidx.shape[0]):
                    bValid = bValid and (torch.sum(self.tStruct[tRidx[k], :]) == 1)
                bErr = bErr or not bValid
                
                vecRidx = list(tRidx)

                #check the size
                iStride, iPad, iPadParity = FindUnfoldStride(self.vecInputModes[i].iSize, [self.vecOutputModes[idx].iSize for idx in vecRidx])
                bErr = bErr or iStride < 0 or iPad < 0

                self.vecUnfolds.append([i, vecRidx, iStride, iPad, iPadParity])

        self.bSetup = not bErr

        return not bErr


    def ComputeOutputShapes(self,) -> bool:
        '''
        Computes copy/flatten shapes. Does not compute unfolds.
        '''

        for vecCopy in self.vecCopies:
            self.vecOutputModes[vecCopy[1]].iSize = self.vecInputModes[vecCopy[0]].iSize
            self.vecSzOut[vecCopy[1]] = self.vecInputModes[vecCopy[0]].iSize

        for vecFlat in self.vecFlattens:
            iSz = 1
            for idxIn in vecFlat[0]: iSz *= self.vecInputModes[idxIn].iSize
            self.vecOutputModes[vecFlat[1]].iSize = iSz
            self.vecSzOut[vecFlat[1]] = iSz

        return len(self.vecUnfolds) == 0
    

    def InputShape(self) -> list[int]:
        return self.vecSzIn
    
    def OutputShape(self) -> list[int]:
        return self.vecSzOut
    

    def AddCopy(self, idxIn: int, idxOut: int) -> bool:
        if torch.sum(self.tStruct[:, idxIn]) > 0:
            print("Tried to AddCopy() between input mode {} and output mode {}, but that input mode is already used.".format(idxIn, idxOut))
            return False
        
        if torch.sum(self.tStruct[idxOut, :]) > 0:
            print("Tried to AddCopy() between input mode {} and output mode {}, but that output mode is already used.".format(idxIn, idxOut))
            return False
        
        if self.vecSzIn[idxIn] != self.vecSzOut[idxOut] and self.vecSzOut[idxOut] != -1:
            print("Tried to AddCopy() between input mode {} and output mode {}, but their shapes are incompatible ({} vs. {}).".format(
                                                        idxIn, idxOut, self.vecSzIn[idxIn, self.vecSzOut[idxOut]]))
            return False

        self.tStruct[idxOut, idxIn] = 1
        self.vecCopies.append([idxIn, idxOut])

        if self.vecSzOut[idxOut] == -1:
            self.vecOutputModes[idxOut].iSize = self.vecSzIn[idxIn]
            self.vecSzOut[idxOut] = self.vecSzIn[idxIn]

        return True
    
    def AddFlatten(self, vecIdxIn: list[int], idxOut: int) -> bool:
        iSz = 1

        for idxIn in vecIdxIn:
            if torch.sum(self.tStruct[:, idxIn]) > 0:
                print("Tried to AddFlatten() between input modes {} and output mode {}, but input mode {} is already used.".format(vecIdxIn, idxOut, idxIn))
                return False
            iSz *= self.vecSzIn[idxIn]
        
        if torch.sum(self.tStruct[idxOut, :]) > 0:
            print("Tried to AddFlatten() between input modes {} and output mode {}, but that output mode is already used.".format(vecIdxIn, idxOut))
            return False
        
        if iSz != self.vecSzOut[idxOut] and self.vecSzOut[idxOut] != -1:
            print("Tried to AddFlatten() between input modes {} and output mode {}, but the shapes are incompatible ({} vs. {}).".format(vecIdxIn, idxOut,
                                                                                                                                         iSz, self.vecSzOut[idxOut]))
            return False

        self.tStruct[idxOut, vecIdxIn] = 1
        self.vecFlattens.append([vecIdxIn, idxOut])

        if self.vecSzOut[idxOut] == -1:
            self.vecOutputModes[idxOut].iSize = iSz
            self.vecSzOut[idxOut] = iSz

        return True
    
    def AddUnfold(self, idxIn: int, vecIdxOut: list[int], vecSz: list[int] = None) -> bool:
        if torch.sum(self.tStruct[:, idxIn]) > 0:
            print("Tried to AddUnfold() between input mode {} and output modes {}, but that input mode is already used.".format(idxIn, vecIdxOut))
            return False
        
        for idxOut in vecIdxOut:
            if torch.sum(self.tStruct[idxOut, :]) > 0:
                print("Tried to AddUnfold() between input mode {} and output modes {}, but output mode {} is already used.".format(idxIn, vecIdxOut, idxOut))
                return False
        
        if vecSz is not None: #pull in the specified sizes
            for i in range(len(vecSz)):
                self.vecSzOut[vecIdxOut[i]] = vecSz[i]
                self.vecOutputModes[vecIdxOut[i]].iSize = vecSz[i]

        iStride, iPad, iPadParity = FindUnfoldStride(self.vecSzIn[idxIn], [self.vecSzOut[idx] for idx in vecIdxOut])
        if iStride < 0 or iPad < 0:
            print("Tried to AddUnfold() between input mode {} and output modes {}, but their shapes are incompatible ({} vs. {}).".format(idxIn, vecIdxOut,
                                                                                                                                          self.vecSzIn[idxIn], [self.vecSzOut[idx] for idx in vecIdxOut]))
            return False

        self.tStruct[vecIdxOut, idxIn] = 1
        self.vecUnfolds.append([idxIn, vecIdxOut, iStride, iPad, iPadParity])

        return True
    
    def AddUnfoldShapes(self, idx: int, vecS: list[int]) -> bool:
        '''
        Tries to switch the outputs of the idx-th unfold to those specified in vecS
        '''
        vecUnfold = self.vecUnfolds[idx]
        if len(vecUnfold[1]) != len(vecS): return False

        iStride, iPad, iPadParity = FindUnfoldStride(self.vecInputModes[vecUnfold[0]].iSize, vecS)
        if iStride < 0 or iPad < 0 or iPadParity < 0: return False

        vecUnfold[2] = iStride
        vecUnfold[3] = iPad
        vecUnfold[4] = iPadParity
        for i in range(len(vecS)):
            self.vecSzOut[vecUnfold[1][i]] = vecS[i]
            self.vecOutputModes[vecUnfold[1][i]].iSize = vecS[i]

        return True



    def forward(self, x: torch.tensor) -> torch.tensor:
        '''
        Processes the input tensor according to the specified flattens/unfolds.
        Assumes all batch/ignored modes are placed at the front.
        '''
        vecSIn = list(x.shape)
        iBatchModes = len(vecSIn) - len(self.vecSzIn)
        assert iBatchModes >= 0, "ModeMap.forward() got input of order {}, but expects order at least {}".format(len(vecSIn), len(self.vecSzIn))

        vecPerm = ListPermuataion(vecSIn[iBatchModes:], self.vecSzIn)
        assert not not vecPerm, "ModeMap.forward() got input of shape {}, but expects sizes {}".format(vecSIn, self.vecSzIn)

        tSIn = torch.tensor(vecSIn)

        #process the flattenings
        for vecFlat in self.vecFlattens:
            vecFIdx = [vecPerm[i] + iBatchModes for i in vecFlat[0]]
            tSIn[vecFIdx[0]] = self.vecSzOut[vecFlat[1]]
            tSIn[vecFIdx[1:]] = 1
            x = x.reshape(list(tSIn))


        #process the unfoldings
        iO = tSIn.shape[0]
        for vecUnfold in self.vecUnfolds:
            #transpose, pad, transpose back
            mIdx = vecPerm[vecUnfold[0]] + iBatchModes - iO
            #print("Before pad:", x.shape)
            x = x.transpose(mIdx, -1)
            x = torch.nn.functional.pad(x, (vecUnfold[3], vecUnfold[3] - vecUnfold[4])) #pad only operates on the last mode :(
            x = x.transpose(mIdx, -1)

            x = x.unfold(vecPerm[vecUnfold[0]] + iBatchModes, self.vecSzOut[vecUnfold[1][1]], vecUnfold[2])

            iO = len(x.shape)

        vecSqzDims = [idx for idx in range(len(x.shape)) if idx >= iBatchModes and x.shape[idx] == 1]
        x = x.squeeze(dim = vecSqzDims)
        vecSOut = list(x.shape)
        vecOutPerm = ListPermuataion(vecSOut[iBatchModes:], self.vecSzOut)

        if not vecOutPerm: print("Error: ModeMap tried to reshape {} into {}".format(vecSOut[iBatchModes:], self.vecSzOut))

        return x.reshape(vecSOut[:iBatchModes] + [vecSOut[idx + iBatchModes] for idx in vecOutPerm])


#-------Main Lvl 2 SM Class-------#

class TEMatrix():
    def __init__(self, tStruct: torch.tensor) -> None:
        '''
        2-SM data-structure class.
        '''

        assert len(tStruct.shape) == 2, "tStruct must be a matrix"
        assert torch.min(torch.sum(tStruct, dim = 0)) > 0, "Warning! Found degenerate column in tStruct"
        assert torch.min(torch.sum(tStruct, dim = 1)) > 0, "Warning! Found degenerate row in tStruct"

        self.tStruct = tStruct.short()
        
        #Equations and Tensor-Complexity
        self.iE, self.iTC = self.tStruct.shape[0], self.tStruct.shape[1]

        #Other useful data
        self.vecA = list(torch.sum(torch.where(self.tStruct > 0, 1, 0), dim = 1))
        self.vecResultCols = []

        #setup some tensorial storage for dest/dep tracking
        self.vecDestinations = [[[] for _ in range(self.vecA[i - 1])] for i in range(self.iE)]
        self.vecDestinations[0] = [[]]
        self.vecDependencies = [[] for _ in range(self.iE)]

        self.Setup()
        
        return
    
    def IsSetup(self) -> bool: return self.bSetup

    def Reset(self) -> None:
        self.tStruct = self.tStruct.short()

        self.iE, self.iTC = self.tStruct.shape[0], self.tStruct.shape[1]

        #Other useful data
        self.vecA = list(torch.sum(torch.where(self.tStruct > 0, 1, 0), dim = 1))
        self.vecResultCols = []

        #setup some tensorial storage for dest/dep tracking
        self.vecDestinations = [[[] for _ in range(self.vecA[i - 1])] for i in range(self.iE)]
        self.vecDestinations[0] = [[]]
        self.vecDependencies = [[] for _ in range(self.iE)]

        return

    def Setup(self) -> bool:
        '''
        Checks for valid 2-SM structure.
        TODO: for now, assumes input multiplicity is always 1
        '''
        bSetup = True

        #input dest/dep
        self.vecDestinations[0][0] = list(torch.argwhere(self.tStruct[:, 0] >= 1)[:,0])
        self.vecDependencies[0].append(TEDep(-1, [0], iDepIdx = -1))

        vecOffsets = []
        for i in range(self.iE):
            tRow = self.tStruct[i, :]
            iResultIdx = torch.where(tRow >= 1)[0][-1]
            #disallow "multi-outputs"
            if self.tStruct[i, iResultIdx] != 1:
                print("Error: TEMatrix found row {} with multi-outputs".format(i))
                bSetup = False
            self.vecResultCols.append(iResultIdx)

            #disallow "pre-filled" result tensors
            if torch.sum(self.tStruct[:i, iResultIdx]) > 0:
                print("Error: TEMatrix found row {} with pre-filled result col {}".format(i, iResultIdx))
                bSetup = False

            #make sure the result is going somewhere
            if i < self.iE - 1:
                tIdx = torch.argwhere(self.tStruct[i, :] >= 1)[:, 0]
                for j in range(tIdx.shape[0]):
                    if i == 0 and j == 0: continue #skip the "special" input
                    idxCol = tIdx[j]
                    bIsResult = torch.sum(self.tStruct[:i, idxCol]) == 0
                    if not bIsResult: continue #only tensors originating from the current row can be passed onwards as deps.

                    tDests = i + 1 + torch.argwhere(self.tStruct[i+1:, idxCol] >= 1)[:, 0]
                    self.vecDestinations[i + 1][j] = list(tDests)

                #check that destinations exist for the "true" result
                if torch.sum(self.tStruct[i+1:, idxCol]) == 0:
                    print("Error: TEMatrix found row {} with no destinations for result".format(i))
                    bSetup = False

            #check for param. only rows
            bFound = False
            tIdx = torch.argwhere(tRow >= 1)[:,0]
            for j in range(tIdx.shape[0]):
                idx = tIdx[j]
                if idx == 0 or idx in vecOffsets: bFound = True
            if not bFound:
                print("Error: TEMatrix found row {} which does not have a valid IO structure".format(i))
                bSetup = False
            vecOffsets.append(tIdx[-1])

            #compute the deps
            iOffset = 0
            for j in range(self.iTC):
                if not tRow[j]:
                    iOffset += 1
                    continue
                for k in range(i):
                    if self.tStruct[k, j]:
                        iDepIdx = torch.argwhere(torch.argwhere(self.tStruct[k, :] >= 1)[:, 0] == j)[0, 0].item()
                        if k == 0 and iDepIdx == 0:
                            self.vecDependencies[i].append(TEDep(-1, [j - iOffset], iDepIdx = -1))
                        else: self.vecDependencies[i].append(TEDep(k, [j - iOffset], iDepIdx = iDepIdx))
                        break


        self.bSetup = bSetup
        return bSetup
    

    def IncreaseArity(self, idxRow: int) -> bool:
        idxCol = self.vecResultCols[idxRow]
        t1 = self.tStruct[:, :idxCol]
        t2 = torch.zeros((self.iE, 1))
        t2[idxRow, 0] = 1
        t3 = self.tStruct[:, idxCol:]

        self.tStruct = torch.cat((t1, t2, t3), dim = 1)
        
        self.Reset()
        return self.Setup()







if __name__ == "__main__":

    def TestFlatten():
        vecInModes = [Mode(3), Mode(8), Mode(8), Mode(5), Mode(5), Mode(7), Mode(5), Mode(4)]
        vecOutModes = [Mode(3), Mode(40), Mode(56), Mode(100)]

        tStruct = torch.tensor([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 1],
        ])

        mmCheck = ModeMap(vecInModes, vecOutModes, tStruct)

        print(mmCheck.bSetup)

        print(mmCheck.vecFlattens)

        X = torch.randn((128, 3, 8, 5, 5, 4, 5, 7, 8))
        Y = mmCheck(X)

        print(Y.shape)

    def TestUnfold():
        vecInModes = [Mode(4), Mode(6)]
        vecOutModes = [Mode(4), Mode(6), Mode(8)]

        mmCheck = ModeMap(vecInModes, vecOutModes)
        mmCheck.AddUnfold(1, [1, 2])

        print(mmCheck.bSetup)

        print(mmCheck.vecUnfolds)

        X = torch.randn((128, 4, 6))
        Y = mmCheck(X)

        print(Y.shape)


    def TestBoth():
        vecInModes = [Mode(3), Mode(16), Mode(16), Mode(8), Mode(4), Mode(8)]
        vecOutModes = [Mode(3), Mode(16), Mode(16), Mode(5), Mode(5), Mode(32), Mode(6), Mode(3)]

        tStruct = torch.tensor([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1],
        ])

        mmCheck = ModeMap(vecInModes, vecOutModes, tStruct)

        print(mmCheck.bSetup)

        print(mmCheck.vecFlattens, mmCheck.vecUnfolds)

        X = torch.randn((128, 16, 3, 16, 4, 8, 8))
        Y = mmCheck(X)

        print(Y.shape)

    #TestUnfold()
    #TestBoth()