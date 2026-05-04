'''
Code written by Nicholas J. Cooper.
Released under the MIT license, see the GitHub page for the full legal boilerplate.
tldr: you freely can do whatever you like with this code so long as this message is retained and you cite the GitHub: https://github.com/Thegolfingocto/ArchitecturalComplexity
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
'''

import torch
import numpy as np
import json
import copy
import numbers
import os
from pathlib import Path
import time
import sys
import random

def GetInput(strPrompt: str) -> bool:
    chC = input(strPrompt)
    return chC == "Y"

class SupressPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

        return

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

        return

def DictFlatten(d: dict, ret: dict = None) -> dict:
    if ret is None: ret = {}
    for k in d.keys():
        if isinstance(d[k], dict): ret = DictFlatten(d[k], ret)
        elif k not in ret.keys(): ret[k] = d[k]
    return ret

def DictFilterByKey(strFilter: str, d: dict, ret: dict = None, bRecursive: bool = False) -> dict:
    '''
    Returns a subdict consisting of KVPs where strFilter is a substring of K
    '''
    if ret is None: ret = {}
    vecKeys = list(d.keys())
    if bRecursive:
        for k in vecKeys:
            if strFilter in k:
                ret[k] = d[k]
                continue
            if isinstance(ret[k], dict):
                ret[k] = DictFilterByKey(strFilter, ret[k], ret, True)
    else:
        for k in vecKeys:
            if strFilter in k:
                ret[k] = d[k]

    return ret

def IsDictSubset(d1: dict, d2: dict) -> bool:
    '''
    Checks if d1 is a sub-dict of d2
    '''
    for key in d1.keys():
        if key not in d2.keys():
            return False
        if d1[key] != d2[key]:
            return False
        
    return True

def DictEquals(d1: dict, d2: dict) -> bool:
    return IsDictSubset(d1, d2) and IsDictSubset(d2, d1)

def PrintDictDiff(d1: dict, d2: dict) -> None:
    vecNewKeys = []
    vecDelKeys = []
    vecModKeys = []

    for key in d1.keys():
        if key not in d2.keys():
            vecDelKeys.append(key)
        elif d1[key] != d2[key]:
            vecModKeys.append(key)

    for key in d2.keys():
        if key not in d1.keys():
            vecNewKeys.append(key)

    print("Dict Diff:")
    print("-------------------------------------")
    print("New KVPs:")
    for nk in vecNewKeys: print("{}: {}".format(nk, d2[nk]))
    print("-------------------------------------")
    print("Missing KVPs:")
    for dk in vecDelKeys: print("{}: {}".format(dk, d1[dk]))
    print("-------------------------------------")
    print("Modified KVPs:")
    for mk in vecModKeys: print("{}: {} -> {}".format(mk, d1[mk], d2[mk]))
    print("-------------------------------------")

    return

def ReadFile(strPath: str) -> object:
    '''
    Generic multi-extension file opener
    '''
    if ".json" in strPath:
        with open(strPath, "r") as f:
            return json.load(f)
    elif ".pkl" in strPath:
        with open(strPath, "rb") as f:
            return torch.load(f)

    print("ReadFile() Found Unsupported Format: {}".format(strPath.split(".")[-1]))
    return None

def WriteFile(oData: object, strPath: str) -> None:
    '''
    Generic multi-extension file writer
    '''
    if ".json" in strPath:
        assert type(oData) == dict or type(oData) == list, "Invalid object of type {} passed for .json writing!".format(type(oData))
        with open(strPath, "w") as f:
            json.dump(oData, f, indent = 2) #pretty formatting
    elif ".pkl" in strPath:
        assert type(oData) == torch.tensor, "Invalid object of type {} passed for .pkl writing!".format(type(oData))
        with open(strPath, "wb") as f:
            torch.save(oData, f)
    else:
        print("WriteFile() Found Unsupported Format: {}".format(strPath.split(".")[-1]))
    return None

def GetRandomSubset(Y: torch.tensor, iN: int) -> torch.tensor:
    if len(Y.shape) > 1:
        iC = Y.shape[1]
        YA = torch.argmax(Y, dim = 1)
    else:
        iC = torch.max(Y) + 1
        YA = Y

    n = iN // iC
    vecIdx = [torch.where(YA == i)[0] for i in range(iC)]
    rng = np.random.default_rng()
    
    idx = torch.zeros((0), dtype=torch.int32)
    for i in range(iC):
        Idx = rng.permutation(vecIdx[i].shape[0])[:n]
        idx = torch.cat((idx, vecIdx[i][Idx]), dim = 0)
    
    return idx

def SplitLabelsByClass(Y: torch.Tensor) ->list[torch.Tensor]:
    c = Y.shape[1]
    Y = torch.argmax(Y, dim = 1).to("cpu")
    vecIdx = [torch.where(Y == i)[0] for i in range(c)]
    return vecIdx

def ListEquals(vecX: list, vecY: list) -> bool:
    if len(vecX) != len(vecY): return False
    vecT = copy.deepcopy(vecY)
    for i in range(len(vecX)):
        if vecX[i] not in vecT: return False
        vecT.remove(vecX[i])

    return True

def ListEqualsRecursive(vecX: list, vecY: list) -> bool:
    if len(vecX) != len(vecY): return False
    vecT = copy.deepcopy(vecY)
    for i in range(len(vecX)):
        if isinstance(vecX[i], list):
            bMatch = False
            for j in range(len(vecT)):
                if ListEqualsRecursive(vecX[i], vecT[j]):
                    bMatch = True
                    break
            if not bMatch: return False
            vecT.pop(j)
        else:
            if vecX[i] not in vecT: return False
            vecT.remove(vecX[i])

    return True

def ListPermuataion(vecX: list, vecY: list) -> list[int]:
    '''
    Returns a permutation which describes how to map vecX bijectively to vecY
    '''
    #assert ListEquals(vecX, vecY), "ListPermutation requires inputs to be equal up to a bijection. X: {}, Y: {}".format(vecX, vecY)
    if not ListEquals(vecX, vecY): return False
    vecT = copy.deepcopy(vecX)
    vecPerm = []
    for i in range(len(vecY)):
        idxT = vecT.index(vecY[i])
        vecPerm.append(idxT)
        vecT[idxT] = None

    return vecPerm

def ListIntersection(vecX: list, vecY: list) -> list:
    '''
    Returns the intersection of two lists, with the sub-ordering from vecX.
    '''
    return [x for x in vecX if x in vecY]

def ListMinus(vecX: list, vecY: list) -> list:
    '''
    Removes values from vecX based on values from vecY
    '''    
    vecRet = []
    vecY = copy.deepcopy(vecY)

    for i in range(len(vecX)):
        if vecX[i] in vecY: vecY.remove(vecX[i])
        else: vecRet.append(vecX[i])
    
    return vecRet


def ListAvgDict(vecResults: list[dict]) -> dict:
    '''
    Assumes a list of flat dictionaries. Returns a dict containing avg/median/std for each numerical-valued key 
    '''
    vecTrackedKeys = []
    dStats = {}
    for strKey in vecResults[0].keys():
        if isinstance(vecResults[0][strKey], numbers.Number):
            vecTrackedKeys.append(strKey)
            dStats[strKey] = []

    for i in range(len(vecResults)):
        dR = vecResults[i]
        for strKey in vecTrackedKeys:
            dStats[strKey].append(dR[strKey])

    dRet = {}
    for strKey in vecTrackedKeys:
        nKey = np.array(dStats[strKey])
        dRet["Avg_" + strKey] = np.mean(nKey)
        dRet["Std_" + strKey] = np.std(nKey)
        dRet["Median_" + strKey] = np.median(nKey)
        dRet["Min_" + strKey] = np.min(nKey)
        dRet["Max_" + strKey] = np.max(nKey)

    return dRet

def DisplayDictFilter(dData: dict, strKeyFilter: str = "") -> None:
    for strKey in dData.keys():
        if strKeyFilter in strKey: print(strKey, ": ", dData[strKey])

    return


class LockedJSONFile():
    def __init__(self, strPath: str, fLockTimer: float = 0.001, indent = None) -> None:
        '''
        Simple concurrency-safe JSON file for use w/ parallel processes
        '''
        self.strPath = strPath
        if "json" not in self.strPath: self.strPath += ".json" #"json" is used as the substring query to support .cjson files as well
        self.strLockPath = strPath.replace(strPath.split(".")[-1], "lck")
        self.pathLock = Path(self.strLockPath)
        if not os.path.exists(self.strLockPath): self.ReturnLock() #TODO: this might get funky w/ a lot of processes

        self.fLockTimer = fLockTimer
        self.indent = indent

        return
    
    def GetLock(self,) -> None:
        bGot = False
        while not bGot:
            while not os.path.exists(self.strLockPath): time.sleep(self.fLockTimer)
            
            #if another process snuck in while we were exiting the inner while loop, wait and retry
            try:
                os.remove(self.strLockPath)
                bGot = True
            except: continue

        return
    
    def ReturnLock(self,) -> None:
        self.pathLock.touch()
        
        return

    def Read(self,) -> dict:
        self.GetLock()
        with open(self.strPath, "r") as f:
            jRet = json.load(f)
        self.ReturnLock()

        return jRet
    
    def Write(self, jData: dict | list) -> None:
        self.GetLock()
        with open(self.strPath, "w") as f:
            json.dump(jData, f, indent = self.indent)
        self.ReturnLock()

        return
    
    def MergeWrite(self, jData: dict | list) -> dict | list:
        '''
        Merges existing dicts/lists into jData before writing to disk.
        Assumes the data stored on disk is of the same type as jData.
        Returns the merged version of jData.
        '''
        self.GetLock()
        if os.path.exists(self.strPath):
            with open(self.strPath, "r") as f:
                if isinstance(jData, dict): jData = jData | json.load(f)
                elif isinstance(jData, list): jData = jData + json.load(f)
        with open(self.strPath, "w") as f:
            json.dump(jData, f, indent = self.indent)
        self.ReturnLock()

        return jData

def TestDictUtils():
     #Dict util tests
    dTest = {
    "Dataset": "CIFAR10",
    "Dataset:CIFAR100S:Params": {"Subset": 7},

    "Model": "ResNet9",
    "Model:Params": {
        "PreTrained": True,
        "PreTrained:Params": {"BackboneLRMultiplier": -1},
        "BatchNorm": True
    },
    "Model:ViT,TcT:Params": {"AvgTokens": True},

    "CustomReLU": False,
    "CustomReLU:Params": {"Kn": 0.236, "Kp": 1.0},

    "FeatureKD": True,
    "FeatureKD:Params": "...",

    "LayerMappingMethod": "One2One",
    "LayerMappingMethod:PreDefined:Params": {"LayerMap": "...", "LayerMapWeights": "..."},

    ":": ["vecIgnoredFields"]
    }

    print(DictFlatten(dTest))
    print(DictFilterByKey(":", dTest))

def TestListUtils():
    X = [[3, 32], [32, 32]]
    Y = [[32, 32], [3, 32]]

    print(ListEqualsRecursive(X, Y))


def TestClasses():
    from threading import Thread
    strPath = "./test.json"

    lfThing = LockedJSONFile(strPath, fLockTimer = 0.01)

    def WriteSomething():
        d = {"1": 2}

        for _ in range(100):
            #with open(strPath, "w") as f: json.dump(d, f)
            #lfThing.Write(d)
            d = lfThing.MergeWrite(d)
            time.sleep(0.01 * random.random())

        return
    
    def WriteSomethingDifferent():
        d = {"3": 4}

        for _ in range(100):
            #with open(strPath, "w") as f: json.dump(d, f)
            #lfThing.Write(d)
            d = lfThing.MergeWrite(d)
            print(d)
            time.sleep(0.01 * random.random())

        return
    
    def ReadSomething():
        for _ in range(100):
            #with open(strPath, "r") as f: print(json.load(f))
            #print(lfThing.Read())
            lfThing.Read()
            time.sleep(0.01 * random.random())

        return
    
    t1 = Thread(target = WriteSomething)
    t2 = Thread(target = WriteSomethingDifferent)
    t3 = Thread(target = ReadSomething)

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    return


if __name__ == "__main__":
    #TestDictUtils()
    #TestListUtils()
    #TestClasses()

    d1 = {'Model': 'ResNet34', 'BatchNorm': True, 'DisableInputLayers': True, 'PreTrained': False, 'RandomInit': False, 'ActivationFunction': 'LeakyReLU', 'Dataset': 'CIFAR100F', 'Normalization': 'MeanVar', 'DownSample': -1, 'SubSample': 1.0, 'DataAugmentation': True, 'RandAug': False, 'DataAugOnTest': False, 'GaussianNoiseAug': False, 'Optimizer': 'AdamW', 'LearningRate': 0.0075, 'WeightDecay': -1, 'LRScheduler': 'OneCycle', 'PctStart': 0.3, 'BatchSize': 128, 'NumEpochs': 50, 'NumRuns': 1}
    
    d2 = {'Model': 'ResNet34', 'Dataset': 'CIFAR100F', 'DataAugmentation': True, 'LRScheduler': 'OneCycle', 'BatchSize': 128, 'SubSample': 1.0, 'NumEpochs': 50, 'BatchNorm': True, 'PreTrained': False, 'RandomInit': False, 'DisableInputLayers': False, 'ActivationFunction': 'LeakyReLU', 'Normalization': 'MeanVar', 'DownSample': -1, 'DataAugOnTest': False, 'RandAug': False, 'GaussianNoiseAug': False, 'Optimizer': 'AdamW', 'LearningRate': 0.0075, 'WeightDecay': -1, 'PctStart': 0.3, 'NumRuns': 1}
    PrintDictDiff(d1, d2)