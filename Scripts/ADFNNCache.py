'''
Code written by Nicholas J. Cooper.
Released under the MIT license, see the GitHub page for the full legal boilerplate.
tldr: you freely can do whatever you like with this code so long as this message is retained and you cite the GitHub: https://github.com/Thegolfingocto/ArchitecturalComplexity
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
'''



import torch

from ADFNN import *
from Arch.ICache import ICache

import os
import hashlib
from typing import Union
from threading import Thread, Lock



class SMCache:
    def __init__(self, strBaseDir: str, strName: str, bForceCreate: bool = False,
                 iRefreshTimer: int = 2, bUseLockedFiles: bool = True, bUseDiskWatchers: bool = False) -> None:
        '''
        Simple structure matrix cache. Saves 1-SMs and 2-SMs to a disk hashmap.
        Set bUseDiskWatchers to True if you want to do things with parallel processes.
        '''

        self.strBaseDir = strBaseDir
        if not os.path.isdir(strBaseDir): os.mkdir(strBaseDir)
        if self.strBaseDir[-1] != "/": self.strBaseDir += "/"
        if ".json" in strName: strName = strName.replace(".json", "") #ensure the name is free of the file extension for cross-machine merging later
        self.strName = strName
        self.strCachePath = self.strBaseDir + self.strName + ".json"
        self.strUser = os.uname().nodename
        self.strBAKName = self.strName  + "_" + self.strUser 
        self.strBAKCachePath = self.strBaseDir + self.strBAKName + ".json"

        #setup the locked file I/O managers
        self.lfCache = LockedJSONFile(self.strCachePath, indent = 2)
        self.lfBAKCache = LockedJSONFile(self.strBAKCachePath, indent = 2)

        self.dCache = {}
        #read in the saved SMs
        if os.path.exists(self.strCachePath):
            if bForceCreate:
                print("Warning! SMCache found existing cache {} and ForceCreate is True")
                if not GetInput("Confirm Overwrite (Y/X)"): return
                os.remove(self.strCachePath)
            else:
                # with open(self.strCachePath, "r") as f:
                #     self.dCache = json.load(f)
                self.dCache = self.lfCache.Read()
        else:
            self.dCache = {}
            self.lfCache.Write({})

        self.MergeCaches()

        self.mtxLock = Lock()

        self.bUseLockedFiles = bUseLockedFiles
        self.bUseDiskWatchers = bUseDiskWatchers
        if self.bUseDiskWatchers:
            #launch the disk watcher
            self.iRefreshTimer = iRefreshTimer
            self.bRun = True
            self.thrdRefresh = Thread(target = self.DiskWatcher, name = "Thread-DiskWatcher")
            self.thrdRefresh.start()

        return
    
    def __del__(self):
        if self.bUseDiskWatchers:
            self.RefreshCache()
            self.MergeCaches()
            self.WriteCache()

        return
    
    def StopWatcher(self,) -> None:
        with self.mtxLock: self.bRun = False
        if self.bUseDiskWatchers: self.thrdRefresh.join()

        return

    def DiskWatcher(self,) -> None:
        while 1:
            time.sleep(self.iRefreshTimer)
            with self.mtxLock:
                if not self.bRun: break
                self.RefreshCache()
                self.MergeCaches()
                self.WriteCache()
            #print(f"[SMCache {self.strCachePath}] DiskWatcher updated triggered")

        print(f"[SMCache {self.strCachePath}] DiskWatcher exiting...")

        return
    
    def Size(self) -> int: return len(self.dCache.keys())
    
    def GetSM(self, strID: Union[str, int]) -> torch.tensor:
        tRet = None
        with self.mtxLock:
            if isinstance(strID, int): tRet = torch.tensor(self.dCache[list(self.dCache.keys())[strID]])
            elif strID in self.dCache.keys(): tRet = torch.tensor(self.dCache[strID])

        return tRet
    
    def MergeCaches(self,) -> None:
        #merge in any backups from other machines
        for f in os.listdir(self.strBaseDir):
            strF = os.fsdecode(f)
            if self.strName in strF and ".json" in strF and strF != self.strName + ".json" and strF != self.strBAKName + ".json":
                with open(self.strBaseDir + strF, "r") as f:
                    self.dCache = self.dCache | json.load(f)

        return

    def RefreshCache(self,) -> None:
        if not os.path.exists(self.strCachePath): return

        
        if self.bUseLockedFiles:
            self.dCache = self.dCache | self.lfCache.Read()
            if os.path.exists(self.strBAKCachePath): self.dCache = self.dCache | self.lfBAKCache.Read()
        else:
            with open(self.strCachePath, "r") as f:
                self.dCache = self.dCache | json.load(f)
            if os.path.exists(self.strBAKCachePath): 
                with open(self.strBAKCachePath, "r") as f:
                    self.dCache = self.dCache | json.load(f)

        return
    
    def WriteCache(self,) -> None:
        
        #TODO: decide if we want to switch to MergeWrite() and refactor out some of the merging within this class
        if self.bUseLockedFiles:
            self.lfCache.Write(self.dCache)
            self.lfBAKCache.Write(self.dCache)
        else:
            with open(self.strCachePath, "w") as f:
                json.dump(self.dCache, f, indent = 2)
            with open(self.strBAKCachePath, "w") as f:
                json.dump(self.dCache, f, indent = 2)

        # with open(self.strBAKCachePath, "w") as f:
        #     json.dump(self.dCache, f, indent = 2)
        
        return

    def CacheSM(self, tSM: torch.tensor) -> str:
        # print(f"Caching SM to: {self.strCachePath}")
        # input()
        if len(tSM.shape) != 2:
            print("Error: CacheSM() got a non-matrix tensor")
            return None

        vecSM = tSM.numpy().tolist()
        strHash = hashlib.md5(json.dumps(vecSM).encode('utf-8')).hexdigest()

        with self.mtxLock:
            if strHash not in self.dCache.keys():
                self.dCache[strHash] = vecSM
                #merge in anything new from the disk
                self.RefreshCache()
                self.MergeCaches()
                #store the new result
                self.WriteCache()

        return strHash
    


def GetHashableTOMData(opData: TOMOpData) -> dict:
    '''
    Packs TOMOpData into a disk-savable dict
    Inverse operation of GetUsableTOMData()
    '''
    if opData is None: return None

    dRet = {}
    dRet["RowSpaceOp"] = opData.strRowSpaceOp
    dRet["ContractOp"] = opData.strContractOp
    dRet["UnaryOps"] = []
    for op in opData.vecUnaryOps:
        
        if isinstance(op.tOp, torch.nn.Module): strName = op.tOp.__class__.__name__
        else: strName = op.tOp.__name__
        print(op)
        #TODO: see if this logic can be cleaned up
        if strName == "LayerNorm": sOp = op.tOp.normalized_shape
        elif strName == "Softmax": sOp = op.tOp.dim
        elif op.iDim is not None: sOp = op.iDim
        else: sOp = None

        dRet["UnaryOps"].append((strName, sOp))
    print(dRet)
    return dRet

def GetUsableTOMData(dOpData: dict) -> TOMOpData:
    '''
    Unpacks dict form TOMData into an actual runnable TOMOpData class
    Inverse operation of GetHashableTOMData()
    '''
    if dOpData is None: return None

    vecUnaryOps = []
    for vecOp in dOpData["UnaryOps"]:
        #TODO: see if we can avoid this ugly LUT
        if vecOp[0] == "leaky_relu": vecUnaryOps.append(UnaryOperation(torch.nn.functional.leaky_relu, iDim = None))
        elif vecOp[0] == "relu6": vecUnaryOps.append(UnaryOperation(torch.nn.functional.relu6, iDim = None))
        elif vecOp[0] == "LayerNorm": vecUnaryOps.append(UnaryOperation(torch.nn.LayerNorm(normalized_shape = vecOp[1]), iDim = None))
        elif vecOp[0] == "Softmax": vecUnaryOps.append(UnaryOperation(torch.nn.Softmax(vecOp[1]), iDim = None))
        elif vecOp[0] == "gumbel_softmax": vecUnaryOps.append(UnaryOperation(torch.nn.functional.gumbel_softmax, iDim = vecOp[1]))
        else:
            print("Warning: GetUsableTOMData() got unrecognized unary operation {}".format(vecOp[0]))

    return TOMOpData(strRowSpaceOp = dOpData["RowSpaceOp"], strContractOp = dOpData["ContractOp"], vecUnaryOps = vecUnaryOps)


class ADFNNCache(ICache):
    def __init__(self, strCacheDir: str, dConfig: dict = {}, bUseDiskWatchers: bool = False) -> None:
        super().__init__(strCacheDir, dConfig)
        self.UpdateCacheMap()

        self.TEMCache = SMCache(self.strBaseDir, "TEMCache", bUseDiskWatchers = bUseDiskWatchers)
        self.TOMCache = SMCache(self.strBaseDir, "TOMCache", bUseDiskWatchers = bUseDiskWatchers)

        self.LoadNameMap()

        return
    
    def Stop(self,) -> None:
        self.TEMCache.StopWatcher()
        self.TOMCache.StopWatcher()

        return
    
    def PrintADFNN(self, strID: str) -> None:
        if strID not in self.CacheMap.keys(): return

        dCfg = self.CacheMap[strID]
        print("TEM:")
        print(self.TEMCache.GetSM(dCfg["TEMID"]))
        print("---------------------------------")
        for i in range(len(dCfg["TOMIDs"])):
            strTOMID = dCfg["TOMIDs"][i]
            print("TOM {}".format(i))
            print(TOMatrix(self.TOMCache.GetSM(strTOMID), tShape = torch.tensor(dCfg["TOMShapes"][i])).GetOperationString())
            print(dCfg["TOMData"][i])

        return


    def LoadNameMap(self) -> None:
        strNMPath = self.strBaseDir + "NameMap.json"
        if os.path.exists(strNMPath):
            with open(strNMPath, "r") as f:
                self.mapNamedADFNNs = json.load(f)
        else: self.mapNamedADFNNs = {}

    def SetName(self, strID: str, strName: str) -> None:
        if strID not in self.CacheMap.keys(): return
        self.mapNamedADFNNs[strName] = strID
        with open(self.strBaseDir + "NameMap.json", "w") as f:
                json.dump(self.mapNamedADFNNs, f, indent = 2)
        return

    def NumTEMs(self) -> int: return self.TEMCache.Size()
    def NumTOMs(self) -> int: return self.TOMCache.Size()
    
    def GetTEM(self, strID: str): return self.TEMCache.GetSM(strID)
    def GetTOM(self, strID: str): return self.TOMCache.GetSM(strID)

    def GetTEMID(self, strModelID: str) -> str:
        if strModelID not in self.CacheMap.keys(): return None
        return self.CacheMap[strModelID]["TEMID"]


    def GetADFNN(self, strHash: Union[str, dict]) -> ADFNN:
        if isinstance(strHash, dict): strHash = self.IGenHash(strHash)

        if strHash not in self.CacheMap.keys():
            if strHash not in self.mapNamedADFNNs.keys(): return None
            else: strHash = self.mapNamedADFNNs[strHash]

        dCfg = self.CacheMap[strHash]

        #setup the TEM
        temArch = TEMatrix(self.TEMCache.GetSM(dCfg["TEMID"]))

        vecTOMs = []
        vecTOMData = []
        for i in range(len(dCfg["TOMIDs"])):
            vecTOMs.append(
                TOMatrix(tStruct = self.TOMCache.GetSM(dCfg["TOMIDs"][i]), tShape = torch.tensor(dCfg["TOMShapes"][i]))
                )
            vecTOMData.append(GetUsableTOMData(dCfg["TOMData"][i]))

        adfnnModel = ADFNN(sIn = dCfg["InputShape"], vecSOut = dCfg["OutputShapes"], temArch = temArch, vecTOMs = vecTOMs, vecRowData = vecTOMData)
        adfnnModel.Setup()

        return adfnnModel
    
    def ConvertADFNNToDict(self, adfnnModel: ADFNN) -> dict:
        dCfg = {}

        dCfg["InputShape"] = adfnnModel.sIn
        dCfg["OutputShapes"] = adfnnModel.vecSOut

        strTEMID = self.TEMCache.CacheSM(adfnnModel.GetTEM())
        dCfg["TEMID"] = strTEMID

        vecTOMIDs = []
        vecTOMShapes = []
        vecTOMData = []
        for i in range(len(adfnnModel.vecTOMs)):
            toLayer = adfnnModel.vecTOLayers[i]
            vecTOMIDs.append(self.TOMCache.CacheSM(toLayer.GetSM()))
            vecTOMShapes.append([s.int().item() for s in list(toLayer.GetShape())])
            vecTOMData.append(GetHashableTOMData(adfnnModel.vecRowData[i]))

        dCfg["TOMIDs"] = vecTOMIDs
        dCfg["TOMShapes"] = vecTOMShapes
        dCfg["TOMData"] = vecTOMData

        # print("DEBUG:")
        # print(dCfg)

        return dCfg
        
    def CheckCache(self, adfnnModel: ADFNN) -> bool:
        '''
        Returns True/False if the model is already present in the cache
        '''
        if not adfnnModel.IsSetup():
            print("Error: CacheADFNN() got an unitialized model")
            return None

        dCfg = self.ConvertADFNNToDict(adfnnModel)

        return self.IGenHash(dCfg) in self.CacheMap.keys()

    def CacheADFNN(self, adfnnModel: ADFNN) -> str:
        '''
        Checks if the cache has the model stored, creates a new entry if not, and returns the corresponding hash ID.
        '''
        if not adfnnModel.IsSetup():
            print("Error: CacheADFNN() got an unitialized model")
            return None

        dCfg = self.ConvertADFNNToDict(adfnnModel)

        return self.UpdateCacheMap(dCfg)




if __name__ == "__main__":
    from ADFNNSamplers import *
    
    def TestADFNNCache():
        sIn = [3, 32, 32]

        vecSOut = [
            [4, 4, 32, 32],
            [4, 4, 32, 32],

        ]

        iMaxOrderVariance = 0

        while 1:
            os.system('clear')
            adfnnModel = SampleADFNN(sIn, vecSOut, iMaxOrderVariance = iMaxOrderVariance, bEnableFlattens = False, iMaxSC = 1, iMaxArity = 4, iMaxOC = 12, bDebug = True)
            if adfnnModel is not None: break

        dOrig = TrainADFNN(torch.nn.Sequential(adfnnModel, torch.nn.Flatten(), torch.nn.Linear(adfnnModel.GetAD(-1), 10)), "CIFAR10")
        
        cacheModels = ADFNNCache("../Data/ADFNNCache/")
        strHash = cacheModels.CacheADFNN(adfnnModel)
        adfnnModel = cacheModels.GetADFNN(strHash)

        dCached = TrainADFNN(torch.nn.Sequential(adfnnModel, torch.nn.Flatten(), torch.nn.Linear(adfnnModel.GetAD(-1), 10)), "CIFAR10")

        print("Pre-cache results:")
        print(dOrig)
        print("-"*42)
        print("Post-cache results:")
        print(dCached)


    TestADFNNCache()