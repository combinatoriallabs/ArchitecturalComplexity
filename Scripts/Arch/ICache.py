'''
Code written by Nicholas J. Cooper.
Released under the MIT license, see the GitHub page for the full legal boilerplate.
tldr: you freely can do whatever you like with this code so long as this message is retained and you cite the GitHub: https://github.com/Thegolfingocto/ArchitecturalComplexity
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
'''


from typing import List, Callable
import os
import shutil
import json
from inspect import ismethod

from Arch.IConfig import IConfig
from Arch.Logger import *
from Arch.Utils.Utils import *
from Arch.Utils.CJSON import *


class ICache(IConfig):
    def __init__(self, strDirectoryName: str = "", dConfig: dict = {}):
        '''
        Abstract class for hash ID equipped config based caching system.
        Inherits from IConfig for hash IDs.
        To use this class, inherit from it and provide a base operating directory at creation time. 
        You can also hardcode a directory name as a derived class member named strBaseDir.
        To provide custom summaries of cached configurations, define a PrintConfigSummary() method. 
        '''
        super().__init__(dConfig)
        
        if not hasattr(self, "strBaseDir"):
            if len(strDirectoryName) == 0:
                printf("No strBaseDir or strDirectoryName provided!", ERROR)
            self.strBaseDir = strDirectoryName
        #Make sure path string is in the correct format
        if self.strBaseDir[-1] != "/": self.strBaseDir += "/"
        #Create main operating directory if it's not already there
        if not os.path.isdir(self.strBaseDir):
            printf("Creating new Cache folder {}".format(self.strBaseDir), INFO)
            os.mkdir(self.strBaseDir)

        #setup the file I/O managers
        self.lfBaselineConfig = LockedJSONFile(self.strBaseDir + "BaselineConfig.cjson", indent = 4)
        self.lfCacheMap = LockedJSONFile(self.strBaseDir + "CacheMap.json", indent = 4)
        
        #setup operating directory
        self.LoadBaselineConfig()
        self.dCfg = self.IValidateConfig() #make sure we start w/ an updated version of the config, if defined by the derived app
        self.CacheMap = {}
        self.LoadCacheMap()
        self.UpdateCacheMap()

        return
    
    def HasMethod(self, strQ: str) -> bool:
        return (hasattr(self, strQ) and ismethod(getattr(self, strQ)))
    
    def GetCurrentFolder(self) -> str:
        '''
        Creates path string of directory for current configuration
        '''
        return self.strBaseDir + self.IGenHash() + "/"
    
    def RemoveFolder(self, strHash) -> None:
        if strHash in self.CacheMap.keys():
            del self.CacheMap[strHash]
            # with open(self.strBaseDir + "CacheMap.json", "w") as f:
            #         json.dump(self.CacheMap, f)
            self.lfCacheMap.Write(self.CacheMap)

        if os.path.isdir(self.strBaseDir + strHash + "/"):
            shutil.rmtree(self.strBaseDir + strHash + "/")

        return
    
    def UpdateBaselineConfig(self, dBaselineCfg: dict = None) -> None:
        if dBaselineCfg is not None and dBaselineCfg == self.dBaselineCfg: return
        if dBaselineCfg is None: dBaselineCfg = self.dBaselineCfg

        self.dBaselineCfg = dBaselineCfg
        self.SetBaselineConfig(self.dBaselineCfg)

        # with open(self.strBaseDir + "BaselineConfig.cjson", "w") as f:
        #     json.dump(self.dBaselineCfg, f, indent = 4)
        self.lfBaselineConfig.Write(self.dBaselineCfg)

        self.LoadCacheMap() #??
        
        return

    def LoadBaselineConfig(self) -> None:
        if not os.path.exists(self.strBaseDir + "BaselineConfig.cjson"):
            printf("No baseline config found for Cache folder {}".format(self.strBaseDir), INFO)
            if GetInput("Save current config as baseline? (Y/X)"):
                # with open(self.strBaseDir + "BaselineConfig.cjson", "w") as f:
                #     json.dump(self.dCfg, f, indent = 2)
                self.dBaselineCfg = self.dCfg
                self.lfBaselineConfig.Write(self.dBaselineCfg)
        else:
            # with open(self.strBaseDir + "BaselineConfig.cjson", "r") as f:
            #     self.dBaselineCfg = json.load(f)
            self.dBaselineCfg = self.lfBaselineConfig.Read()

        self.SetBaselineConfig(self.dBaselineCfg)
        
        return
    
    def UpdateCacheMap(self, dTempCfg: dict = None) -> str:
        '''
        Adds current configuration to the cache map if it does not already exist and returns its hash ID
        '''
        if dTempCfg is None: dTempCfg = copy.deepcopy(self.dCfg)

        strHash = self.IGenHash(dTempCfg)
        if strHash not in self.CacheMap.keys():
            self.CacheMap[strHash] = dTempCfg
            # with open(self.strBaseDir + "CacheMap.json", "w") as f:
            #     json.dump(self.CacheMap, f)
            self.lfCacheMap.Write(self.CacheMap)

        strCfgFolder = self.strBaseDir + strHash + "/"
        if not os.path.isdir(strCfgFolder):
            os.mkdir(strCfgFolder)
            with open(strCfgFolder + "Config.json", "w") as f:
                json.dump(dTempCfg, f, indent = 2)
        
        return strHash

    def LoadCacheMap(self, bCheckHash = True) -> None:
        '''
        read/generate hash ID -> config map.
        This function automatically checks for updates to the hashing scheme. To disable this functionality, pass bCheckHash = False
        This function is called upon ICache creation
        '''
        if not os.path.exists(self.strBaseDir + "CacheMap.json"):
            printf("Generating CacheMap for directory " + self.strBaseDir, NOTICE)
            self.CacheMap = {}
        else:
            printf("Loading Cache Map", INFO)
            # with open(self.strBaseDir + "CacheMap.json", "r") as f:
            #     self.CacheMap = json.load(f)
            self.CacheMap = self.lfCacheMap.Read()
        
        #first, collect all the hashes
        vecHashPaths = []
        for f in os.listdir(self.strBaseDir):
            strHash = os.fsdecode(f)
            if not os.path.isdir(self.strBaseDir + strHash) or "BAK" in strHash or "cache" in strHash.lower(): continue #results/models/configs stored in subdirs only. Skip BAK directories and sub-caches
            vecHashPaths.append(strHash)

        while len(vecHashPaths) > 0:
            #print(vecHashPaths[:10])
            strHash = vecHashPaths[0]

            if not os.path.exists(self.strBaseDir + strHash + "/Config.json"):
                printf("Warning! Found directory with no config: " + strHash, WARNING)
                printf("Contents of Directory to be Overwritten:")
                for f2 in os.listdir(self.strBaseDir + strHash):
                    printf(os.fsdecode(f2))
                printf("----------------------------------")
                if GetInput("Delete? (Y/X)"):
                    self.RemoveFolder(strHash)
                    vecHashPaths.remove(strHash)
                continue
            #print(strHash)
            with open(self.strBaseDir + strHash + "/Config.json", "r") as f:
                TempCfg = json.load(f)
            
            if bCheckHash:
                strNewHash = self.IGenHash(TempCfg)
                if strNewHash != strHash:
                    printf("Outdated hash for configuration " + strHash + ". New hash: " + strNewHash, NOTICE)
                    #input("Safety!")

                    if os.path.isdir(self.strBaseDir + strNewHash) and len(os.listdir(self.strBaseDir + strNewHash)) > 0:
                        printf("Existing folder {} found during hash update for old config {}.".format(strNewHash, strHash), WARNING)

                        #if we haven't re-hashed the problem yet, move it to the front of the line and hope we get lucky
                        if strNewHash in vecHashPaths:
                            vecHashPaths.remove(strNewHash)
                            vecHashPaths = [strNewHash] + vecHashPaths
                            printf("Attempting to re-order around the problem...")
                            continue
                        
                        #otherwise, we're stuck with the hash conflict, let the user decide what to do
                        with open(self.strBaseDir + strNewHash + "/Config.json", "r") as fc:
                            OldCfg = json.load(fc)
                        PrintDictDiff(TempCfg, OldCfg)
                        printf("Contents of Directory to be Overwritten:")
                        for f2 in os.listdir(self.strBaseDir + strNewHash):
                            printf(os.fsdecode(f2))
                        printf("----------------------------------")
                        if GetInput("Delete? (Y/X)"): self.RemoveFolder(strNewHash)
                        else: continue

                    shutil.move(self.strBaseDir + strHash, self.strBaseDir + strNewHash)

                if strNewHash not in self.CacheMap.keys(): self.CacheMap[strNewHash] = TempCfg
            else:
                self.CacheMap[strHash] = TempCfg

            vecHashPaths.remove(strHash)
        

        # with open(self.strBaseDir + "CacheMap.json", "w") as f:
        #     json.dump(self.CacheMap, f)
        self.lfCacheMap.Write(self.CacheMap)
                        
        return

    
    def DisplayCache(self) -> None:
        '''
        Prints a summary of each cached config to the console. 
        By default, prints the first key->value pair of each config. 
        If additional information is desired, define a PrintConfigSummary() method.
        PrintConfigSummary should accept a config dict and print to console any desired information.
        '''
        self.LoadCacheMap()
        for f in os.listdir(self.strBaseDir):
            strHash = os.fsdecode(f)
            if not os.path.isdir(self.strBaseDir + strHash) or "cache" in strHash.lower(): continue #results/models/configs stored in subdirs only
            with open(self.strBaseDir + strHash + "/Config.json", "r") as f:
                TempCfg = json.load(f)
            printf("Summary of Configuration: " + self.IGenHash(TempCfg), NOTICE)
            if self.HasMethod("PrintConfigSummary"):
                self.PrintConfigSummary(TempCfg)
            else:
                try:
                    strKey = list(TempCfg.keys())[0]
                    strValue = str(TempCfg[strKey])
                    printf("No PrintConfigSummary() method provided. First Key->Value pair is:", NOTICE)
                    printf(strKey + "->" + strValue, NOTICE)
                except:
                    printf("No PrintConfigSummary() method provided, and first value is not representable as a string", NOTICE)
            printf("-----------------------------------------------------------------", NOTICE)
            
        return
    
    def QueryCache(self, dQ: dict) -> None:
        '''
        Pass an input dict with any number of KVPs, this prints out all configs with matching KVPs
        '''
        
        self.LoadCacheMap()
        for f in os.listdir(self.strBaseDir):
            strHash = os.fsdecode(f)
            if not os.path.isdir(self.strBaseDir + strHash) or "cache" in strHash.lower(): continue #results/models/configs stored in subdirs only
            with open(self.strBaseDir + strHash + "/Config.json", "r") as f:
                TempCfg = json.load(f)
            
            if not IsDictSubset(dQ, TempCfg): continue
            
            printf("Summary of Configuration: " + self.IGenHash(TempCfg), NOTICE)
            if self.HasMethod("PrintConfigSummary"):
                self.PrintConfigSummary(TempCfg)
            else:
                try:
                    strKey = list(TempCfg.keys())[0]
                    strValue = str(TempCfg[strKey])
                    printf("No PrintConfigSummary() method provided. First Key->Value pair is:", NOTICE)
                    printf(strKey + "->" + strValue, NOTICE)
                except:
                    printf("No PrintConfigSummary() method provided, and first value is not representable as a string", NOTICE)
            printf("-----------------------------------------------------------------", NOTICE)
        
        return
    
    def QueryNumCached(self, dQ: dict) -> int:
        '''
        Pass an input dict with any number of KVPs, this returns the number of cache entries matching the query
        '''
        
        iCnt = 0
        self.LoadCacheMap()
        for f in os.listdir(self.strBaseDir):
            strHash = os.fsdecode(f)
            if not os.path.isdir(self.strBaseDir + strHash) or "cache" in strHash.lower(): continue #results/models/configs stored in subdirs only
            with open(self.strBaseDir + strHash + "/Config.json", "r") as f:
                TempCfg = json.load(f)
            
            if not IsDictSubset(dQ, TempCfg): continue
            iCnt += 1

        return iCnt
    
    def GetCachedIDs(self, dQ: dict) -> int:
        '''
        Pass an input dict with any number of KVPs, this returns a list of hash IDs matching the query
        '''
        
        vecRet = []
        self.LoadCacheMap()
        for f in os.listdir(self.strBaseDir):
            strHash = os.fsdecode(f)
            if not os.path.isdir(self.strBaseDir + strHash) or "cache" in strHash.lower(): continue #results/models/configs stored in subdirs only
            with open(self.strBaseDir + strHash + "/Config.json", "r") as f:
                TempCfg = json.load(f)
            
            if not IsDictSubset(dQ, TempCfg): continue
            vecRet.append(strHash)

        return vecRet
    
    def GetValueset(self, strKey: str) -> list:
        '''
        Returns a list of distinct values corresponding to the provided key
        '''
        vecRet = []
        self.LoadCacheMap()
        for f in os.listdir(self.strBaseDir):
            strHash = os.fsdecode(f)
            if not os.path.isdir(self.strBaseDir + strHash) or "cache" in strHash.lower(): continue #results/models/configs stored in subdirs only
            with open(self.strBaseDir + strHash + "/Config.json", "r") as f:
                TempCfg = json.load(f)
            
            if strKey not in TempCfg.keys(): continue
            if TempCfg[strKey] not in vecRet: vecRet.append(TempCfg[strKey])

        return vecRet
    
    def FindCachedStuff(self, strThing: str, dFilter: dict = None, bDir: bool = False, bPrint: bool = False, cFoundItemCB: Callable[[str], any] = None) -> None:
        '''
        Searches through (and optionally filters) existing configs and checks if strThing file/folder exists in the corresponding 
        subdirectory
        '''
        self.LoadCacheMap()
        for f in os.listdir(self.strBaseDir):
            strHash = os.fsdecode(f)
            if not os.path.isdir(self.strBaseDir + strHash) or "cache" in strHash.lower(): continue #results/models/configs stored in subdirs only
            with open(self.strBaseDir + strHash + "/Config.json", "r") as f:
                dTempCfg = json.load(f)

            dTempCfg = self.IValidateConfig(dTempCfg)
            
            if dFilter is not None and not IsDictSubset(dFilter, dTempCfg): continue
            
            if not os.path.exists(self.strBaseDir + strHash + "/" + strThing): continue
            
            if bDir:
                if os.path.isdir(self.strBaseDir + strHash + "/" + strThing):
                    if not os.listdir(self.strBaseDir + strHash + "/" + strThing): continue #skip empty directories
            
            printf("Found {} for config {}".format(strThing, strHash))

            if bPrint:
                printf("Summary of Configuration: " + self.IGenHash(dTempCfg), NOTICE)
                if self.HasMethod("PrintConfigSummary"):
                    self.PrintConfigSummary(dTempCfg)
                else:
                    try:
                        strKey = list(dTempCfg.keys())[0]
                        strValue = str(dTempCfg[strKey])
                        printf("No PrintConfigSummary() method provided. First Key->Value pair is:", NOTICE)
                        printf(strKey + "->" + strValue, NOTICE)
                    except:
                        printf("No PrintConfigSummary() method provided, and first value is not representable as a string", NOTICE)
                

            if cFoundItemCB is not None:
                cFoundItemCB(self.strBaseDir + strHash + "/") #Let the caller do whatever they want with the found path

            if bPrint: 
                printf("-----------------------------------------------------------------", NOTICE)
            
        return
    
    
#Copy paste and modify this skeleton code to suit project needs
class SampleCache(ICache):
    #If inheriting only from ICache, __init__() can be removed
    def __init__(self, strDirectoryName: str = "", dConfig: dict = {}, vecIgnoredFields: List[str] = []):
        super().__init__(strDirectoryName, dConfig, vecIgnoredFields)
    
    # def PrintConfigSummary(self, dTempCfg: dict) -> None:
    #     iL = len(dTempCfg.keys())
    #     printf("There are {} keys in the config".format(iL), INFO)
    #     return

if __name__ == "__main__":
    #example usage
    
    #uncomment these lines to enable logging to file
    #if Logger.GLOG is None:
    #    Logger.GLOG = Logger.Logger("Sample.log")
    Cache = SampleCache(strDirectoryName = "SampleCache", dConfig = {"test": 7, "memes": 42}, vecIgnoredFields = ["memes"])
    Cache.UpdateCacheMap()
    Cache.DisplayCache()
    