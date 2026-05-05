'''
Code written by Nicholas J. Cooper.
Released under the MIT license, see the GitHub page for the full legal boilerplate.
tldr: you freely can do whatever you like with this code so long as this message is retained and you cite the GitHub: https://github.com/combinatoriallabs/ArchitecturalComplexity
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
'''


from typing import Union
import os
import shutil
import json
import time
from inspect import ismethod
import hashlib
import numbers
import copy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tqdm

import torch

from Arch.Datasets import IDataset
from Arch.ICache import ICache
from Arch.Models.IModel import IModel
from Arch.Utils.Utils import *
from Arch.Models.ModelUtils import *
from Arch.Logger import *
import Arch.Logger as Logger


def CountParams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

'''
General info for Arch submodule:
Naming convention: lowercase prefix denoting variable type followed by PascalCase name. Examples:
dX:= dict, fX:= float, iX:= int, tX:= pytorch object, vecX:= list or iterable, strX:= string, etc.
Directory sturcture: {ProvidedOperatingDirectory}/{HashID}/{Config.json, Model.pth, Result.json}
'''

class ITrainer(ICache):
    '''
    Abstract trainer class. Inherits hash-id config caching system and provides basic model/result saving/loading logic.
    To use this class, inherit from it, modify config details as desired, provide impls for model specifics, and add any additional
    functionality (e.g. latent target calculators for HSKD). Set strCacheDir to desired operating folder. All models/results/configs
    will stored there. 
    '''
    def __init__(self, strCacheDir: str = "", dConfig: dict = {}, bStartLog: bool = False):
        super().__init__(strCacheDir, dConfig)
        #print("ITrainer setup, SaveModel is {}".format(self.dCfg["SaveModel"]))
        #add things which should always be ignored for hash generation here
        self.UpdateCacheMap()
        if ":" not in self.dBaselineCfg.keys():
            self.dBaselineCfg[":"] = ["SaveModel", "SaveResult", "EvalInterval", "CheckpointInterval"]
            self.UpdateBaselineConfig()
        
        self.tModel: IModel = None
        self.dModules: torch.nn.ModuleDict = None
        self.tBestModel = None #state dict not nn.Module
        self.fBestPerf: float = -1
        self.iPerfIdx: int = 0
        self.bHigherBetter: bool = True
        self.dResult: dict = None
        
        self.dsData: IDataset = None
        
        self.tLossFcn = None
        self.tOpt = None
        self.tSch = None
        
        self.vecTrainMetrics = []
        self.vecTestMetrics = []

        self.vecTrainMetricNames = []
        self.vecTestMetricNames = []
        
        self.iTestBatchSize: int = -1
        self.iMaxRuns: int = 25

        self.iCurrentEpoch: int = -1
        self.iSkipEpochs: int = 0
        self.iNEpochCBInterval: int = -1
        self.cNEpochCallback: callable = None
        self.cStopCondCallback: callable = None
        
        self.device = "cpu" if not torch.cuda.is_available() else "cuda"
        
        if "SaveModel" not in self.dCfg.keys():
            self.dCfg["SaveModel"] = True
        if "SaveResult" not in self.dCfg.keys():
            self.dCfg["SaveResult"] = True
            
        if bStartLog and Logger.GLOG is None:
            Logger.GLOG = Logger.Logger(self.strBaseDir + "trainer.log")
        
        return
    
    #----------------Utility Functions------------------#
    '''
    Some random QOL things
    '''
    
    def HasMethod(self, strQ: str) -> bool:
        return (hasattr(self, strQ) and ismethod(getattr(self,strQ)))
    
    def GetBestPerf(self, dR: dict) -> float:
        fRet = 0 if self.bHigherBetter else 2.**64
        for dRun in dR["Runs"]:
            for vecTsM in dRun["TestMetrics"]:
                if self.bHigherBetter and vecTsM[self.iPerfIdx] > fRet:
                    fRet = vecTsM[self.iPerfIdx]
                if not self.bHigherBetter and vecTsM[self.iPerfIdx] < fRet:
                    fRet = vecTsM[self.iPerfIdx]

        return fRet
    
    def SetNEpochCallback(self, iN: int, cCB: callable) -> None:
        '''
        cCB should accept the following args: (vecTrM, vecTsM, fTime)
        vecTrM - list of in progress train metrics
        vecTsM - list of in progress test metrics
        fTime - current time spent in ITrain()
        '''
        self.cNEpochCallback = cCB
        self.iNEpochCBInterval = iN
        return
    
    def SetStopConditionCallback(self, cCB: callable) -> None:
        '''
        cCB should be: (vecTrM, vecTsM, fTime, fBestPerf)-->bool
        vecTrM - list of in progress train metrics
        vecTsM - list of in progress test metrics
        fTime - current time spent in ITrain()
        fBestPerf - currently storved value of user configured fBestPerf

        Upon False return, stops training cycle.
        '''
        self.cStopCondCallback = cCB
        return
    
    def UpdateBatchSize(self, iNewBS: int) -> None:
        self.dCfg["BatchSize"] = iNewBS
        self.dsData.SetBatchSize(iNewBS)
        self.UpdateCacheMap()

        return
    
    def UpdateModel(self, strNewModel: str) -> None:
        self.dCfg["Model"] = strNewModel
        self.ILoadModel()
        self.UpdateCacheMap()

        return
    
    def ResetResult(self) -> None:
        self.tBestModel = None
        self.fBestPerf: float = -1
        #self.iPerfIdx: int = 0
        self.dResult: dict = None

        return

    def IPrintConfigSummary(self, dCfg: dict = None) -> None:
        if dCfg is None: dCfg = self.dCfg
        return self.PrintConfigSummary(dCfg)

    def PrintConfigSummary(self, dCfg: dict) -> None:
        '''
        This is a simple default config printing function. It only assumes the existence of KVPs which are directly integrated into ITrainer. 
        These are: Model, Dataset, Optimizer, NumRuns, NumEpochs.
        Overwrite this method in derived class if desired.
        '''
        printf("Model: {}, Dataset: {}".format(dCfg.get("Model"), dCfg.get("Dataset")), INFO)
        printf("Optimizer: {}".format(dCfg.get("Optimizer")), INFO)
        printf("Runs: {}, Epochs: {}".format(dCfg.get("NumRuns"), dCfg.get("NumEpochs")), INFO)

        return
    
    def GetResult(self, dCfg: Union[dict, str] = None, bY: bool = False) -> dict:
        '''
        When dCfg is passed, this temporarily overwrites the config, queries the cache, and returns the result if found.
        Otherwise, the current Trainer's result is returned.
        '''

        if dCfg is None: self.LoadResult(bY = bY)

        dRet = None
        dCurrentCfg = copy.deepcopy(self.dCfg)
        
        if type(dCfg) == str:
            with open(dCfg, "r") as f:
                dCfg = json.load(f)
        
        dCfg = self.IValidateConfig(dCfg)
        self.dCfg = dCfg
        
        iFoundRuns = self.FindConfigRuns()
        if iFoundRuns == self.GetValue("NumRuns"):
            with open(self.GetCurrentFolder() + "Result.json", "r") as f:
                dRet = json.load(f)
        elif iFoundRuns > 0:
            printf("WARNING! Only found config with {}/{} runs for query: {}".format(iFoundRuns, self.GetValue("NumRuns"), self.IGenHash()))
            self.dCfg["NumRuns"] = iFoundRuns
            with open(self.GetCurrentFolder() + "Result.json", "r") as f:
                dRet = json.load(f)
        else:
            self.IPrintConfigSummary()
            printf("Failed to find result for config: " + self.IGenHash())
            
        self.dCfg = dCurrentCfg

        return dRet
    
    def FindConfigRuns(self) -> int:
        '''
        This function checks if the same config has been processed with more/fewer runs and consolidates if need be
        '''
        iRuns = self.GetValue("NumRuns")
        vecCfgsWithRuns = []
        vecResults = []
        vecFolders = []
        vecHashes = []

        for i in range(1, self.iMaxRuns + 1):
            self.dCfg["NumRuns"] = i
            if self.IGenHash() in self.CacheMap.keys() and os.path.exists(self.GetCurrentFolder() + "Result.json"):
                vecCfgsWithRuns.append(i)
                with open(self.GetCurrentFolder() + "Result.json", "r") as f:
                    self.dResult = json.load(f)
                self.ProcessResults()
                vecResults.append(copy.deepcopy(self.dResult))
                vecFolders.append(self.GetCurrentFolder())
                vecHashes.append(self.IGenHash())

        #if we didn't find anything, just quit out
        if len(vecResults) == 0:
            self.dCfg["NumRuns"] = iRuns
            return 0

        #grab the config with the most runs
        self.dCfg["NumRuns"] = vecCfgsWithRuns[-1]
        #consolidate the experiments if they are segmented
        if len(vecResults) > 1:
            strBestModelDir = self.GetCurrentFolder() if os.path.exists(self.GetCurrentFolder() + "Model.pth") else None
            fBestModelPerf = self.GetBestPerf(self.dResult)
            for i in range(len(vecResults[:-1])):
                dR = vecResults[i]
                #keep track of where the best model is located
                if os.path.exists(vecFolders[i] + "Model.pth"):
                    if strBestModelDir is None:
                        strBestModelDir = vecFolders[i]
                        fBestModelPerf = self.GetBestPerf(dR)
                    elif self.bHigherBetter and self.GetBestPerf(dR) > fBestModelPerf:
                        strBestModelDir = vecFolders[i]
                        fBestModelPerf = self.GetBestPerf(dR)
                    elif not self.bHigherBetter and self.GetBestPerf(dR) < fBestModelPerf:
                        strBestModelDir = vecFolders[i]
                        fBestModelPerf = self.GetBestPerf(dR)

                self.dResult["Runs"] += dR["Runs"]
                self.dResult["NumRuns"] += len(dR["Runs"])

            #allow the derived class to process the consolidated dict
            self.ProcessResults()

            self.dCfg["NumRuns"] = self.dResult["NumRuns"]
            #save the consolidated result
            self.SaveResult()
            #move the best model directory (including features/etc...) over, if it exists
            if strBestModelDir is not None:
                for f in os.listdir(strBestModelDir):
                    strF = os.fsdecode(f)
                    if strF in ["Config.json", "Result.json"]: continue
                    shutil.move(strBestModelDir + strF, self.GetCurrentFolder() + strF)
            #copy anything floating around in the other folders that wasn't already taken care of by the best version
            for strDir in vecFolders:
                if strDir == strBestModelDir: continue
                for f in os.listdir(strDir):
                    strF = os.fsdecode(f)
                    if strF in ["Config.json", "Result.json"] or os.path.exists(self.GetCurrentFolder() + strF): continue
                    shutil.move(strDir + strF, self.GetCurrentFolder() + strF)
            #remove the old folders
            for strHash in vecHashes:
                self.RemoveFolder(strHash)
            printf("Consolidated experiments into config " + self.IGenHash() + " with a total of " + str(self.GetValue("NumRuns")) + " runs.")

        iRet = self.GetValue("NumRuns")
        if iRuns > iRet: self.dCfg["NumRuns"] = iRuns
        #print(self.GetCurrentFolder(), iRet, iRuns)
        return iRet
            

    #----------------Generic Functions------------------#
    '''
    This section of functions provide the generic functionality of the ITrainer class. This includes model/result loading/saving, 
    and config/hash id management.
    Generally, nothing else is required of derived classes in order to benefit from this functionality.
    '''
    
    
    def SetEvalMetric(self, iIdx: int, bHB: bool = True) -> None:
        self.iPerfIdx = iIdx
        self.bHigherBetter = bHB
        return
    
    def SaveModel(self, bY: bool = False) -> None:
        #print("Hello from SaveModel()!")
        if self.tBestModel is None:
            printf("ERROR! SaveModel() called but no model present.", ERROR)
            return
        
        strDir = self.GetCurrentFolder()
        
        #make sure the current config exists in the cache
        self.UpdateCacheMap()
            
        if os.path.exists(strDir + "/Model.pth"):
            printf("A model already exists for this configuration. Overwrite? (Y/X)", WARNING)
            chC = "Y" if bY else input()
            if chC != "Y":
                printf("Aborting SaveModel()", NOTICE)
                return
            printf("Overwriting model for configuration " + self.IGenHash(), NOTICE)
        
        torch.save(self.tBestModel, strDir + "Model.pth")
        
        if self.HasMethod("SaveSubmodels"):
            self.SaveSubmodels()
        
        return
    
    def SaveModelCheckpoint(self, iC: int) -> None:
        strDir = self.GetCurrentFolder()
        #make sure the current config exists in the cache
        if not os.path.isdir(strDir) or self.IGenHash() not in self.CacheMap.keys():
            self.UpdateCacheMap()
            
        torch.save(self.tModel.state_dict(), strDir + "ModelCkpt_" + str(iC) + ".pth")
        
        if self.HasMethod("SaveSubmodelCheckpoint"):
            self.SaveSubmodelCheckpoint(iC)
        
        return
    
    def GetCheckpointIntervals(self) -> list[int]:
        #find all available checkpoints
        vecCkpts = []
        for f in os.listdir(self.GetCurrentFolder()):
            strF = os.fsdecode(f)
            if "Ckpt" in strF: vecCkpts.append(int(strF.split("_")[-1].split(".")[0]))
    
        return list(torch.sort(torch.tensor(vecCkpts))[0])
    
    def SaveResult(self, bY: bool = False) -> None:
        if self.dResult is None:
            printf("ERROR! SaveResult() called but no result present. Launch training run? (Y/X)", ERROR)
            return
            
        strDir = self.GetCurrentFolder()
        
        #make sure the current config exists in the cache
        self.UpdateCacheMap()
            
        if os.path.exists(strDir + "Result.json"):
            printf("A result already exists for this configuration. Overwrite? (Y/X)", WARNING)
            chC = "Y" if bY else input()
            if chC != "Y":
                printf("Aborting SaveResult()", INFO)
                return
            printf("Overwriting result for configuration " + self.IGenHash(), INFO)
            
        with open(strDir + "/Result.json", "w") as f:
            json.dump(self.dResult, f, indent = 2)
        
        return
    
    def LoadTrainedModel(self, bY: bool = False, iCkpt: int = -1) -> None:
        iRuns = self.GetValue("NumRuns")
        iFoundRuns = self.FindConfigRuns()

        strDir = self.GetCurrentFolder()
        strM = "Model"
        if iCkpt > -1: strM += "Ckpt_" + str(iCkpt)
        strM += ".pth"

        if iFoundRuns == 0:
            self.IPrintConfigSummary()
            printf("Failed to find model for configuration: " + self.IGenHash() + ". Launch training run? (Y/X)", ERROR)
            chC = "Y" if bY else input()
            if chC == "Y":
                printf("Launching Train()", INFO)
                self.dCfg["SaveModel"] = True
                self.ITrain(bY)
            else:
                printf("Aborting LoadTrainedModel()", INFO)
                return
            
        if iFoundRuns > 0 and iFoundRuns < iRuns:
            self.IPrintConfigSummary()
            printf("Failed to find model for configuration: " + self.IGenHash() + ", but found configuration with " + str(iFoundRuns) + " runs instead.", NOTICE)
            printf("Launch training run to finish the remaining " + str(iRuns - iFoundRuns) + " runs? (Y/X)", ERROR)
            chC = "Y" if bY else input()
            if chC == "Y":
                printf("Launching Train()", INFO)
                self.dCfg["SaveModel"] = True
                self.ITrain(bY, iNewRuns = iRuns - iFoundRuns)


                #move the stuff from the partial experiment over to the new location
                self.dCfg["NumRuns"] = iFoundRuns
                strPartialDir = self.GetCurrentFolder()
                strPartialHash = self.IGenHash()
                self.dCfg["NumRuns"] = iRuns
                for f in os.listdir(strPartialDir):
                    strF = os.fsdecode(f)
                    if strF == "Config.json" or os.path.exists(self.GetCurrentFolder() + strF): continue
                    shutil.move(strPartialDir + strF, self.GetCurrentFolder() + strF)

                
                #only delete the partial folder after we're sure the training went thru OK
                self.RemoveFolder(strPartialHash)
            else:
                self.dCfg["NumRuns"] = iFoundRuns        

        if self.tModel is None:
            if not self.ILoadModel():
                printf("ILoadModel() failed! Check that LoadModel() is defined properly.", ERROR)
                return
        self.tModel.load_state_dict(torch.load(strDir + strM), strict = False)
        
        self.tModel = self.tModel.to(self.device)
        
        #quick and dirty hack to allow derived classes to save/load sub models
        if self.HasMethod("LoadSubmodels"):
            self.LoadSubmodels(iCkpt)
        
        return
    
    def LoadResult(self, bY: bool = False) -> None:
        iRuns = self.GetValue("NumRuns")
        iFoundRuns = self.FindConfigRuns()

        strDir = self.GetCurrentFolder()
        
        if iFoundRuns == 0:
            self.IPrintConfigSummary()
            
            printf("Failed to find result for configuration: " + self.IGenHash() + ". Launch training run? (Y/X)", ERROR)
            chC = "Y" if bY else input()
            if chC == "Y":

                #see if there are any saved checkpoints
                iC = -1
                for f in os.listdir(strDir):
                    strF = os.fsdecode(f)
                    if "Ckpt" in strF:
                        ic = int(strF.split("_")[1].split(".")[0])
                        if ic > iC:
                            iC = ic
                if iC > 0:
                    printf("Found Checkpoint {}. Resume training from here? (Y/X)".format(iC), ERROR)
                    chC = "Y" if bY else input()
                    if chC == "Y":
                        strM = "ModelCkpt_" + str(iC) + ".pth"
                        if self.tModel is None:
                            if not self.ILoadModel():
                                printf("ILoadModel() failed! Check that LoadModel() is defined properly.", ERROR)
                                return
                        self.tModel.load_state_dict(torch.load(strDir + strM), strict = False)
                        
                        self.tModel = self.tModel.to(self.device)
                        
                        #quick and dirty hack to allow derived classes to save/load sub models
                        if self.HasMethod("LoadSubmodels"):
                            self.LoadSubmodels(iC)

                        self.iSkipEpochs = iC

                printf("Launching Train()", INFO)
                self.dCfg["SaveResult"] = True
                #print("Calling ITrain(), SaveModel is {}".format(self.dCfg["SaveModel"]))
                self.ITrain(bY)
            else:
                self.dCfg["NumRuns"] = iFoundRuns
            
        if iFoundRuns > 0 and iFoundRuns < iRuns:
            self.IPrintConfigSummary()
            printf("Failed to find result for configuration: " + self.IGenHash() + ", but found configuration with " + str(iFoundRuns) + " runs instead.", NOTICE)
            printf("Launch training run to finish the remaining " + str(iRuns - iFoundRuns) + " runs? (Y/X)", ERROR)
            chC = "Y" if bY else input()
            if chC == "Y":
                printf("Launching Train()", INFO)

                self.dCfg["SaveResult"] = True
                self.ITrain(bY, iNewRuns = iRuns - iFoundRuns)
                
                #move the stuff from the partial experiment over to the new location
                self.dCfg["NumRuns"] = iFoundRuns
                strPartialDir = self.GetCurrentFolder()
                strPartialHash = self.IGenHash()
                self.dCfg["NumRuns"] = iRuns
                for f in os.listdir(strPartialDir):
                    strF = os.fsdecode(f)
                    if strF == "Config.json" or os.path.exists(self.GetCurrentFolder() + strF): continue
                    shutil.move(strPartialDir + strF, self.GetCurrentFolder() + strF)
                
                #only delete the partial folder after we're sure the training went thru OK
                self.RemoveFolder(strPartialHash)
            else:
                printf("Aborting LoadResult()", INFO)
                return  

        with open(strDir + "Result.json", "r") as f:
            self.dResult = json.load(f)
        
        strH = hashlib.md5(json.dumps(self.dResult, sort_keys=True).encode('utf-8')).hexdigest()
        
        self.ProcessResults()
        
        #If the result was changed (e.g. from a derived class impl update), update the record on disk
        if strH != hashlib.md5(json.dumps(self.dResult, sort_keys=True).encode('utf-8')).hexdigest():
            printf("Result updated by ProcessResults(). Saving", NOTICE)
            with open(self.GetCurrentFolder() + "Result.json", "w") as f:
                json.dump(self.dResult, f, indent = 2)
                
        return
    
    
    #----------------Interface Functions------------------#
    '''
    Below are the various interface points used for loading models/dataloaders/optimizers/etc into the abstract trainer class.
    Generally, IX() requires a derived class method called X() to exist and to return the corresponding object(s). 
    Be sure to include all necessary KVPs in the Config{} so you can implement X() functions properly. No arguments are passed 
    to derived class impls. 
    '''
    
    def ILoadDataset(self, **kwargs) -> bool:
        '''
        Interface point for loading datasets into the trainer. Provide a LoadDataset() impl which returns desired tensors.
        Be sure to include all relevent info for LoadDataset() in self.dCfg{} so you can access it.
        '''
        if self.HasMethod("LoadDataset"):
            self.dsData = self.LoadDataset(**kwargs)
            return True
        return False
    
    def ILoadModel(self) -> bool:
        '''
        Interface point for loading models into the trainer. Provide a LoadModel() impl which returns a torch.nn.Module.
        Be sure to include all relevent info for LoadModel() in self.dCfg{} so you can access it.
        Define a ModuleDict called dParamGroups if you require subdivisions at this level.
        '''
        if self.HasMethod("LoadModel"):
            self.tModel = self.LoadModel()
            if self.dModules is None:
                self.dModules = torch.nn.ModuleDict()
            #copy over the model's defined groups, if they exist. Otherwise, just add the whole model as a module
            if hasattr(self.tModel, "dParamGroups"):
                for key in self.tModel.dParamGroups.keys():
                    self.dModules[key] = self.tModel.dParamGroups[key]
            else:
                self.dModules["Model"] = self.tModel #wrapper to allow easy additions to the main optimizer
            
            # torch.set_float32_matmul_precision('high')
            # if isinstance(self.tModel, torch.nn.Module):
            #     self.tModel = torch.compile(self.tModel)
            # elif isinstance(self.tModel, IModel):
            #     self.tModel.forward = torch.compile(self.tModel.forward)
            
            return True
        
        return False
    
    def ILoadOptimizer(self) -> bool:
        '''
        Interface point for loading optimizers into the trainer. Provide a LoadOptimizer() impl which returns desired object.
        Be sure to include all relevent info for LoadOptimizer() in self.dCfg{} so you can access it.
        '''
        if self.HasMethod("LoadOptimizer"):
            self.tOpt = self.LoadOptimizer()
            return True

        return False
    
    def ILoadLossFcn(self) -> bool:
        '''
        Interface point for loading optimizers into the trainer. Provide a LoadLossFcn() impl which returns desired object.
        Be sure to include all relevent info for LoadLossFcn() in self.dCfg{} so you can access it.
        '''
        if self.HasMethod("LoadLossFcn"):
            self.tLossFcn = self.LoadLossFcn()
            return True
        
        return False
    
    def ILoadScheduler(self) -> bool:
        '''
        Interface point for loading learning rate schedulers into the trainer. Provide a LoadScheduler() impl 
        which returns desired object. Be sure to include all relevent info for LoadScheduler() in self.dCfg{} so you can access it.
        '''
        if self.HasMethod("LoadScheduler"):
            self.tSch = self.LoadScheduler()
            return True

        return False
    
    def ILoadComponents(self) -> bool:
        if self.dsData is None or not self.dsData.Loaded():
            if not self.ILoadDataset():
                printf("LoadDataset Not Implemented.", ERROR)
                return False
        if not self.ILoadModel():
            printf("LoadModel Not Implemented.", ERROR)
            return False 
        if not self.ILoadLossFcn():
            printf("LoadLossFcn Not Implemented.", ERROR)
            return False
        if self.HasMethod("LoadComponents"):
            self.LoadComponents()
        if not self.ILoadOptimizer():
            printf("LoadOptimizer Not Implemented.", ERROR)
            return False
        if not self.ILoadScheduler():
            printf("LoadScheduler Not Implemented.", ERROR)
            return False
        return True
    
    def ITrain(self, bY: bool = False, iNewRuns: int = -1) -> bool:
        '''
        This interface point is somewhat more complex. ITrain() first tries to load all basic components, resets the result dict, 
        starts a timer, places model in train mode, then passes control to the derived class Train() and Eval() impls.
        If a NumRuns key is present in self.dCfg, ITrain() will loop accordingly and aggregate the results.
        ITrain() will also check for NumEpochs in self.dCfg.
        Train() and Eval() should take no arguments each return any number of floats comprised of whatever training run metrics 
        are desired. If additional components are required, define a LoadComponents() method. By default, ITrain() assumes that the first 
        float returned by Eval() is used for determining the best model, and that higher is better. Call SetValMetric() to change this
        behaviour.
        ITrainer provides a default PrintTrainingProgress() method which requires derived class to update the vec{Split}MetricNames
        variables. If desired, overwrite in derived class.
        If provided, ITrain will call ProcessResults() after all training runs have completed. This method should add/modify any 
        desired KVPs in self.dResult{}. 
        '''
        if not self.HasMethod("Train") or not self.HasMethod("Eval"):
            printf("Train and/or Eval methods missing!", ERROR)
            return False
        
        #manage the desired number of training runs
        if iNewRuns > 0:
            iR = iNewRuns
        elif self.GetValue("NumRuns") is not None:
            iR = self.GetValue("NumRuns")
        else: iR = 1
        
        if self.GetValue("EvalInterval") is not None:
            iE = self.GetValue("EvalInterval")
        else: iE = 1
        
        if self.GetValue("CheckpointInterval") is not None:
            iC = self.GetValue("CheckpointInterval")
        else: iC = -1
        
        #reset the result
        if iNewRuns == -1:
            self.dResult = {}
            if self.GetValue("NumEpochs") is not None:
                self.dResult["NumEpochs"] = self.GetValue("NumEpochs")
            else:
                self.dResult["NumEpochs"] = -1
            self.dResult["NumRuns"] = iR
            self.dResult["Runs"] = []
        else:
            self.dResult["NumRuns"] += iR
        self.vecTrainMetrics = []
        self.vecTestMetrics = []
        
        #reset the best performing metric
        if iNewRuns == -1:
            if self.bHigherBetter: self.fBestPerf = -1 * (2.**64)
            else: self.fBestPerf = 2.**64
            self.tBestModel = None
        else:
            self.fBestPerf = self.GetBestPerf(self.dResult)

        
        for n in range(iR):
            dRun = {}
            #Load all components
            if not self.ILoadComponents():
                printf("Error encountered attempting to load components for training run.", ERROR)
                return False
            start = time.time()
            cbtime = 0
            
            vecTrM = []
            vecTsM = []
            self.iCurrentEpoch = self.iSkipEpochs
            for e in range(self.iSkipEpochs, self.GetValue("NumEpochs")):
                TrainMetrics = self.Train()
                self.iCurrentEpoch += 1
                if isinstance(TrainMetrics, tuple):
                    vecTrM.append(list(TrainMetrics))
                elif isinstance(TrainMetrics, list):
                    vecTrM.append(TrainMetrics)
                elif isinstance(TrainMetrics, dict):
                    vecTrM.append(list(TrainMetrics.values()))
                    if len(self.vecTrainMetricNames) != len(TrainMetrics.keys()): self.vecTrainMetricNames = list(TrainMetrics.keys())
                else:
                    vecTrM.append([TrainMetrics])
                if (e + 1) % iE == 0:
                    TestMetrics = self.Eval()
                    if isinstance(TestMetrics, tuple):
                        fPerf = TestMetrics[self.iPerfIdx]
                        vecTsM.append(list(TestMetrics))
                    elif isinstance(TestMetrics, list):
                        fPerf = TestMetrics[self.iPerfIdx]
                        vecTsM.append(TestMetrics)
                    elif isinstance(TestMetrics, dict):
                        fPerf = TestMetrics[list(TestMetrics.keys())[self.iPerfIdx]]
                        vecTsM.append(list(TestMetrics.values()))
                        if len(self.vecTestMetricNames) != len(TestMetrics.keys()): self.vecTestMetricNames = list(TestMetrics.keys())
                    else:
                        fPerf = TestMetrics
                        vecTsM.append([TestMetrics])
                    if (self.bHigherBetter and fPerf > self.fBestPerf) or (not self.bHigherBetter and fPerf < self.fBestPerf):
                        #this allows derived classes to update their submodels in sync with the main model
                        if self.HasMethod("BestModelCB"):
                            self.BestModelCB()
                        self.fBestPerf = fPerf
                        self.tBestModel = copy.deepcopy(self.tModel.state_dict())
                        
                    self.PrintTrainingProgress(vecTrM[-1], vecTsM[-1])
                        
                if iC > 0 and (e + 1) % iC == 0:
                    self.SaveModelCheckpoint(e + 1)

                if self.iNEpochCBInterval > 0 and self.cNEpochCallback is not None and (e + 1) % self.iNEpochCBInterval == 0:
                    timeCBStart = time.time()
                    self.cNEpochCallback(vecTrM, vecTsM, time.time() - start)
                    cbtime += time.time() - timeCBStart

                if self.cStopCondCallback is not None:
                    timeCBStart = time.time()
                    if not self.cStopCondCallback(vecTrM, vecTsM, time.time() - start, self.fBestPerf):
                        printf("Aborting training run")
                        self.CacheMap["NumEpochs"] = e + 1
                        self.dResult["NumEpochs"] = e + 1
                        break
                    cbtime += time.time() - timeCBStart

                
            if iNewRuns > 0 and self.tBestModel is None and iR > 1:
                printf("No training progress has been made! Continue training? (Y/X)")
                chC = "Y" if bY else input()
                if chC != "Y":
                    return False
                        
            end = time.time()
            
            printf("Training run {}/{} completed in {}s".format(n + 1, iR, end - start - cbtime), INFO)

            self.vecTrainMetrics.append(vecTrM)
            self.vecTestMetrics.append(vecTsM)
            
            dRun["TrainMetrics"] = vecTrM
            dRun["TestMetrics"] = vecTsM
            dRun["TotalTime"] = end - start - cbtime
            
            self.dResult["Runs"].append(dRun)

            self.iSkipEpochs = 0 #only use checkpoints for one run
        
        self.dResult["Parameters"] = CountParams(self.tModel)
        
        try:
            self.ProcessResults()
        except:
            printf("WARNING, POSSIBLE LOSS OF DATA! ProcessResults() failed after training run")
            if GetInput("Save result as-is to folder {}? (Y/X)".format(self.GetCurrentFolder())):
                self.SaveResult(bY or iNewRuns > 0)
            if GetInput("Save model as-is to folder {}? (Y/X)".format(self.GetCurrentFolder())):
                self.SaveModel(bY or iNewRuns > 0)
            return
        
        if self.GetValue("SaveModel") and (iNewRuns == -1 or self.tBestModel is not None):
            self.SaveModel(bY or iNewRuns > 0)
        if self.GetValue("SaveResult"):
            self.SaveResult(bY or iNewRuns > 0)
        
        return True
    
    def PrintTrainingProgress(self, TrM, TsM) -> None:
        '''
        This is a simple default training progress function. Overwrite in derived class if desired.
        '''
        if len(self.vecTestMetricNames) != len(TsM) or len(self.vecTrainMetricNames) != len(TrM):
            printf("Warning! Metric names are not set up properly, default PrintTrainingProgress() cannot proceed", WARNING)
            return
        
        strP = "Epoch: {}/{}".format(self.iCurrentEpoch, self.GetValue("NumEpochs"))
        for i in range(len(self.vecTestMetricNames)):
            if isinstance(TsM[i], numbers.Number): strP += " " + self.vecTestMetricNames[i] + ": {:.4f}".format(TsM[i])
        for i in range(len(self.vecTrainMetricNames)):
            if isinstance(TrM[i], numbers.Number): strP += " " + self.vecTrainMetricNames[i] + ": {:.4f}".format(TrM[i])
        printf(strP, INFO)

        return
    
    def ProcessResults(self) -> None:
        '''
        This is a simple default result processing function. Computes basic stats of the set Eval Metric.
        '''
        if "TrainMetricNames" not in self.dResult.keys() or "TestMetricNames" not in self.dResult.keys():
            self.dResult["TrainMetricNames"] = self.vecTrainMetricNames
            self.dResult["TestMetricNames"] = self.vecTestMetricNames

        if len(self.dResult["TestMetricNames"]) < self.iPerfIdx + 1:
            printf("Warning! vecTestMetricNames is too short, default ProcessResults() cannot proceed", WARNING)
            return

        stats = np.zeros((self.dResult["NumRuns"]))
        for i in range(self.dResult["NumRuns"]):
            if self.bHigherBetter: stats[i] = max([self.dResult["Runs"][i]["TestMetrics"][j][self.iPerfIdx] for j in range(self.dResult["NumEpochs"])])
            else: stats[i] = min([self.dResult["Runs"][i]["TestMetrics"][j][self.iPerfIdx] for j in range(self.dResult["NumEpochs"])])
        self.dResult["Avg" + self.dResult["TestMetricNames"][self.iPerfIdx].replace(" ", "")] = np.mean(stats)
        self.dResult["Std" + self.dResult["TestMetricNames"][self.iPerfIdx].replace(" ", "")] = np.std(stats)
        if self.bHigherBetter: self.dResult["Max" + self.dResult["TestMetricNames"][self.iPerfIdx].replace(" ", "")] = np.max(stats)
        else: self.dResult["Min" + self.dResult["TestMetricNames"][self.iPerfIdx].replace(" ", "")] = np.min(stats)

        return

    def IDisplayResult(self, bPlotOnly: bool = False, bY: bool = False) -> None:
        '''
        This is a simple wrapper which loads/generates result and calls the derived class' DisplayResult() impl. 
        '''
        if self.dResult is None or len(list(self.dResult.keys())) == 0:
            self.LoadResult(bY = bY)
            
        return self.DisplayResult(bPlotOnly = bPlotOnly)
    
    def DisplayResult(self, bPlotOnly: bool = False) -> None:
        '''
        This is a simple default result plotter. Plots all numerical train/test metrics vs. epochs. Overwrite in derived class if desired.
        '''
        vecPlotColors = ["black", "red", "blue", "lime", "cyan", "forestgreen", "darkgray"]
        X = [i for i in range(1, self.GetValue("NumEpochs") + 1)]
        for i in range(len(self.dResult["Runs"])):
            dRun = self.dResult["Runs"][i]
            if i == len(self.dResult["Runs"]) - 1:
                for j in range(len(self.dResult["Runs"][0]["TrainMetrics"][0])):
                    if not isinstance(dRun["TrainMetrics"][0][j], numbers.Number): continue #skip non-numeric data
                    plt.plot(X, [dRun["TrainMetrics"][k][j] for k in range(len(dRun["TrainMetrics"]))], 
                             color = vecPlotColors[j % len(vecPlotColors)], 
                             label = self.dResult["TrainMetricNames"][j], linewidth = 3)
                for j in range(len(self.dResult["Runs"][0]["TestMetrics"][0])):
                    if not isinstance(dRun["TestMetrics"][0][j], numbers.Number): continue #skip non-numeric data
                    plt.plot(X, [dRun["TestMetrics"][k][j] for k in range(len(dRun["TestMetrics"]))], 
                             color = vecPlotColors[(j + len(self.dResult["TrainMetricNames"])) % len(vecPlotColors)],
                             label = self.dResult["TestMetricNames"][j], linewidth = 3)
            else:
                for j in range(len(self.dResult["Runs"][0]["TrainMetrics"][0])):
                    if not isinstance(dRun["TrainMetrics"][0][j], numbers.Number): continue #skip non-numeric data
                    plt.plot(X, [dRun["TrainMetrics"][k][j] for k in range(len(dRun["TrainMetrics"]))], 
                             color = vecPlotColors[j % len(vecPlotColors)], linewidth = 3)
                for j in range(len(self.dResult["Runs"][0]["TestMetrics"][0])):
                    if not isinstance(dRun["TestMetrics"][0][j], numbers.Number): continue #skip non-numeric data
                    plt.plot(X, [dRun["TestMetrics"][k][j] for k in range(len(dRun["TestMetrics"]))], 
                             color = vecPlotColors[(j + len(self.dResult["TrainMetricNames"])) % len(vecPlotColors)], linewidth = 3)
        
        if bPlotOnly: return

        plt.legend(fontsize=24)
        plt.title("Results for Configuration: " + self.IGenHash(), fontsize=32)
        plt.xlabel("Epoch Number", fontsize = 18)
        plt.xticks(X, fontsize = 12)
        plt.ylabel("Metrics", fontsize = 18)
        plt.grid()

        printf("--------------------------------------------------------", INFO)
        printf("Results for Configuration: " + self.IGenHash(), INFO)
        self.IPrintConfigSummary()

        if self.dResult["Parameters"] >= 1e9:
            fDiv = 1e9
            strDiv = "B"
        elif self.dResult["Parameters"] >= 1e6:
            fDiv = 1e6
            strDiv = "M"
        elif self.dResult["Parameters"] >= 1e3:
            fDiv = 1e3
            strDiv = "K"
        
        printf("Parameters: {:.2f} {}, Train Time: {:.2f}s/e".format(self.dResult["Parameters"] / fDiv, strDiv, self.dResult["Runs"][0]["TotalTime"] / self.dResult["NumEpochs"]))
        printf("Average {}: {:.2f} +/- {:.2f}".format(self.dResult["TestMetricNames"][self.iPerfIdx],
                                                        100 * self.dResult["Avg" + self.dResult["TestMetricNames"][self.iPerfIdx].replace(" ", "")], 
                                                        100 * self.dResult["Std" + self.dResult["TestMetricNames"][self.iPerfIdx].replace(" ", "")]), INFO)
        strMm = "Max" if self.bHigherBetter else "Min"
        printf(strMm + " {}: {:.2f}".format(self.dResult["TestMetricNames"][self.iPerfIdx],
                                            100 * self.dResult[strMm + self.dResult["TestMetricNames"][self.iPerfIdx].replace(" ", "")]))
        vecFinalEvalMetric = np.array([dRun["TestMetrics"][-1][self.iPerfIdx] for dRun in self.dResult["Runs"]])
        printf("Final {}: {:.2f}".format(self.dResult["TestMetricNames"][self.iPerfIdx], 
                                         100 * np.mean(vecFinalEvalMetric)))
        plt.show()
        plt.close()

        return
    

    #-------------------------------Analysis Section-------------------------------------#
    '''
    These functions provide per-layer feature saving functionality, and other useful things for post-training analysis.
    '''


    def GenFeatureStringBase(self) -> str:
        strN = self.GetValue("Model")
        strN += "_"
        strN += self.GetValue("Dataset")

        return strN

    def GenFeatureString(self, iL: int, strSplit = "train") -> str:
        strN = self.GenFeatureStringBase()
        strN += "_L"
        strN += str(iL)
        strN += "_" + strSplit
        return strN
    
    def GenFeatureStringLabel(self, strSplit = "train") -> str:
        strN = self.GenFeatureStringBase()
        strN += "_Y"
        strN += "_" + strSplit
        return strN

    #This is base impl of this function. Some templates override
    def GenPerLayerFeatures(self, vecLt: list[int] = None, iBatchSize: int = 500, strSplit: str = "train", bY: bool = False) -> None:

        self.LoadTrainedModel()
        self.tModel.eval()

        vecL = copy.deepcopy(vecLt) if vecLt is not None else [i for i in range(self.tModel.iGenLayers)]

        strDir = self.GetCurrentFolder() + "Features/"
        if not os.path.isdir(strDir):
            os.mkdir(strDir)

        if not os.path.isdir("./TensorCache/"):
            os.mkdir("./TensorCache/")
        if len(os.listdir("./TensorCache/")) > 0:
            printf("TensorCache is not empty! Continuing to generate features may erase any leftovers.", WARNING)
            if not GetInput("Overwrite? (Y/X)"): return
            for f in os.listdir("./TensorCache/"):
                strF = "./TensorCache/" + os.fsdecode(f)
                os.remove(strF)
        
        vecR = []
        for iL in vecL:
            strPath = strDir + self.GenFeatureString(iL, strSplit = strSplit) + ".pkl"
            #print(strPath)
            if os.path.exists(strPath):
                vecR.append(iL)
        for iR in vecR: vecL.remove(iR)

        strLabelPath = strDir + self.GenFeatureStringLabel(strSplit = strSplit) + ".pkl"
                
        if len(vecL) == 0 and os.path.exists(strLabelPath): return

        if not self.dsData.Loaded(): self.ILoadDataset() #only load dataset if necessary

        bDisableFirstPool = False

        #This is modified in templates
        if strSplit == "train": X, Y = self.dsData.X, self.dsData.Y
        else: X, Y = self.dsData.Xt, self.dsData.Yt

        torch.save(Y, strLabelPath)

        if len(vecL) == 0: return

        cnt = 0
        
        iN = X.shape[0]
        iL = iN // iBatchSize
        if iN % iBatchSize != 0: iL += 1
        vecX = []
        for i in range(iL):
            iE = (i+1)*iBatchSize
            if iE >= iN: iE = iN
            vecX.append(X[i*iBatchSize:iE,...])

        print("Generating missing features from layers: {}".format(vecL))
        #input()
        bFP = True

        if vecL[0] > 0:
            #skip ahead to avoid OOM errors in some fringe cases
            with torch.no_grad():
                for i in tqdm.tqdm(range(iL)):
                    for layer in self.tModel.vecLayers[:self.tModel.mapGenIdxToLiteralIdx[vecL[0]]]:
                        vecX[i] = layer(vecX[i].to(self.device))
                        if i == 0 and IsGeneralizedLayer(layer): cnt += 1
                    vecX[i] = vecX[i].to("cpu")
            iStartIdx = self.tModel.mapGenIdxToLiteralIdx[vecL[0]]
        else: iStartIdx = 0

        for layer in self.tModel.vecLayers[iStartIdx:]:
            if cnt > max(vecL): break
            
            if bDisableFirstPool and "pool" in layer.__class__.__name__.lower() and bFP:
                bFP = False
                continue

            with torch.no_grad():
                for i in tqdm.tqdm(range(iL)):
                    vecX[i] = layer(vecX[i].to(self.device)).to("cpu")

            if IsGeneralizedLayer(layer):
                if cnt in vecL:
                    #extremely jank saving scheme to avoid pytorch's unavoidable memcpy when using the cat() function
                    #save the representations batch-wise
                    print("Saving batched tensor")
                    sz = vecX[0].shape
                    for i in range(iL):
                        with open("./TensorCache/" + "temp" + str(i) + ".pkl", "wb") as f:
                            torch.save(vecX[i], f)
                    #delete this copy
                    del vecX
                    #now allocate a contiguous tensor and reload from disk
                    sz = list(sz)
                    sz[0] = iN
                    X = torch.zeros(sz)
                    print("Allocated contiguous memory for tensor")
                    for i in range(iL):
                        with open("./TensorCache/" + "temp" + str(i) + ".pkl", "rb") as f:
                            iE = (i+1)*iBatchSize
                            if iE >= iN: iE = iN
                            X[i*iBatchSize:iE,...] = torch.load(f) #yes, this way is actually MUCH more memory efficient than cat()
                    print("Loaded batched tensor into contiguous memory")
                    #now save it again in the correct place
                    with open(strDir + self.GenFeatureString(cnt, strSplit = strSplit) + ".pkl", "wb") as f:
                        torch.save(X.to("cpu"), f)
                    print("Saved contiguous tensor")
                    #delete the contiguous version and reload the batched versions to continue the forward prop.
                    del X
                    vecX = []
                    for i in range(iL):
                        with open("./TensorCache/" + "temp" + str(i) + ".pkl", "rb") as f:
                            vecX.append(torch.load(f))
                        #now we can remove the temporary file
                        os.remove("./TensorCache/" + "temp" + str(i) + ".pkl")
                    print("Reloaded batched tensor")
                cnt += 1
        
        del vecL
        
        return
    
    def GetLayerFeatures(self, vecLt: list[int] = None, strSplit: str = "train") -> list[torch.tensor]:
        iFoundRuns = self.FindConfigRuns()
        if iFoundRuns > 0: self.dCfg["NumRuns"] = iFoundRuns #overwrite NumRuns because stored features only come from the best run anyways

        if vecLt is None: vecLt = [self.tModel.iGenLayers - 1]

        vecRet = []

        for idx in vecLt:
            strF = self.GetCurrentFolder() + "Features/" + self.GenFeatureString(idx, strSplit = strSplit) + ".pkl"
            if not os.path.exists(strF):
                print(f"Cannot find features {strF}")
                continue

            vecRet.append(torch.load(strF))

        return vecRet
    
    def GetLayerFeatureLabels(self, strSplit: str = "train") -> torch.tensor:

        strF = self.GetCurrentFolder() + "Features/" + self.GenFeatureStringLabel(strSplit = strSplit) + ".pkl"
        if not os.path.exists(strF):
            print(f"Cannot find features {strF}")
            return None

        return torch.load(strF)
        


    def GenLogitPath(self) -> str:
        strPath = self.GetCurrentFolder() + "Features/"
        strPath += self.GetValue("Model") + "_"
        strPath += self.GetValue("Dataset") + "_Logits.pkl"
        return strPath
    
    def GenLogitLabelPath(self) -> str:
        strPath = self.GetCurrentFolder() + "Features/"
        strPath += self.GetValue("Model") + "_"
        strPath += self.GetValue("Dataset") + "_LogitLabels.pkl"
        return strPath

    def GenLogits(self, iBatchSize: int = 500) -> None:
        strFeatureDir = self.GetCurrentFolder() + "Features/"
        if not os.path.isdir(strFeatureDir):
            os.mkdir(strFeatureDir)

        strLogitPath = self.GenLogitPath()
        if os.path.exists(strLogitPath):
            print("Warning! Found existing logits: ", strLogitPath)
            if not GetInput("Overwrite? (Y/X)"):
                return

        self.LoadTrainedModel()
        self.tModel.eval()
        if not self.dsData.Loaded(): self.ILoadDataset()

        print("Generating Logits: ", strLogitPath)
        iN = (self.dsData.Size("train") // self.dsData.iBatchSize) + 1
        tLogits = torch.zeros((self.dsData.Size("train"), self.dsData.Classes()))
        tY = torch.zeros((self.dsData.Size("train")))

        with torch.no_grad():
            for i in tqdm.tqdm(range(iN)):
                iE = min([(i+1) * self.dsData.iBatchSize, tLogits.shape[0]])
                idx = torch.arange(i*self.dsData.iBatchSize, iE, 1)
                x, y = self.dsData.GetSamples(idx, "train")
                tL = self.tModel(x.to(self.device))
                tLogits[i*self.dsData.iBatchSize:iE,...] = tL.to("cpu")
                tY[i*self.dsData.iBatchSize:iE] = y

        with open(strLogitPath, "wb") as f: torch.save(tLogits, f)
        with open(self.GenLogitLabelPath(), "wb") as f: torch.save(tY, f)

        return
    
    def LoadLogits(self) -> tuple[torch.tensor, torch.tensor]:
        strLogitPath = self.GenLogitPath()
        if not os.path.exists(strLogitPath): self.GenLogits()

        with open(strLogitPath, "rb") as f: tLogits = torch.load(f)#.to(self.device)
        with open(self.GenLogitLabelPath(), "rb") as f: tY = torch.load(f)#.to(self.device)
            
        return tLogits, tY