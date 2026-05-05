'''
Code written by Nicholas J. Cooper.
Released under the MIT license, see the GitHub page for the full legal boilerplate.
tldr: you freely can do whatever you like with this code so long as this message is retained and you cite the GitHub: https://github.com/combinatoriallabs/ArchitecturalComplexity
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
'''


from typing import List
import hashlib
import json
import copy
from inspect import ismethod

from Arch.Utils.Utils import *
from Arch.Utils.CJSON import *
from Arch.Logger import *

class IConfig():
    def __init__(self, dConfig: dict) -> None:
        '''
        Abstract class for hash-id equipped configs.
        Pass complete config and a list of any key names to ignore when generating hash ID.
        To use this class, inherit from it and define a ModifyHashCfg() method to inject further modifications to the config
        used for generating the hash ID. 
        Define a GenHash() method to completely overwrite hash generation code
        '''
        self.dCfg, _ = SplitCJSON(dConfig)
        self.dBaselineCfg = None
        self.dFlatCfg = None
        self.dModifyCfg = None
        
        return
    
    def GetValue(self, strKey: str):
        '''
        Error safe config lookup. More importantly it sends an annoying print!
        '''
        if strKey not in self.dCfg.keys():
            printf("Key " + strKey + " not present in config!", ERROR)
            return None
        return self.dCfg[strKey]
    
    def GetBlankConfig(self) -> dict:
        return copy.deepcopy(self.dFlatCfg)
    
    def SetBaselineConfig(self, dBaselineCfg: dict) -> None:
        self.dFlatCfg, self.dModifyCfg = SplitCJSON(dBaselineCfg)
        return
    
    def IValidateConfig(self, dTempCfg: dict = None) -> dict:
        '''
        Ensures default values are copied into dTempCfg if they're missing, then calls user defined ValidateConfig() method if it exists.
        '''
        if dTempCfg is None: dTempCfg = copy.deepcopy(self.dCfg)

        vecKeys = dTempCfg.keys()
        for strK in self.dFlatCfg.keys():
            if strK not in vecKeys:
                #printf("Config is missing key {}".format(strK), INFO)
                #input()
                dTempCfg[strK] = self.dFlatCfg[strK]

        vecKeys = self.dFlatCfg.keys()
        for strK in list(dTempCfg.keys()):
            if strK not in vecKeys:
                #printf("Config has illegal key {}".format(strK), INFO)
                #input()
                del dTempCfg[strK]

        if self.HasMethod("ValidateConfig"):
            dTempCfg = self.ValidateConfig(dTempCfg)
        return dTempCfg
        
    def IGenHash(self, dTempCfg: dict = None, bPrint: bool = False) -> str:
        '''
        Interface method for generating hash ID for current configuration.
        Removes all KVPs from dTempCfg based on the CJSON syntax in BaselineConfig.json before passing control to user defined
        ModifyHashCfg(), if defined. Finally, md5 hash is computed. 
        '''
        
        if dTempCfg is None: dTempCfg = copy.deepcopy(self.dCfg)
        
        dTempCfg = self.IValidateConfig(dTempCfg)
        dTempCfg = ParseCJSON(dTempCfg, self.dModifyCfg)
            
        #pass the temp config thru user defined modification function if it exists
        if hasattr(self, "ModifyHashCfg") and ismethod(getattr(self, "ModifyHashCfg")):
            dTempCfg = self.ModifyHashCfg(dTempCfg)
            
        if bPrint: print(dTempCfg)
        #print(dTempCfg) #helpful debug

        return hashlib.md5(json.dumps(dTempCfg, sort_keys=True).encode('utf-8')).hexdigest()
    
if __name__ == "__main__":
    dTest = {
        "Dataset": "CIFAR10",
        "Dataset:CIFAR100S:PARAMS": {"Subset": 7},

        "Model": "ViT_B",
        "Model:PARAMS": {
            "PreTrained": True,
            "PreTrained:PARAMS": {"BackboneLRMultiplier": -1},
            "BatchNorm": True
        },
        "Model:ViT,TcT:PARAMS": {"AvgTokens": True},

        "CustomReLU": True,
        "CustomReLU:PARAMS": {"Kn": 0.236, "Kp": 1.0},

        "FeatureKD": True,
        "FeatureKD:PARAMS": {"key1": "val"},

        "LayerMappingMethod": "One2One",
        "LayerMappingMethod:PreDefined:PARAMS": {"LayerMap": "val", "LayerMapWeights": "val"},

        "NumRuns": 3,

        ":": ["NumRuns"]
    }
    dFlatCfg, dModifyCfg = SplitCJSON(dTest)

    Cfg = IConfig(dFlatCfg)

    print(dFlatCfg)
    print(ParseCJSON(dFlatCfg, dModifyCfg))

    dTest = {
        "k1": 1,
        "k2": 2,
        "k3": 3
    }

    dFlatCfg, dModifyCfg = SplitCJSON(dTest)
    print(dFlatCfg, dModifyCfg)