import json
import os
import getpass

try:
    from Arch.Utils.Utils import *
except:
    from Utils import *


class WeightLoader():
    def __init__(self, strBaseDir: str = None):
        #try to find a path to the datasets
        self.strBaseDir = None
        strCfgPath = os.fsdecode(os.path.realpath(__file__)).replace("WeightLoader.py", "ModelsCfg.json")
        if strBaseDir is not None: self.strBaseDir = strBaseDir
        elif os.path.exists(strCfgPath):
            with open(strCfgPath, "r") as f:
                dCfg = json.load(f)
                if "WeightsPath" in dCfg.keys(): self.strBaseDir = dCfg["WeightsPath"]
        
        #if nothing worked, ask the user what they want to do
        if self.strBaseDir is None:
            strGuessDir = "/home/" + getpass.getuser() + "/ImagenetPretrainedModels/"
            print("WeightLoader could not find a path to the Weights folder and will attempt to use {} instead.".format(strGuessDir))
            if GetInput("Would you like to enter a path now? (Y/X)"):
                self.strBaseDir = input("Enter path to Weights folder: ")
                with open(strCfgPath, "w") as f:
                    json.dump({"WeightsPath": self.strBaseDir}, f)

            else: self.strBaseDir = strGuessDir

    def WeightsPath(self,) -> str: return self.strBaseDir