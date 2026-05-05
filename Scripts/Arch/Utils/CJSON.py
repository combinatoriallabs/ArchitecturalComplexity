'''
Code written by Nicholas J. Cooper.
Released under the MIT license, see the GitHub page for the full legal boilerplate.
tldr: you freely can do whatever you like with this code so long as this message is retained and you cite the GitHub: https://github.com/combinatoriallabs/ArchitecturalComplexity
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
'''

try:
    from Arch.Utils.Utils import *
    from Arch.Logger import *
except:
    from Utils import *

def ParseCJSON(dTempCfg: dict, dModifyCfg: dict) -> dict:
    '''
    Modifies dTempCfg based on the CJSON Syntax present in dModifyCfg. CJSON example:
    {
    "key1": "value1",
    "key2": "value2",
    "key1:Params": {"subkey1": "subvalue1"} #indicates that subkey1 is only used when value1 evaluates to True when cast as a bool.
    "key1:q:Params: {"subkey2": "subvalue2"} #indicates that subkey2 is only used when q is either equal to or a substring of key1.
    "key1:!q:Params: {"subkey3": "subvalue3"} #indicates that subkey3 is only used when NOT(q is either equal to or a substring of key1).
    "key1:q1,q2:Params": {"subkey4": "subvalue4"} #indicates that subkey4 is only used when either q1 OR q2 are equal to (or substrings of) key1.
    "key1:q3;q4:Params": {"subkey5": "subvalue5"} #indicates that subkey5 is only used when q1 AND q2 are equal to (or substrings of) key1.

    "key1,key2:Params": {"subkey6": "subvalue6} #indicates that subkey6 is only used when either value1 or value2 evaluates to True when cast as a bool.
    "key1;key2:Params": {"subkey6": "subvalue6} #indicates that subkey6 is only used when value1 and value2 evaluate to True when cast as a bool.
    
    ":": ["subkeyx", subkeyy", ...] #indicates a list of always ignored keys.
    }
    '''

    vecConditions = list(dModifyCfg.keys())
    #print("Top of ParseCJSON:", dTempCfg)

    for strCondition in vecConditions:
        bFlip = "!" in strCondition
        vecArgs = [s.replace("!", "") for s in strCondition.split(":")]
        strKey = vecArgs[0]
        #print("Condition:", strCondition, "Key:", strKey)

        bDelete = True
        if len(vecArgs) == 2:
            if "," in vecArgs[0]:
                vecKeys = vecArgs[0].split(",")
                for key in vecKeys:
                    if dTempCfg[key]:
                        bDelete = False
                        break
            elif ";" in vecArgs[0]:
                vecKeys = vecArgs[0].split(";")
                bDelete = False
                for key in vecKeys:
                    if not dTempCfg[key]:
                        bDelete = True
                        break
            else: bDelete = (strCondition == ":") or (not dTempCfg[strKey])
        elif "," in vecArgs[1]:
            vecVals = vecArgs[1].split(",")
            for val in vecVals:
                if (isinstance(val, str) and val in dTempCfg[strKey]) or (not isinstance(val, str) and val == dTempCfg[strKey]):
                    bDelete = False
                    break
        elif ";" in vecArgs[1]:
            vecVals = vecArgs[1].split(";")
            bDelete = False
            for val in vecVals:
                if (isinstance(val, str) and val not in dTempCfg[strKey]) or (not isinstance(val, str) and val != dTempCfg[strKey]):
                    bDelete = True
                    break
        else: bDelete = (isinstance(vecArgs[1], str) and vecArgs[1] not in dTempCfg[strKey]) or (not isinstance(vecArgs[1], str) and vecArgs[1] != dTempCfg[strKey])

        if bFlip: bDelete = not bDelete

        if bDelete:
            vecRemoveKeys = dModifyCfg[strCondition]
            if isinstance(vecRemoveKeys, dict): vecRemoveKeys = list(DictFlatten(vecRemoveKeys).keys())
            for strRemoveKey in vecRemoveKeys:
                if strRemoveKey in dTempCfg.keys(): del dTempCfg[strRemoveKey]
        else:
            dTempCfg = ParseCJSON(dTempCfg, DictFilterByKey(":", dModifyCfg[strCondition], bRecursive = False))

    return dTempCfg

def SplitCJSON(dTempCfg: dict) -> tuple[dict, dict]:
    dFlatCfg = DictFlatten(dTempCfg)
    if ":" in dFlatCfg.keys(): del dFlatCfg[":"]
    dModifyCfg = DictFilterByKey(":", dTempCfg, bRecursive = False)

    for strKey in dFlatCfg.keys():
        if ":" in strKey:
            printf("ERROR! Found CJSON ':' key {} without a corresponding {{}}".format(strKey))
            quit()
    for strKey in dModifyCfg.keys():
        if strKey == ":": continue
        if "Params" not in strKey:
            printf("ERROR! Found CJSON key {} without Params".format(strKey))
            quit()
        if not isinstance(dModifyCfg[strKey], dict):
            printf("ERROR! Found CJSON value {} which is not a dict!".format(dModifyCfg[strKey]))
            quit()

    return dFlatCfg, dModifyCfg

if __name__ == "__main__":
    dTest = {
    "Dataset": "TinyImagenet",
    "EmbedDimension": 100,
    "Subset": 7,
    "Model": "MobileNetV2",
    "PreTrained": True,
    "UseCustomReLU": False,
    "BackboneLRMultiplier": 0.01,
    "AvgTokens": True,
    "BatchNorm": True,
    "UseCELoss": True,
    "SeperateLinear": False,
    "Teacher": "ResNet34",
    "VanillaKD": False,
    "Logits": "CIFAR10_Layer30_Logits_T2.pkl",
    "Temperature": 2,
    "FeatureKD": False,
    "StoreFeatures": False,
    "UseCosineLoss": False,
    "UseHuberLoss": False,
    "TeacherLayers": [
        3,
        7,
        13,
        16
    ],
    "StudentLayers": [
        3,
        6,
        13,
        17
    ],
    "LayerMappingMethod": "One2One",
    "TeacherPowers": [],
    "LayerMap": [
        []
    ],
    "LayerMapWeights": [
        []
    ],
    "StudentPartitions": [],
    "TeacherPartitions": [],
    "AttnArchitecture": [
        256,
        128
    ],
    "AttnInput": "DPS",
    "ProjectionMethod": "LearnedProjector",
    "UseGlobalCenter": True,
    "LandmarkMethod": "CC",
    "LandmarkPostScaleFactor": 1,
    "ProjectorArchitecture": "SingleLayerConv",
    "RelationFunction": "DotProd",
    "CELossWeight": 1,
    "LGLossWeight": 1,
    "LSLossWeight": 1,
    "LearningRate": 0.001,
    "Optimizer": "AdamW",
    "LRScheduler": "OneCycle",
    "MaxLR": 0.0075,
    "BatchSize": 128,
    "NumEpochs": 25,
    "NumRuns": 1,
    "EvalInterval": 1,
    "CheckpointInterval": -1,
    "SaveResult": True,
    "SaveModel": True
    }

    dFormat = {
    "Dataset": "",
    "DownSample": -1,
    "Normalization": "MeanVar",
    "Dataset:CIFAR100S:Params": {"Subset": 7},
    "Dataset:News,DBpedia:Params": {"EmbedDimension": 100},

    "Model": "",
    "PreTrained": False,
    "BatchNorm": True,
    "UseCustomReLU": False,
    "PreTrained:Params": {"BackboneLRMultiplier": -1},
    "Model:Vit,TcT:Params": {"AvgTokens": True},
    "UseCustomReLU:Params": {"Kn": 0.236, "Kp": 1},

    "UseCELoss": True,
    "SeperateLinear": False,

    "VanillaKD,FeatureKD:Params": {"Teacher": ""},

    "VanillaKD": False,
    "Temperature": 4,
    "NormalizeLogits": False,
    "FeatureKD": False,
    "StoreFeatures": False,
    "UseTeacherClassifier": False,
    "LearnTeacherClassifier": False,
    "UseCosineLoss": False,
    "UseHuberLoss": False,
    "TeacherLayers": [
    ],
    "StudentLayers": [
    ],
    "LayerMappingMethod": "",
    "TeacherPowers": [],
    "LayerMap": [
    []
    ],
    "LayerMapWeights": [
    []
    ],
    "StudentPartitions": [],
    "TeacherPartitions": [],
    "AttnArchitecture": [
    ],
    "AttnInput": "",
    "ProjectionMethod": "",
    "UseGlobalCenter": True,
    "LandmarkMethod": "",
    "LandmarkPostScaleFactor": 1,
    "ProjectorArchitecture": "",
    "PoolMode": True,
    "LearnProjectors": True,
    "RelationFunction": "",
    "CELossWeight": 1,
    "LGLossWeight": 16,
    "LSLossWeight": 1,
    "LearningRate": 0.01,
    "WeightDecay": -1,
    "DataAugmentation": False,
    "ExpPowersetErasing": False,
    "AugmentTeacher": True,
    "Optimizer": "",
    "LRSteps": [],
    "LRStepMult": 0.1,
    "ExpPower": 0.9,
    "LRScheduler": "",
    "PctStart": 0.3,
    "MaxLR": 0.0075,
    "BatchSize": 128,
    "NumEpochs": 50,
    "NumRuns": 1,
    "EvalInterval": 1,
    "CheckpointInterval": -1,
    "SaveResult": True,
    "SaveModel": True,

    ":": ["EvalInterval", "CheckpointInterval", "SaveModel", "SaveResult", "StoreFeatures"]
    }

    dFlatCfg, dModifyCfg = SplitCJSON(dFormat)

    print(ParseCJSON(dTest, dModifyCfg))