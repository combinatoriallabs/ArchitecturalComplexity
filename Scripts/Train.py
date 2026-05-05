'''
Code written by Nicholas J. Cooper.
Released under the MIT license, see the GitHub page for the full legal boilerplate.
tldr: you freely can do whatever you like with this code so long as this message is retained and you cite the GitHub: https://github.com/combinatoriallabs/ArchitecturalComplexity
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
'''



from TrainerUtils import *
from Arch.Models.ModelUtils import *

import sys

def main():
    Config = {
        #"ADFNNCacheDir": "ADFNNCache",
        "ADFNNCacheDir": "../Dataset/CIFAR100F/ResNet34H/ADFNNCache",

        "Model": "ADFNN",

        "ADFNNModelID": "3cad4fc28af6344afe380ee3bd9092a7", #f79cca331d95b3d2fe4cc897454f8e11   31764e8111b9cb64840e8046f5650ebc   0b8f4f2f1e620c008deae84b74c1b5ce, d193f86b7e588d6a4f862a14a4a1dc91
        "ADFNNStacks": 1,
        "ADFNNInputModule": "ResNet34HStage1",
        #"ADFNNInputModule": "SWIN_TStage1",
        "ADFNNOutputModule": "None",
        #"ADFNNOutputModule": "AvgDim0",

        #Dataset section
        "Dataset": "CIFAR100F",
        "Normalization": "MeanVar",
        "DownSample": -1,
        "SubSample": 1.0,
        "DataAugmentation": True,
        "RandAug": False,
        "DataAugOnTest": False,

        "Optimizer": "AdamW",
        "LearningRate": 0.0075,
        #"LearningRate": 0.001,
        "WeightDecay": -1,

        "LRScheduler": "OneCycle",
        "PctStart": 0.3,

        "BatchSize": 128,
        "NumEpochs": 50,
        "NumRuns": 1,

        "EvalInterval": 1,
        "CheckpointInterval": -1,

        "SaveModel": True,
    }

    T = ADFNNTrainer("../Trainer/", Config, bStartLog = True)

    #T.FindCachedStuff("Result.json", bPrint = True)

    T.ILoadModel()
    PrintModelSummary(T.tModel)
    print(CountParams(T.tModel))
    #T.IGenHash(bPrint = True)
    T.IDisplayResult(bY = True if (len(sys.argv) == 2 and sys.argv[1].lower() == "-y") else False)


if __name__ == "__main__":
    main()