
LEARNING_RATES = {
    "SWIN": 0.001,
    "ResNet": 0.0075,
}

OUTPUT_MODULES = {
    "SWIN": "AvgDim0",
    "ResNet": "None",
}

MAP_DS_TO_GIVEUPTHRESHOLD = {
    "CIFAR10": {
        "ResNet34HStage1": 0.675,
        "SWIN_TStage1": 0.475,
    },

    "CIFAR100F": {
        "ResNet34HStage1": 0.36,
        "SWIN_TStage1": 0.20,
    },

    "TinyImagenet": {
        "ResNet34HStage1": 0.25,
        "SWIN_TStage1": 0.18,
    }
}

MAP_DS_TO_OUTPUTSHAPE = {
    "CIFAR10": {
        "ResNet34HStage1": [16, 16],
        "SWIN_TStage1": [16, 64, 3],
    },

    "CIFAR100F": {
        "ResNet34HStage1": [16, 16],
        "SWIN_TStage1": [16, 64, 3],
    },

    "TinyImagenet": {
        "ResNet34HStage1": [16, 16],
        "SWIN_TStage1": [64, 64, 3],
    }
}


REFERENCE_MODELS = {
    "ResNet34": ["ResNet34", "ResNet34H", "ResNet34Q", "ResNet34S", "ResNet34S2", "ResNet34QQ", "ResNet34SS", "ResNet34QQQ",],
    "SWIN_T": ["SWIN_T", "SWIN_T2", "SWIN_TH", "SWIN_TH2", "SWIN_TQ", "SWIN_TQQ", "SWIN_TQQQ"],
}

MAP_INPUTMODULES_TO_STEMS = {
    "ResNet34HStage1": "ResNet34HStage1DDSz",
    "SWIN_TStage1": "SWIN_TStage1",
}