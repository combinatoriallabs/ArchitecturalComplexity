'''
Code written by Nicholas J. Cooper.
Released under the MIT license, see the GitHub page for the full legal boilerplate.
tldr: you freely can do whatever you like with this code so long as this message is retained and you cite the GitHub: https://github.com/Thegolfingocto/ArchitecturalComplexity
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
'''



#library deps
import torch
from torchvision.transforms import v2
import tqdm
import copy
import math
from functools import partial

#Arch deps
import Arch.Models.IModel as IModel
from Arch.Models import *
from Arch.Models.ModelUtils import *

import Arch.Datasets.IDataset as IDataset
from Arch.Datasets import *

from Arch.ITrainer import ITrainer
from Arch.Utils.Utils import *
from Arch.Logger import *

class ImageTrainer(ITrainer):
    def __init__(self, strCacheDir: str = "", dConfig: dict = {}, bStartLog: bool = False):
        print("Initializing ImageTrainer...")
        super().__init__(strCacheDir, dConfig, bStartLog)

        self.SetEvalMetric(iIdx = 1, bHB = True) #set test accuracy (higher better) as the reference metric for best model tracking
        
        return
    
    def PrintConfigSummary(self, dTempCfg: dict = None) -> None:
        if dTempCfg is None: dTempCfg = self.dCfg
        iDS = dTempCfg["DownSample"]
        strResolution = "--> {}x{}".format(iDS, iDS) if iDS > 0 else ""
        printf("Model: {}, Dataset: {} {}".format(dTempCfg["Model"], dTempCfg["Dataset"], strResolution), NOTICE)
        if "PreTrained" in dTempCfg.keys() and dTempCfg["PreTrained"]:
            strB = "Frozen Backbone" if self.GetValue("BackboneLRMultiplier") == -1 else "Backbone LR Mult: " + str(self.GetValue("BackboneLRMultiplier"))
            printf("Imagenet Pretrained, " + strB)
        printf(f"Optimizer: {dTempCfg["Optimizer"]}")
        printf(f"    Learning Rate: {dTempCfg["LearningRate"]}")
        printf(f"    Schedule: {dTempCfg["LRScheduler"]}")
        printf(f"    Batch Size: {dTempCfg["BatchSize"]}")
        if dTempCfg["LRScheduler"] == "MultiStep":
            printf("Steps: {}, Step Mult: {}".format(dTempCfg["LRSteps"], dTempCfg["LRStepMult"]))
        if dTempCfg.get("DataAugmentation"):
            printf("w/ Data Augmentation")
            if dTempCfg["RandAug"]: printf("    RandAug")
            if dTempCfg["DataAugOnTest"]: printf("    Aug. on Test")
        if dTempCfg.get("SubSample") < 1.0: printf("w/ {:.2f}% of Training Data".format(100 * dTempCfg.get("SubSample")))
        printf("Runs: {}, Epochs: {}".format(dTempCfg["NumRuns"], dTempCfg["NumEpochs"]), NOTICE)
        return

    def InitDataset(self, strData: str = None) -> IDataset:
        if strData is None: strData = self.GetValue("Dataset")

        dsData = None
        iBS = self.GetValue("BatchSize")
        strNorm = self.GetValue("Normalization")
        iDS = self.GetValue("DownSample")

        if strData == "CIFAR10": dsData = CIFAR10.CIFAR10(iBatchSize = iBS, strNormalization = strNorm, iDownSample = iDS)
        elif "CIFAR100" in strData: dsData = CIFAR100.CIFAR100(iBatchSize = iBS, strNormalization = strNorm, iDownSample = iDS, bFineLabels = strData[-1] == "F")
        elif strData == "TinyImagenet": dsData = TinyImagenet.TinyImagenet(iBatchSize = iBS, strNormalization = strNorm, iDownSample = iDS)
        elif strData == "Imagenet": dsData = Imagenet.Imagenet(iBatchSize = iBS, iDownSample = iDS)

        elif strData == "StanfordCars": dsData = StanfordCars.StanfordCars(iBatchSize = iBS, strNormalization = strNorm, iDownSample = iDS)
        elif strData == "CUB200": dsData = CUB200.CUB200(iBatchSize = iBS, strNormalization = strNorm, iDownSample = iDS)

        elif strData == "EuroSAT": dsData = EuroSAT.EuroSAT(iBatchSize = iBS, strNormalization = strNorm, iDownSample = iDS)
        elif strData == "PatternNet": dsData = PatternNet.PatternNet(iBatchSize = iBS, strNormalization = strNorm, iDownSample = iDS)

        elif strData == "OxfordPets": dsData = OxfordPets.OxfordPets(iBatchSize = iBS, strNormalization = strNorm, iDownSample = iDS)

        #allow trainers derived from this template to simply add extra datasets w/o copy-pasting the whole mess that is the above
        if dsData is None and self.HasMethod("LoadDatasetExt"): dsData = self.LoadDatasetExt()

        if dsData is None: printf("Failed to load dataset!", ERROR)

        return dsData

    def LoadDataset(self, strData: str = None) -> IDataset:
        dsData = self.InitDataset(strData)

        dsData.ILoad()
        dsData.Normalize()

        if self.GetValue("DataAugmentation"):
            #v2.RandomPerspective(distortion_scale = 0.5, p = 0.5),
            if self.GetValue("RandAug"):
                '''
                The below is essentially a condensed version of RandAugment from: https://arxiv.org/pdf/1909.13719
                Posterize is not included to prevent re-casting tensors to uint8 image format
                perspective and crop and used for shear and translate, respectively.
                This seems to be a *stronger* version of augmentation than the built-in RandAug, perhaps too strong
                '''
                # vecAugs = [v2.RandomRotation(degrees = (0, 30)),
                #            v2.RandomHorizontalFlip(p = 1.0),
                #            v2.RandomVerticalFlip(p = 1.0),
                #            v2.RandomAdjustSharpness(sharpness_factor = 2.0, p = 1.0),
                #            v2.RandomSolarize(0.75, p = 1.0),
                #            v2.RandomAutocontrast(p = 1.0),
                #            v2.RandomEqualize(p = 1.0),
                #            v2.RandomCrop(dsData.Shape()[1], padding = dsData.Shape()[1] // 8),
                #            v2.RandomPerspective(distortion_scale = 0.1, p = 1.0),]
                # vecRand = [v2.RandomChoice(vecAugs), v2.RandomChoice(vecAugs)]
                # dsData.SetAugmentation(v2.RandomApply(vecRand, p = 0.5),
                #                         bDataAugOnTest = self.GetValue("DataAugOnTest"))

                vecAugs = [
                    v2.Lambda(partial(torch.multiply, other = dsData.tStd + dsData.fEpsilon)),
                    v2.Lambda(partial(torch.add, other = dsData.tMean)),
                    v2.RandAugment(),
                    v2.Lambda(partial(torch.subtract, other = dsData.tMean)),
                    v2.Lambda(partial(torch.divide, other = dsData.tStd + dsData.fEpsilon))
                ]
                #v2.ToDtype(torch.uint8),     
                #v2.ToDtype(torch.float32),
                dsData.SetAugmentation(v2.Compose(vecAugs))
            else:
                #v2.RandomVerticalFlip(p=0.5),
                #Note: adding vertical flips changes things *a lot*
                dsData.SetAugmentation(v2.Compose([
                                        v2.RandomCrop(dsData.Shape()[1], padding = 4),
                                        v2.RandomHorizontalFlip(p=0.5),]),
                                        bDataAugOnTest = self.GetValue("DataAugOnTest"))
            
        if self.GetValue("SubSample"): dsData.SubSample(self.GetValue("SubSample"))

        return dsData
    
    def LoadModel(self) -> torch.nn.Module:
        tModel = None
        if self.dsData is None: self.dsData = self.InitDataset()

        iCls = self.dsData.Classes()
        strDS = self.GetValue("Dataset")
        strM = self.GetValue("Model")
        bPT = self.GetValue("PreTrained")
        bBN = self.GetValue("BatchNorm")
        bAT = self.GetValue("AvgTokens")

        bMaxPool = strDS in ["TinyImagenet", "StanfordCars", "CUB_200", "Imagenet"]

        bFS = False
        iSz = 32
        iP = 4
        if strDS == "TinyImagenet":
            iSz = 64
            iP = 8
        elif strDS in ["StanfordCars", "CUB_200", "Imagenet"]:
            iSz = 224
            iP = 16
        elif strDS == "StanfordCarsSmall":
            iSz = 112
            iP = 8

        if self.GetValue("DownSample") > 0: iSz = self.GetValue("DownSample")

        if bPT and strM not in ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", 
                                "VGG11", "VGG13", "VGG16", "VGG19", "ViT_B",
                                "MobileNetV2", "ShuffleNetV2", "SWIN_T"]:
            printf("Warning! Model {} does not support pre-training! Double check Config['Model']")
            return None
        
        #Mobile/shuffle net
        if strM == "MobileNetV2":
            tModel = MobileNetV2.mobilenet_v2(iCls = iCls, iSz = iSz, bImNPreTrained = bPT)
        elif strM == "MobileNetV4":
            tModel = MobileNetV4.MobileNetV4("MobileNetV4ConvSmall", iCls = iCls)
        elif strM == "ShuffleNetV2":
            tModel = ShuffleNetV2.ShuffleNetV2(n_class = iCls, input_size = iSz, bImNPreTrained = bPT)

        #Resnets
        #'mini' Resnets
        if strM == "MiniResNet":
            tModel = Resnet.MiniResNet(iC = iCls)
        elif strM == "MiniResNetX2":
            tModel = Resnet.MiniResNetX2(iC = iCls)
        
        #'normal' Resnets
        elif strM == "ResNet9":
            tModel = Resnet.resnet9(batch_norm=bBN, num_classes = iCls, bInitialMaxPool = bMaxPool,
                             bImNet = strDS == "Imagenet", kn = self.GetValue("Kn") if self.GetValue("UseCustomReLU") else None,
                                                                              kp = self.GetValue("Kp") if self.GetValue("UseCustomReLU") else None)
        elif strM == "ResNet18":
            tModel = Resnet.resnet18(batch_norm=bBN, num_classes = iCls, bInitialMaxPool = bMaxPool,
                              bImNet = strDS == "Imagenet", bImNPreTrained = bPT)
        elif strM == "ResNet34":
            tModel = Resnet.resnet34(batch_norm=bBN, num_classes = iCls, bInitialMaxPool = bMaxPool,
                              bImNPreTrained = bPT, bImNet = strDS == "Imagenet")
        elif strM == "ResNet50":
            tModel = Resnet.resnet50(batch_norm=bBN, num_classes = iCls, bInitialMaxPool = bMaxPool,
                              bImNPreTrained = bPT, bImNet = strDS == "Imagenet")
        elif strM == "ResNet101":
            tModel = Resnet.resnet101(batch_norm=bBN, num_classes = iCls, bInitialMaxPool = bMaxPool,
                               bImNPreTrained = bPT, bImNet = strDS == "Imagenet")
        elif strM == "ResNet152":
            tModel = Resnet.resnet152(batch_norm=bBN, num_classes = iCls, bInitialMaxPool = bMaxPool,
                               bImNPreTrained = bPT, bImNet = strDS == "Imagenet")
            
        #half-channel versions
        elif strM == "ResNet9H":
            tModel = Resnet.resnet9H(batch_norm=bBN, num_classes = iCls, bInitialMaxPool = bMaxPool
                                    , bImNet = strDS == "Imagenet")
        elif strM == "ResNet18H":
            tModel = Resnet.resnet18H(batch_norm=bBN, num_classes = iCls, bInitialMaxPool = bMaxPool
                                    , bImNet = strDS == "Imagenet")
        elif strM == "ResNet34H":
            tModel = Resnet.resnet34H(batch_norm=bBN, num_classes = iCls, bInitialMaxPool = bMaxPool
                                    , bImNet = strDS == "Imagenet")
        elif strM == "ResNet50H":
            tModel = Resnet.resnet50H(batch_norm=bBN, num_classes = iCls, bInitialMaxPool = bMaxPool
                                    , bImNet = strDS == "Imagenet")
        elif strM == "ResNet101H":
            tModel = Resnet.resnet101H(batch_norm=bBN, num_classes = iCls, bInitialMaxPool = bMaxPool
                                    , bImNet = strDS == "Imagenet")
        elif strM == "ResNet152H":
            tModel = Resnet.resnet152H(batch_norm=bBN, num_classes = iCls, bInitialMaxPool = bMaxPool
                                    , bImNet = strDS == "Imagenet")
            

        #Quarter channel and below variants
        elif strM == "ResNet34Q":
            tModel = Resnet.resnet34Q(batch_norm=bBN, num_classes = iCls, bInitialMaxPool = bMaxPool
                                    , bImNet = strDS == "Imagenet")
        elif strM == "ResNet34S":
            tModel = Resnet.resnet34S(batch_norm=bBN, num_classes = iCls, bInitialMaxPool = bMaxPool
                                    , bImNet = strDS == "Imagenet")
        elif strM == "ResNet34S2":
            tModel = Resnet.resnet34S2(batch_norm=bBN, num_classes = iCls, bInitialMaxPool = bMaxPool
                                    , bImNet = strDS == "Imagenet")
        elif strM == "ResNet34QQ":
            tModel = Resnet.resnet34QQ(batch_norm=bBN, num_classes = iCls, bInitialMaxPool = bMaxPool
                                    , bImNet = strDS == "Imagenet")
        elif strM == "ResNet34SS":
            tModel = Resnet.resnet34SS(batch_norm=bBN, num_classes = iCls, bInitialMaxPool = bMaxPool
                                    , bImNet = strDS == "Imagenet")
        elif strM == "ResNet34QQQ":
            tModel = Resnet.resnet34QQQ(batch_norm=bBN, num_classes = iCls, bInitialMaxPool = bMaxPool
                                    , bImNet = strDS == "Imagenet")
        elif strM == "ResNet34QQQQ":
            tModel = Resnet.resnet34QQQQ(batch_norm=bBN, num_classes = iCls, bInitialMaxPool = bMaxPool
                                    , bImNet = strDS == "Imagenet")

            
        #low-resolution tweaked versions from: https://github.com/chenyaofo/pytorch-cifar-models/blob/master/pytorch_cifar_models/resnet.py
        elif strM == "ResNet7":
            tModel = Resnet.resnet7C(batch_norm=bBN, num_classes = iCls, bInitialMaxPool = bMaxPool)
        elif strM == "ResNet20":
            tModel = Resnet.resnet20C(batch_norm=bBN, num_classes = iCls, bInitialMaxPool = bMaxPool)
        elif strM == "ResNet32":
            tModel = Resnet.resnet32C(batch_norm=bBN, num_classes = iCls, bInitialMaxPool = bMaxPool)
        elif strM == "ResNet44":
            tModel = Resnet.resnet44C(batch_norm=bBN, num_classes = iCls, bInitialMaxPool = bMaxPool)
        elif strM == "ResNet56":
            tModel = Resnet.resnet56C(batch_norm=bBN, num_classes = iCls, bInitialMaxPool = bMaxPool)
        
        #VGGs
        if strM == "VGG4":
            tModel = VGG.vgg4(batch_norm=bBN, num_classes = iCls, bFullSize = bFS)
        elif strM == "VGG7":
            tModel = VGG.vgg7(batch_norm=bBN, num_classes = iCls, bFullSize = bFS)
        elif strM == "VGG11":
            tModel = VGG.vgg11(batch_norm=bBN, num_classes = iCls,
                           bImNPreTrained = bPT, bFullSize = bFS)
        elif strM == "VGG13":
            tModel = VGG.vgg13(batch_norm=bBN, num_classes = iCls,
                           bImNPreTrained = bPT, bFullSize = bFS)
        elif strM == "VGG16":
            tModel = VGG.vgg16(batch_norm=bBN, num_classes = iCls,
                           bImNPreTrained = bPT, bFullSize = bFS)
        elif strM == "VGG19":
            tModel = VGG.vgg19(batch_norm=bBN, num_classes = iCls,
                           bImNPreTrained = bPT, bFullSize = bFS)
        
        #Vision transformers
        if strM == "ViT_ETTTL":
            tModel = ViT.VisionTransformer(iSz, iP, 3, 16, 4, 128, 384,
                                     dropout=0.0, bAvgTokens = bAT, num_classes = iCls, representation_size = None)
        elif strM == "ViT_ETTT":
            tModel = ViT.VisionTransformer(iSz, iP, 3, 8, 4, 128, 384,
                                     dropout=0.0, bAvgTokens = bAT, num_classes = iCls, representation_size = None)
        elif strM == "ViT_ETTL":
            tModel = ViT.VisionTransformer(iSz, iP, 3, 16, 6, 192, 576,
                                     dropout=0.0, bAvgTokens = bAT, num_classes = iCls, representation_size = None)
        elif strM == "ViT_ETT":
            tModel = ViT.VisionTransformer(iSz, iP, 3, 8, 6, 192, 576,
                                     dropout=0.0, bAvgTokens = bAT, num_classes = iCls, representation_size = None)
        elif strM == "ViT_ET":
            tModel = ViT.VisionTransformer(iSz, iP, 3, 8, 8, 256, 768,
                                     dropout=0.0, bAvgTokens = bAT, num_classes = iCls, representation_size = None)
        elif strM == "ViT_ETL":
            tModel = ViT.VisionTransformer(iSz, iP, 3, 16, 8, 256, 768,
                                     dropout=0.0, bAvgTokens = bAT, num_classes = iCls, representation_size = None)
        elif strM == "ViT_T": #A tiny version of a ViT
            tModel = ViT.VisionTransformer(iSz, iP, 3, 8, 8, 384, 1024,
                                     dropout=0.0, bAvgTokens = bAT, num_classes = iCls, representation_size = None)
        elif strM == "ViT_S": #A small version of a ViT
            tModel = ViT.VisionTransformer(iSz, iP, 3, 12, 12, 576, 2304,
                                     dropout=0.0, bAvgTokens = bAT, num_classes = iCls, representation_size = None)
        elif strM == "ViT_B": #Same as ViT_Base from original ViT paper: https://arxiv.org/pdf/2010.11929.pdf
            tModel = ViT.VisionTransformer(iSz, iP, 3, 12, 12, 768, 3072,
                                     dropout=0.0, bAvgTokens = bAT, num_classes = iCls, representation_size = None)
            if bPT: tModel.LoadWeights("vit_b_16.pth")
        elif strM == "ViT_BH":
            tModel = ViT.VisionTransformer(iSz, iP, 3, 12, 12, 384, 1536,
                                     dropout=0.0, bAvgTokens = bAT, num_classes = iCls, representation_size = None)
        elif strM == "ViT_BQ":
            tModel = ViT.VisionTransformer(iSz, iP, 3, 12, 12, 192, 768,
                                     dropout=0.0, bAvgTokens = bAT, num_classes = iCls, representation_size = None)
        elif strM == "ViT_BQS":
            tModel = ViT.VisionTransformer(iSz, iP, 3, 12, 12, 144, 576,
                                     dropout=0.0, bAvgTokens = bAT, num_classes = iCls, representation_size = None)
        elif strM == "ViT_BQQ":
            tModel = ViT.VisionTransformer(iSz, iP, 3, 12, 12, 96, 384,
                                     dropout=0.0, bAvgTokens = bAT, num_classes = iCls, representation_size = None)
        elif strM == "ViT_BQQQ":
            tModel = ViT.VisionTransformer(iSz, iP, 3, 12, 12, 48, 192,
                                     dropout=0.0, bAvgTokens = bAT, num_classes = iCls, representation_size = None)
        
        #SWIN transformers
        elif strM == "SWIN_T":
            tModel = SWIN.SwinTransformer(iSz, 4, num_classes = iCls, window_size = self.GetValue("WindowSize"))
            if bPT: tModel.LoadWeights("swin_t.pth")
        elif strM == "SWIN_T2":
            tModel = SWIN.SwinTransformer(iSz, 4, num_classes = iCls, window_size = self.GetValue("WindowSize"), embed_dim = 60)
        elif strM == "SWIN_TH":
            tModel = SWIN.SwinTransformer(iSz, 4, num_classes = iCls, window_size = self.GetValue("WindowSize"), embed_dim = 48)
        elif strM == "SWIN_TH2":
            tModel = SWIN.SwinTransformer(iSz, 4, num_classes = iCls, window_size = self.GetValue("WindowSize"), embed_dim = 36)
        elif strM == "SWIN_TQ":
            tModel = SWIN.SwinTransformer(iSz, 4, num_classes = iCls, window_size = self.GetValue("WindowSize"), embed_dim = 24)
        elif strM == "SWIN_TQQ":
            tModel = SWIN.SwinTransformer(iSz, 4, num_classes = iCls, window_size = self.GetValue("WindowSize"), embed_dim = 12)
        elif strM == "SWIN_TQQQ":
            tModel = SWIN.SwinTransformer(iSz, 4, num_classes = iCls, window_size = self.GetValue("WindowSize"), embed_dim = 6)

        #PolyNets
        elif strM == "PolyNetV2_T": tModel = PolyNet.PolyNetV2_T(iChIn = 3, iSz = iSz, iCls = iCls)
        elif strM == "PolyNetV2_S": tModel = PolyNet.PolyNetV2_S(iChIn = 3, iSz = iSz, iCls = iCls)

        
        #allow trainers derived from this template to simply add extra models w/o copy-pasting the whole mess that is the above
        if tModel is None and self.HasMethod("LoadModelExt"): tModel = self.LoadModelExt()

        if tModel is None:
            printf("Invalid Model {} passed in config!".format(strM), ERROR)
        
        return IModel.IModel(tModel).to(self.device) #maintain an upper level model class for the main trainer
    

    def LoadOptimizer(self) -> any:
        #setup the parameter groups if necessary
        if self.GetValue("PreTrained"):
            lParams = []
            idx = 0
            for key in self.dModules.keys():
                if key == "backbone":
                    if self.GetValue("BackboneLRMultiplier") == -1:
                        for p in self.dModules[key].parameters(): p.requires_grad = False #disable grad tracking
                        continue
                    else:
                        lParams.append({"params": self.dModules[key].parameters(), "lr": self.GetValue("LearningRate") * self.GetValue("BackboneLRMultiplier")})
                        self.iBackboneGroupIdx = idx
                else:
                    lParams.append({"params": self.dModules[key].parameters(), "lr": self.GetValue("LearningRate")})
                idx += 1

        fWD = self.GetValue("WeightDecay")

        if self.GetValue("Optimizer") == "SGD":
            if self.GetValue("PreTrained"):
                return torch.optim.SGD(lParams, momentum = 0.9, weight_decay = fWD if fWD >= 0 else 1e-4)
            else:
                return torch.optim.SGD(self.dModules.parameters(), lr = self.GetValue("LearningRate"), momentum = 0.9, weight_decay = fWD if fWD >= 0 else 1e-4)
        elif self.GetValue("Optimizer") == "AdamW":
            if self.GetValue("PreTrained"):
                return torch.optim.AdamW(lParams, weight_decay = fWD if fWD >= 0 else 1e-2)
            else:
                return torch.optim.AdamW(self.dModules.parameters(), lr=self.GetValue("LearningRate"), weight_decay = 0.01) #0.1 if "ViT" in self.GetValue("Model") else 
        elif self.GetValue("Optimizer") == "Adam":
            if self.GetValue("PreTrained"):
                return torch.optim.Adam(lParams, weight_decay = 0) #hardcode WD = 0 b/c otherwise just use AdamW
            else:
                return torch.optim.Adam(self.dModules.parameters(), lr=self.GetValue("LearningRate"), weight_decay = 0)
        else:
            printf("Invalid optimizer!", ERROR)
        return None
    

    def LoadLossFcn(self) -> any:
        return torch.nn.CrossEntropyLoss()
    

    def LoadScheduler(self) -> any:
        if self.GetValue("LRScheduler") == "OneCycle":
            N = self.dsData.Size()
            if self.GetValue("PreTrained"):
                lMaxLR = [self.GetValue("LearningRate") for _ in range(len(self.tOpt.param_groups))]
                if self.GetValue("BackboneLRMultiplier") > 0:
                    lMaxLR[self.iBackboneGroupIdx] *= self.GetValue("BackboneLRMultiplier")

                topt = torch.optim.lr_scheduler.OneCycleLR(self.tOpt, lMaxLR, epochs = self.GetValue("NumEpochs"), 
                                                           pct_start = self.GetValue("PctStart"), steps_per_epoch = int((N / self.GetValue("BatchSize")) + 1))
            else:
                topt = torch.optim.lr_scheduler.OneCycleLR(self.tOpt, self.GetValue("LearningRate"), epochs = self.GetValue("NumEpochs"), 
                                                           pct_start = self.GetValue("PctStart"), steps_per_epoch = int((N / self.GetValue("BatchSize")) + 1))

            if self.iSkipEpochs > 0:
                #skip the scheduler ahead if we're resuming from checkpoints
                for _ in range(self.iSkipEpochs):
                    for _ in range(int((N / self.GetValue("BatchSize")) + 1)):
                        topt.step()

            return topt
        
        elif self.GetValue("LRScheduler") == "Exp":
            return torch.optim.lr_scheduler.ExponentialLR(self.tOpt, self.GetValue("ExpPower"))
        elif self.GetValue("LRScheduler") == "MultiStep":
            return torch.optim.lr_scheduler.MultiStepLR(self.tOpt, milestones = self.GetValue("LRSteps"), gamma = self.GetValue("LRStepMult"))

        elif self.GetValue("LRScheduler") != "None":
            printf("Invalid LRScheduler {}".format(self.GetValue("LRScheduler")), ERROR)
        
        return None
    
    def LoadComponents(self) -> None:
        #Add any extra stuff here!
        return
    
    def Eval(self, dsData: IDataset = None, vecSkipLayers: list[int] = []) -> dict:
        if dsData is None:
            if not self.dsData.Loaded(): self.ILoadDataset()
            dsData = self.dsData #this is here for easy eval on OOD datasets
        if self.tModel is None:
            self.LoadTrainedModel()
        if self.tLossFcn is None:
            self.ILoadLossFcn()

        #put model in test mode
        self.tModel.eval()

        with torch.no_grad():
            acc = 0
            avg_loss = 0
            iN = math.ceil(dsData.Size("test") / dsData.iBatchSize)

            for i in tqdm.tqdm(range(iN)):
                iE = min(((i+1)*dsData.iBatchSize, dsData.Size("test")))
                idx = torch.arange(i*dsData.iBatchSize, iE, 1)
                x, y = dsData.GetSamples(idx, "test")
                x = x.to(self.device)
                y = y.to(self.device)

                y_hat = self.tModel(x, vecSkipLayers = vecSkipLayers)

                avg_loss += self.tLossFcn(y_hat, y).item() * y.shape[0]
                acc += torch.sum(torch.where(torch.argmax(y_hat, dim = 1) == y, 1, 0)).item()

        return {
                "TestLoss": avg_loss / dsData.Size("test"),
                "TestAcc": acc / dsData.Size("test")
                }
    
    def Train(self) -> dict:
        self.tModel.train()

        loss = 0
        iN = self.dsData.Size() // self.dsData.iBatchSize

        #rng = np.random.default_rng()
        #vecIdx = rng.permutation(N * self.GetValue("BatchSize"))
        vecIdx = torch.randperm(self.dsData.Size())

        for i in tqdm.tqdm(range(iN)):
            idx = vecIdx[i*self.dsData.iBatchSize:(i+1)*self.dsData.iBatchSize]
            x, y = self.dsData.GetSamples(idx, "train")

            x = x.to(self.device)
            y = y.to(self.device)

            y_hat = self.tModel(x)
            ce_loss = self.tLossFcn(y_hat, y)
            total_loss = ce_loss
            if total_loss != total_loss:
                print("NaNs detected!", total_loss, y_hat, y)
                input()
            loss += total_loss.item()

            self.tOpt.zero_grad()
            total_loss.backward()
            self.tOpt.step()
            if self.GetValue("LRScheduler") == "OneCycle": self.tSch.step()

        if self.GetValue("LRScheduler") in ["Exp", "MultiStep"]: self.tSch.step()

        return {"TrainLoss": loss / iN}
    


    #This impl hardcodes "good" sample sizes for some image datasets.
    def GenPerLayerFeatures(self, vecLt: list[int] = None, iBatchSize: int = 500) -> None:
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
            strPath = strDir + self.GenFeatureString(iL) + ".pkl"
            #print(strPath)
            if os.path.exists(strPath):
                vecR.append(iL)
        for iR in vecR: vecL.remove(iR)

        strLabelPath = strDir + self.GenFeatureStringLabel() + ".pkl"
                
        if len(vecL) == 0 and os.path.exists(strLabelPath): return

        if not self.dsData.Loaded(): self.ILoadDataset() #only load dataset if necessary

        bDisableFirstPool = False

        if self.GetValue("Dataset") == "TinyImagenet":
            X, Y = self.dsData.GetRandomSubset(5000, strSplit = "train")
        elif self.GetValue("Dataset") == "StanfordCars":
            X, Y = self.dsData.GetRandomSubset(5000, strSplit = "train")
        elif self.GetValue("Dataset") == "Imagenet":
            X, Y = self.dsData.GetRandomSubset(5000, strSplit = "train")
        else:
            X, Y = self.dsData.X, self.dsData.Y

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
                    with open(strDir + self.GenFeatureString(cnt) + ".pkl", "wb") as f:
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
