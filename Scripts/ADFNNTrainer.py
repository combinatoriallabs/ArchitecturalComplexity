'''
Code written by Nicholas J. Cooper.
Released under the MIT license, see the GitHub page for the full legal boilerplate.
tldr: you freely can do whatever you like with this code so long as this message is retained and you cite the GitHub: https://github.com/Thegolfingocto/ArchitecturalComplexity
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
'''



#library deps
import torch
import tqdm
import os
import copy
import json

#Arch deps
from Arch.Datasets import *
from Arch.ITrainer import *
from Arch.Templates.ImageTrainer import ImageTrainer
from Arch.Models.IModel import *
from Arch.Models.Misc import NLayerMLP
from TOModels import *
from ADFNN import *
from ADFNNCache import *
from InputLayers import *
from ADFNNImpls import *
from Arch.Models.Resnet import *
from Arch.Models.SWIN import *

class ADFNNTrainer(ImageTrainer):
    def __init__(self, strCacheDir: str = "", dConfig: dict = {}, bStartLog: bool = False, bUseDiskWatchers: bool = False):
        super().__init__(strCacheDir, dConfig, bStartLog)

        self.SetEvalMetric(iIdx = 1, bHB = True) #set test accuracy (higher better) as the reference metric for best model tracking

        #setup extra stuff for ADFNNs
        if "ADFNN" in self.GetValue("Model"):
            strADFNNCacheDir = self.strBaseDir + dConfig["ADFNNCacheDir"]
            if strADFNNCacheDir[-1] != "/": strADFNNCacheDir += "/"

            if not os.path.isdir(strADFNNCacheDir): os.mkdir(strADFNNCacheDir)
            if not os.path.exists(strADFNNCacheDir + "BaselineConfig.cjson"):
                #copy over the ADFNN params
                dCfg = copy.deepcopy(self.dBaselineCfg["Model:ADFNN:Params"])
                dCfg[":"] = ["ADFNNCacheDir", "ADFNNModelID", "ADFNNStacks"] #add hash-ignored keys
                dCfg["ADFNNInputModule:PatchUnfold:Params"] = {"PreConvChannels": -1}
                with open(strADFNNCacheDir + "BaselineConfig.cjson", "w") as f:
                    json.dump(dCfg, f, indent = 2)

            self.cacheADFNN: ADFNNCache = ADFNNCache(strADFNNCacheDir, bUseDiskWatchers = bUseDiskWatchers) #startup the sub-cache
        
        return
    
    def ModifyHashCfg(self, dTempCfg: dict) -> dict:
        if dTempCfg["Model"] not in ["FishCNN", "FishMLP", "FishySimpleCNN", "ResFish9", "ResFish9Q"] and "FishPooling" in dTempCfg.keys() and not dTempCfg["FishPooling"]:
            dTempCfg.pop("FishType", None)

        return dTempCfg
    
    def SetADFNNModelID(self, strID: str, iStacks: int = 1) -> None:
        self.dCfg["ADFNNModelID"] = strID
        self.dCfg["ADFNNStacks"] = iStacks
        self.ResetResult()
        self.ILoadModel()
        self.UpdateCacheMap()

        return
    
    def SetADFNNModel(self, adfnnModel: ADFNN, iStacks: int = 1) -> None:
        strID = self.cacheADFNN.CacheADFNN(adfnnModel)
        return self.SetADFNNModelID(strID, iStacks)
    
    def LookupActivation(self) -> torch.nn.Module:
        strAct = self.GetValue("ActivationFunction").lower()

        if strAct == "none": return None
        elif strAct == "leakyrelu": return torch.nn.LeakyReLU
        elif strAct == "relu": return torch.nn.ReLU
        elif strAct == "gelu": return torch.nn.GELU
        elif strAct == "tanh": return torch.nn.Tanh
        elif strAct == "sigmoid": return torch.nn.Sigmoid

        print("Double check config field 'ActivationFunction'")
        return -1 #this will ensure errors get thrown downstream

    def GetADFNNInputModules(self, sIn: list[int], iImSz: int, iP: int) -> list[torch.nn.Module]:
        vecInputModules = []
        strInputModule = self.GetValue("ADFNNInputModule")
        if strInputModule == "ViT1D":
            iModelDim = ListMinus(sIn, [(iImSz // iP)**2])
            vecInputModules = [ViTInput1D(iImSz, iP, 3, iModelDim)]
        elif strInputModule == "PatchUnfold": vecInputModules = [PatchUnfold(iP, iChannels = self.GetValue("PreConvChannels"))]
        elif strInputModule == "OverlapPatchUnfold": vecInputModules = [OverlapPatchUnfold(iP, iChannels = self.GetValue("PreConvChannels"))]
        elif strInputModule == "ResNet34Stage1":
            bMaxPool = self.GetValue("Dataset") in ["TinyImagenet", "StanfordCars", "CUB_200", "Imagenet"]
            tModel = IModel(resnet34(batch_norm=True, num_classes = 1, bInitialMaxPool = bMaxPool,
                              bImNPreTrained = False, bImNet = False))
            tModel.SetEndLayer(5)
            vecInputModules = [tModel, UnaryOperation(partial(Conv2dUnfold, iK = 5, iS = 1, iP = 2))]
        elif strInputModule == "ResNet34HStage1":
            bMaxPool = self.GetValue("Dataset") in ["TinyImagenet", "StanfordCars", "CUB_200", "Imagenet"]
            tModel = IModel(resnet34H(batch_norm=True, num_classes = 1, bInitialMaxPool = bMaxPool,
                              bImNPreTrained = False, bImNet = False))
            tModel.SetEndLayer(5)
            vecInputModules = [tModel, UnaryOperation(partial(Conv2dUnfold, iK = 5, iS = 1, iP = 2))]
        elif strInputModule == "SWIN_TStage1":
            tSWIN = IModel(tModel = SwinTransformer(iImSz, 4, num_classes = 1, window_size = 4))
            tSWIN.SetEndLayer(4)
            vecInputModules = [tSWIN]

        return vecInputModules
    
    def GetADFNNOutputModules(self, sIn: list[int]) -> list[torch.nn.Module]:
        vecOutputModules = []
        strOutputModule = self.GetValue("ADFNNOutputModule")
        if strOutputModule == "Conv2Downsample":
            vecOutputModules.append(Conv2Downsample(self.GetValue("PostConvChannels")))
        elif strOutputModule == "AvgDim0":
            vecOutputModules.append(UnaryOperation(partial(torch.mean, dim = 1)))

        return vecOutputModules

    def LoadModelExt(self) -> torch.nn.Module:
        tModel = None
        if not self.dsData.Loaded(): self.ILoadDataset()
        
        iCls = self.dsData.Classes()
        strDS = self.GetValue("Dataset")
        bMaxPool = strDS in ["TinyImagenet", "StanfordCars", "CUB_200", "Imagenet"] # 
        strM = self.GetValue("Model")
        vecShape = self.dsData.Shape()
        iSz = int(torch.prod(torch.tensor(vecShape)).item())
        #bPT = self.GetValue("PreTrained")
        bBN = self.GetValue("BatchNorm")
        #bAT = self.GetValue("AvgTokens")
        tmAct = self.LookupActivation()

        iImSz = 32
        iP = 4
        if strDS == "TinyImagenet":
            iImSz = 64
            iP = 8
        elif strDS in ["StanfordCars", "CUB_200", "Imagenet"]:
            iImSz = 224
            iP = 16
        elif strDS == "StanfordCarsSmall":
            iImSz = 112
            iP = 8

        if self.GetValue("DownSample") > 0: iImSz = self.GetValue("DownSample")

        if strM == "ADFNN":
            strID = self.GetValue("ADFNNModelID")
            if strID and len(strID) == 32: tModel = self.cacheADFNN.GetADFNN(strID)
            else: tModel = self.cacheADFNN.GetADFNN(self.dCfg)

            if tModel is None: print("WARNING: Could not find ADFNN Model")
            else:
                vecInputModules = self.GetADFNNInputModules(tModel.sIn, iImSz, iP)
                vecOutputModules = self.GetADFNNOutputModules(tModel.vecSOut[-1])
                tModel = GenADFNN(tModel, self.GetValue("ADFNNStacks"), vecInitModules = vecInputModules, iResidualFq = self.GetValue("ADFNNResidualFq"),
                                                                        vecOutModules = vecOutputModules).to(self.device)
                
                t = torch.randn([1] + vecShape).to(self.device)
                with torch.no_grad(): t = tModel(t)
                iOSz = torch.prod(torch.tensor(t.shape)[1:]).item()
                #print(t.shape, iOSz)
                if iOSz != iCls: tModel = torch.nn.Sequential(tModel, torch.nn.Flatten(), torch.nn.Linear(iOSz, iCls))

        elif strM == "ADFNN_ViT_ETT":
            #tModel = self.cacheADFNN.GetADFNN("ADFNN_ViT_ETT") #TODO: figure this out
            #if tModel is None:
            
            iNumTokens = (iImSz // iP)**2 + 1
            tModel = BuildViT(iNumTokens, 6, 192, 576, 4, iImSz, 3, iP)
            tHead = torch.nn.Linear(192, iCls)
            torch.nn.init.zeros_(tHead.weight)
            torch.nn.init.zeros_(tHead.bias)
            tModel = torch.nn.Sequential(tModel, UnaryOperation(partial(torch.mean, dim = 1)), torch.nn.Flatten(), tHead)

        elif strM == "ADFNN_MLP":
            tModel = BuildMLP(iSz, iSz, iLayers = len(self.GetValue("MLPStructure")) + 1, iCls = iCls)
        
        elif strM == "MLP": tModel = NLayerMLP(iSz, self.GetValue("MLPStructure"), iCls, bNormalizeOutput = False, tmActivation = tmAct)

        elif strM == "ResNet34HStage1":
            bMaxPool = strDS in ["TinyImagenet", "StanfordCars", "CUB_200", "Imagenet"]
            tRN = IModel(resnet34H(batch_norm=True, num_classes = 1, bInitialMaxPool = bMaxPool,
                              bImNPreTrained = False, bImNet = False))
            tRN.SetEndLayer(5)

            tModel = torch.nn.Sequential(
                tRN,
                torch.nn.Flatten(),
                torch.nn.Linear(64 * (iImSz // 2)**2, iCls)
            )

        elif strM == "ResNet34HStage1DSz":
            bMaxPool = strDS in ["TinyImagenet", "StanfordCars", "CUB_200", "Imagenet"]
            tRN = IModel(resnet34H(batch_norm=True, num_classes = 1, bInitialMaxPool = bMaxPool,
                              bImNPreTrained = False, bImNet = False))
            tRN.SetEndLayer(5)

            iDSz = 16 if strDS == "Imagenet" else 4
            tModel = torch.nn.Sequential(
                tRN,
                torch.nn.MaxPool2d(iDSz, iDSz),
                torch.nn.Flatten(),
                torch.nn.Linear(64 * (iImSz // (2 * iDSz * (int(bMaxPool) + 1)))**2, iCls)
            )

        elif strM == "ResNet34HStage1DSz":
            bMaxPool = strDS in ["TinyImagenet", "StanfordCars", "CUB_200", "Imagenet"]
            tRN = IModel(resnet34H(batch_norm=True, num_classes = 1, bInitialMaxPool = bMaxPool,
                              bImNPreTrained = False, bImNet = False))
            tRN.SetEndLayer(5)

            iDSz = 16 if strDS == "Imagenet" else 4
            tModel = torch.nn.Sequential(
                tRN,
                torch.nn.MaxPool2d(iDSz, iDSz),
                torch.nn.Flatten(),
                torch.nn.Linear(64 * (iImSz // (2 * iDSz * (int(bMaxPool) + 1)))**2, iCls)
            )
        elif strM == "ResNet34HStage1DDSz":
            bMaxPool = strDS in ["TinyImagenet", "StanfordCars", "CUB_200", "Imagenet"]
            tRN = IModel(resnet34H(batch_norm=True, num_classes = 1, bInitialMaxPool = bMaxPool,
                              bImNPreTrained = False, bImNet = False))
            tRN.SetEndLayer(5)

            iDSz = 16 if strDS == "Imagenet" else 8
            tModel = torch.nn.Sequential(
                tRN,
                torch.nn.MaxPool2d(iDSz, iDSz),
                torch.nn.Flatten(),
                torch.nn.Linear(64 * (iImSz // (2 * iDSz * (int(bMaxPool) + 1)))**2, iCls)
            )


        elif strM == "SWIN_TStage1":
            tSWIN = IModel(tModel = SwinTransformer(iImSz, 4, num_classes = iCls, window_size = 4))
            tSWIN.SetEndLayer(4)

            tModel = torch.nn.Sequential(
                tSWIN,
                UnaryOperation(partial(torch.mean, dim = 1)),
                torch.nn.Linear(192, iCls)
            )


        return tModel
    

    def Train(self) -> dict:
        self.tModel.train()

        loss = 0
        acc = 0
        iN = self.dsData.Size() // self.dsData.iBatchSize

        vecIdx = torch.randperm(self.dsData.Size())

        for i in tqdm.tqdm(range(iN)):
            idx = vecIdx[i*self.dsData.iBatchSize:(i+1)*self.dsData.iBatchSize]
            x, y = self.dsData.GetSamples(idx, "train")

            x = x.to(self.device)
            y = y.to(self.device)

            y_hat = self.tModel(x)

            acc += torch.sum(torch.where(torch.argmax(y_hat, dim = 1) == y, 1, 0)).item()

            ce_loss = self.tLossFcn(y_hat, y)
            total_loss = ce_loss
            if total_loss != total_loss:
                print("NaNs detected!", total_loss, y_hat, y)
                input()
            l = total_loss.item()
            loss += l

            self.tOpt.zero_grad()
            total_loss.backward()

            self.tOpt.step()
            if self.GetValue("LRScheduler") == "OneCycle": self.tSch.step()

        if self.GetValue("LRScheduler") in ["Exp", "MultiStep"]: self.tSch.step()

        return {"TrainLoss": loss / iN, "TrainAcc": acc / (iN * self.dsData.iBatchSize)}
    

    def GenPerLayerFeatures(self, vecLt: list[int] = None, iBatchSize: int = 500, strSplit: str = "train") -> None:
        if not self.GetValue("RandomInit"): self.LoadTrainedModel()
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

        if self.GetValue("Dataset") == "TinyImagenet":
            X, Y = self.dsData.GetRandomSubset(5000, strSplit = strSplit)
        elif self.GetValue("Dataset") == "StanfordCars":
            X, Y = self.dsData.GetRandomSubset(5000, strSplit = strSplit)
        elif self.GetValue("Dataset") == "Imagenet":
            X, Y = self.dsData.GetRandomSubset(5000, strSplit = strSplit)
        elif strSplit == "train":
            X, Y = self.dsData.X, self.dsData.Y
        else:
            X, Y = self.dsData.Xt, self.dsData.Yt

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