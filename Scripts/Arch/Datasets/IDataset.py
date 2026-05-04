'''
Code written by Nicholas J. Cooper.
Released under the MIT license, see the GitHub page for the full legal boilerplate.
tldr: you freely can do whatever you like with this code so long as this message is retained and you cite the GitHub: https://github.com/Thegolfingocto/ArchitecturalComplexity
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
'''


import torch
import torchvision
import urllib.request
import getpass
import os
import time
from inspect import ismethod
from typing import Callable

try:
    from Arch.Utils.Utils import *
except:
    from Utils import *

class IDataset:
    def __init__(self, iBatchSize: int, strNormalization: str = "MeanVar", iDownSample: int = 1, dtype: torch.dtype = torch.float32, 
                 fEpsilon: float = 1e-8, strBaseDir: str = None):
        self.iN: int = -1
        self.iNt: int = -1
        self.iC: int = -1
        self.vecShape: list = None
        self.iBatchSize: int = iBatchSize
        self.iDownSample: int = iDownSample

        #try to find a path to the datasets
        self.strBaseDir = None
        strCfgPath = os.fsdecode(os.path.realpath(__file__)).replace("IDataset.py", "DatasetsCfg.json")
        if strBaseDir is not None: self.strBaseDir = strBaseDir
        elif os.path.exists(strCfgPath):
            with open(strCfgPath, "r") as f:
                dCfg = json.load(f)
                if "DatasetsPath" in dCfg.keys(): self.strBaseDir = dCfg["DatasetsPath"]
        
        #if nothing worked, ask the user what they want to do
        if self.strBaseDir is None:
            strGuessDir = "/home/" + getpass.getuser() + "/Datasets/"
            print("IDataset could not find a path to the Datasets folder and will attempt to use {} instead.".format(strGuessDir))
            if GetInput("Would you like to enter a path now? (Y/X)"):
                self.strBaseDir = input("Enter path to Datasets folder: ")
                with open(strCfgPath, "w") as f:
                    json.dump({"DatasetsPath": self.strBaseDir}, f)

            else: self.strBaseDir = strGuessDir

        if self.strBaseDir[-1] != "/": self.strBaseDir += "/"
        
        self.strFolder = ""
        self.strCacheDir = "./DownloadCache/"
        self.strDownloadURL: str = ""
        self.strDownloadPath: str = None

        self.X = None
        self.Xt = None
        self.Y = None
        self.Yt = None

        self.tdlTrain = None
        self.itrTrain = None
        self.tdlTest = None
        self.itrTest = None

        self.tAugTransform: Callable[[torch.tensor], torch.tensor] = None
        self.bDataAugOnTest = False

        self.tDSTransform: Callable[[torch.tensor], torch.tensor] = None

        self.strNorm = strNormalization
        self.tMean, self.tStd = None, None
        self.tDT = dtype
        self.fEpsilon = fEpsilon

        self.bTensorFormat: bool = True

        #setup cache directory for temp download files
        if not os.path.isdir(self.strCacheDir): os.mkdir(self.strCacheDir)

    def GetSamples(self, tIdx: torch.tensor, strSplit: str = "train") -> tuple[torch.tensor, torch.tensor]:
        if strSplit == "train":
            if self.bTensorFormat:
                x, y = self.X[tIdx, ...], self.Y[tIdx, ...]
            else:
                try:
                    x, y = next(self.itrTrain)
                except:
                    self.itrTrain = iter(self.tdlTrain)
                    x, y = next(self.itrTrain)
            if self.tDSTransform is not None: x = self.tDSTransform(x)
            #always augment training samples if a DA transform is present
            if self.tAugTransform is not None: x = self.tAugTransform(x)
        elif strSplit == "test":
            if self.bTensorFormat:
                x, y = self.Xt[tIdx, ...], self.Yt[tIdx, ...]
            else:
                try:
                    x, y = next(self.itrTest)
                except:
                    self.itrTest = iter(self.tdlTest)
                    x, y = next(self.itrTest)
            if self.tDSTransform is not None: x = self.tDSTransform(x)
            #always augment training samples if a DA transform is present
            if self.bDataAugOnTest and self.tAugTransform is not None: x = self.tAugTransform(x)
        else:
            print("Unsupported split {}!".format(strSplit))
            return None, None

        return x, y
    
    def Size(self, strSplit: str = "train") -> int:
        '''
        Returns the number of samples in the specified dataset split, which is by default "train"
        '''
        assert strSplit in ["train", "test"], "Invalid split {}".format(strSplit)

        if self.Loaded():
            if self.bTensorFormat: return self.X.shape[0] if strSplit == "train" else self.Xt.shape[0]
            return int(self.iBatchSize * len((self.tdlTrain if strSplit == "train" else self.tdlTest)))

        if strSplit == "train" and self.iN > 0: return self.iN
        elif strSplit == "test" and self.iNt > 0: return self.iNt

        return -1
        
    def Classes(self) -> int: return self.iC
    def Order(self) -> int: return len(self.vecShape)
    def Shape(self) -> list[int]: return self.vecShape
    def Loaded(self) -> bool: return self.X is not None or self.tdlTrain is not None

    def SetAugmentation(self, tAug: Callable[[torch.tensor], torch.tensor], bDataAugOnTest: bool = False) -> None:
        self.tAugTransform = tAug
        self.bDataAugOnTest = bDataAugOnTest

        return

    def SetBatchSize(self, iBS: int) -> None:
        self.iBatchSize = iBS
        if not self.bTensorFormat: print("WARNING! SetBatchSize() not implemented for dataloader format")

        return

    def HasMethod(self, strQ: str) -> bool:
        return (hasattr(self, strQ) and ismethod(getattr(self,strQ)))

    def GetDownloadPath(self) -> str:
        if self.strDownloadPath is None:
            if isinstance(self.strDownloadURL, list): strDL = self.strDownloadURL[-1]
            else: strDL = self.strDownloadURL
            return self.strCacheDir + strDL.split("/")[-1]
        return self.strCacheDir + self.strDownloadPath
    
    def GetFolder(self) -> str:
        return self.strBaseDir + self.strFolder

    def IDownload(self) -> None:
        print("Downloading dataset: {}...".format(self.strFolder.replace("/", "")))
        if self.HasMethod("Download"): return self.Download()
        return self.DefaultDownload()

    def DefaultDownload(self) -> None:
        if isinstance(self.strDownloadURL, str): urllib.request.urlretrieve(self.strDownloadURL, self.GetDownloadPath())
        elif isinstance(self.strDownloadURL, list):
            for strDL in self.strDownloadURL: urllib.request.urlretrieve(strDL, self.strCacheDir + strDL.split("/")[-1])
        return
    
    def IUnpack(self) -> None:
        if not self.HasMethod("Unpack"):
            print("Error! Unpack() not defined for dataset: {}!".format(self.strFolder.replace("/", "")))
            quit()

        bDownloaded = os.path.exists(self.GetDownloadPath())
        if not bDownloaded:
            self.IDownload()
            print("Unpacking dataset: {}...".format(self.strFolder.replace("/", "")))
            self.Unpack()
        else:
            try:
                self.Unpack()
            except:
                print("Unpacking failed for dataset: {}".format(self.strFolder.replace("/", "")))
                if not GetInput("Re-download and try again? (Y/X)"): return
                self.IDownload()
                print("Unpacking dataset: {}...".format(self.strFolder.replace("/", "")))
                self.Unpack()

        return
    
    def ILoad(self) -> None:
        tStart = time.time()
        if not self.HasMethod("Load"):
            print("Error! Load() not defined for dataset: {}!".format(self.strFolder.replace("/", "")))
            quit()

        if not os.path.isdir(self.strBaseDir + self.strFolder): os.mkdir(self.strBaseDir + self.strFolder)

        if len(os.listdir(self.strBaseDir + self.strFolder)) == 0:
            print("Could not find dataset: {}".format(self.strFolder.replace("/", "")))
            self.IUnpack()

        try:
            self.Load()
        except:
            print("Load failed for dataset: {}".format(self.strFolder.replace("/", "")))
            if not GetInput("Re-unpack and try again? (Y/X)"): return
            self.IUnpack()
            print("Load dataset: {}...".format(self.strFolder.replace("/", "")))
            self.Load()

        tEnd = time.time()

        if self.X is None and self.tdlTrain is None:
            print("Load() failed for {}!".format(self.strFolder.replace("/", "")))
            return
        elif self.X is not None and self.tdlTrain is not None:
            print("Error! IDataset expects either tensor OR dataloader mode, not both! Fix Load() for {}".format(self.strFolder.replace("/", "")))
            return
        
        if self.X is not None: self.bTensorFormat = True
        else: self.bTensorFormat = False

        if self.bTensorFormat:
            self.vecShape = list(self.X.shape)[1:]
        else:
            tdlTr = iter(self.tdlTrain)
            x, _ = next(tdlTr)
            self.vecShape = list(x.shape)[1:]

        if self.iDownSample > 0: self.DownSample()

        print("Loaded {} in {}s for {} mode".format(self.strFolder.replace("/", ""), tEnd - tStart, "tensor" if self.X is not None else "dataloader"))

        return
    
    def DownSample(self) -> None:
        assert self.Order() == 3, "DownSample() is currently only supported for images!"
        if self.bTensorFormat:
            tDS = torchvision.transforms.Resize((self.iDownSample, self.iDownSample))
            self.X = tDS(self.X).to(self.tDT)
            self.Xt = tDS(self.Xt).to(self.tDT)
        else: self.tDSTransform = torchvision.transforms.Resize((self.iDownSample, self.iDownSample))
        self.vecShape[1] = self.iDownSample
        self.vecShape[2] = self.iDownSample
        return
    
    def SubSample(self, fDataFraction: float) -> None:
        if not self.bTensorFormat:
            print("WARNING: SubSample() not implemented for dataloader mode")
            return
        if fDataFraction >= 1.0: return
        
        self.X, self.Y = self.GetRandomSubset(int(fDataFraction * self.Size("train")), "train")
        self.TX, self.TY = self.GetRandomSubset(int(fDataFraction * self.Size("test")), "test")

        return

    def Normalize(self) -> None:
        if not self.Loaded(): self.ILoad()
        if "none" in self.strNorm.lower(): return

        if not self.bTensorFormat:
            print("Normalize() is not supported for dataloader format datasets!")
            if GetInput("Switch to 'None' normalization? (Y/X)"): self.strNorm = "None"
            return

        self.X = self.X.to(self.tDT)
        self.Xt = self.Xt.to(self.tDT)

        if "meanvar" in self.strNorm.lower():
            if "elementwise" in self.strNorm.lower():
                self.tMean = torch.mean(self.X, dim = 0, keepdim = True)
                self.tStd = torch.std(self.X, dim = 0, keepdim = True) + self.fEpsilon
                self.X -= self.tMean
                self.X /= self.tStd
                self.Xt -= self.tMean
                self.Xt /= self.tStd
            else:
                if self.Order() == 3:
                    #images
                    self.tMean = torch.mean(self.X, dim = (0, 2, 3), keepdim = True)
                    self.tStd = torch.std(self.X, dim = (0, 2, 3), keepdim = True) + self.fEpsilon
                    self.X -= self.tMean
                    self.X /= self.tStd
                    self.Xt -= self.tMean
                    self.Xt /= self.tStd
                elif self.iOrder == 2:
                    print("NOT IMPLEMENTED YET!")
                    quit()
                elif self.iOrder == 1:
                    self.tMean = torch.mean(self.X)
                    self.tStd = torch.std(self.X) + self.fEpsilon
                    self.X -= self.tMean
                    self.X /= self.tStd
                    self.Xt -= self.tMean
                    self.Xt /= self.tStd

        return
    
    def GetRandomSubset(self, n: int = None, strSplit: str = "train") -> tuple[torch.tensor, torch.tensor]:
        if strSplit not in ["train", "test"]:
            print("Invalid split {}!".format(strSplit))
            return
        
        if n is None: n = self.Size(strSplit)
        
        fP = n / self.iN
        nc = n // self.iC
        
        if self.bTensorFormat:
            if len(self.Y.shape) > 1:
                YA = torch.argmax(self.Y, dim = 1) if strSplit == "train" else torch.argmax(self.Yt, dim = 1)
            else:
                YA = self.Y if strSplit == "train" else self.Yt
            vecIdx = [torch.where(YA == i)[0] for i in range(self.iC)]
            rng = np.random.default_rng()
            
            idx = torch.zeros((0), dtype=torch.int32)
            for i in range(self.iC):
                Idx = rng.permutation(vecIdx[i].shape[0])[:int(vecIdx[i].shape[0] * fP)]
                idx = torch.cat((idx, vecIdx[i][Idx]), dim = 0)
            XR = self.X[idx] if strSplit == "train" else self.Xt[idx]
            return XR, YA[idx]

        XR = torch.zeros([n] + self.vecShape)
        YA = torch.zeros((n, self.iC))
        tNC = torch.zeros((self.iC))
        idx = 0
        while torch.min(tNC) < nc:
            x, y = self.GetSamples([], strSplit)
            for i in range(y.shape[0]):
                if tNC[y[i]] < 5:
                    XR[idx, ...] = x[i, ...]
                    YA[idx, y[i]] = 1
                    idx += 1
                    tNC[y[i]] += 1
        return XR, YA

if __name__ == "__main__":
    from CIFAR10 import *
    from Imagenet import *

    #ds = CIFAR10(128, strNormalization = "meanvarelementwise")
    ds = Imagenet(128, strNormalization = "none")
    ds.ILoad()
    ds.Normalize()

    print(ds.Size(), ds.Shape())

    x, y = ds.GetRandomSubset(5000)
    print(x.shape, y.shape)