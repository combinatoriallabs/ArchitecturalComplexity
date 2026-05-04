import torch
import numpy as np
import os
import zipfile
from PIL import Image
import tqdm

try:
    from Arch.Datasets.IDataset import IDataset
except:
    from Datasets.IDataset import IDataset


class EuroSAT(IDataset):
    def __init__(self, iBatchSize: int, strNormalization: str = "MeanVar", 
                 iDownSample: int = 1, dtype: torch.dtype = torch.float32, fEpsilon: float = 1e-8):
        super().__init__(iBatchSize, strNormalization, iDownSample, dtype, fEpsilon)

        self.iC = 10
        self.strDownloadURL = "https://www.kaggle.com/api/v1/datasets/download/apollo2506/eurosat-dataset"
        self.strFolder = "EuroSAT/"

    def Unpack(self) -> None:
        if not os.path.isdir(self.strCacheDir + "EuroSAT"): os.mkdir(self.strCacheDir + "EuroSAT")

        if len(os.listdir(self.strCacheDir + "EuroSAT")) > 0:
            print("Skipping unzip operation for StanfordCars. If this is undesired behavior delete or rename DownloadCache/EuroSAT/")
        else:
            with zipfile.ZipFile(self.GetDownloadPath(), 'r') as zip:
                zip.extractall(self.strCacheDir + "EuroSAT/")

        vecDirs = [self.strCacheDir + "EuroSAT/EuroSAT/AnnualCrop/", self.strCacheDir + "EuroSAT/EuroSAT/Forest/", self.strCacheDir + "EuroSAT/EuroSAT/HerbaceousVegetation/",
               self.strCacheDir + "EuroSAT/EuroSAT/Highway/", self.strCacheDir + "EuroSAT/EuroSAT/Industrial/", self.strCacheDir + "EuroSAT/EuroSAT/Pasture/",
               self.strCacheDir + "EuroSAT/EuroSAT/PermanentCrop/", self.strCacheDir + "EuroSAT/EuroSAT/Residential/", self.strCacheDir + "EuroSAT/EuroSAT/River/",
               self.strCacheDir + "EuroSAT/EuroSAT/SeaLake/"]

        XA = torch.zeros((27000, 3, 64, 64))
        YA = torch.zeros((27000), dtype = torch.long)

        idx = 0
        for i in range(len(vecDirs)):
            strDir = vecDirs[i]
            for f in tqdm.tqdm(os.listdir(strDir)):
                strF = os.fsdecode(f)
                im = Image.open(strDir + strF)
                tIm = torch.tensor(np.array(im))
                XA[idx, ...] = torch.permute(tIm, (2, 0, 1))
                YA[idx] = i
                idx += 1

        assert idx == 27000, "Expected 27000 images but found {} instead!".format(idx)

        vecIdx = [torch.where(YA == i)[0] for i in range(self.iC)]

        X = torch.zeros((21600, 3, 64, 64))
        Y = torch.zeros((21600), dtype = torch.long)
        Xt = torch.zeros((5400, 3, 64, 64))
        Yt = torch.zeros((5400), dtype = torch.long)

        idx = 0
        idx2 = 0
        for i in range(len(vecDirs)):
            sz = vecIdx[i].shape[0]
            X[idx:idx + int(sz * 0.8), ...] = XA[vecIdx[i][:int(sz * 0.8)], ...]
            Y[idx:idx + int(sz * 0.8)] = i
            Xt[idx2:idx2 + int(sz * 0.2), ...] = XA[vecIdx[i][int(sz * 0.8):], ...]
            Yt[idx2:idx2 + int(sz * 0.2)] = i
            idx += int(sz * 0.8)
            idx2 += int(sz * 0.2)

        torch.save(X, self.GetFolder() + "X_Train.pkl")
        torch.save(Xt, self.GetFolder() + "X_Test.pkl")
        
        torch.save(Y, self.GetFolder() + "Y_Train.pkl")
        torch.save(Yt, self.GetFolder() + "Y_Test.pkl")

        return

    def Load(self) -> None:
        self.X = torch.load(self.GetFolder() + "X_Train.pkl")
        self.Xt = torch.load(self.GetFolder() + "X_Test.pkl")
        
        self.Y = torch.load(self.GetFolder() + "Y_Train.pkl")
        self.Yt = torch.load(self.GetFolder() + "Y_Test.pkl")

        return