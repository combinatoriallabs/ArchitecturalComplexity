import torch
import torchvision
import numpy as np
import zipfile
import os
import tqdm
from PIL import Image

try:
    from Arch.Datasets.IDataset import IDataset
except:
    from Datasets.IDataset import IDataset


class StanfordCars(IDataset):
    def __init__(self, iBatchSize: int, strNormalization: str = "MeanVar", 
                 iDownSample: int = 1, dtype: torch.dtype = torch.float32, fEpsilon: float = 1e-8):
        super().__init__(iBatchSize, strNormalization, iDownSample, dtype, fEpsilon)

        self.iC = 196
        self.strDownloadURL = "https://www.kaggle.com/api/v1/datasets/download/jutrera/stanford-car-dataset-by-classes-folder"
        self.strFolder = "StanfordCars/"

    def Unpack(self) -> None:
        if not os.path.isdir(self.strCacheDir + "stanford-cars"): os.mkdir(self.strCacheDir + "stanford-cars")

        if len(os.listdir(self.strCacheDir + "stanford-cars")) > 0:
            print("Skipping unzip operation for StanfordCars. If this is undesired behavior delete or rename DownloadCache/stanford-cars/")
        else:
            with zipfile.ZipFile(self.GetDownloadPath(), 'r') as zip:
                zip.extractall(self.strCacheDir + "stanford-cars/")

        X = torch.zeros((8144, 3, 224, 224))
        Y = torch.zeros((8144), dtype = torch.long)
        Xt = torch.zeros((8041, 3, 224, 224))
        Yt = torch.zeros((8041), dtype = torch.long)

        tTransform = torchvision.transforms.Resize((224, 224))

        vecLabels = []
        for f in os.listdir(self.strCacheDir + "stanford-cars/car_data/car_data/train/"):
            vecLabels.append(os.fsdecode(f))
        
        assert len(vecLabels) == 196, "Expected 196 labels but found {} instead!".format(len(vecLabels))

        idx = 0
        for i in tqdm.tqdm(range(len(vecLabels))):
            for f in os.listdir(self.strCacheDir + "stanford-cars/car_data/car_data/train/" + vecLabels[i] + "/"):
                strPath = os.fsdecode(f)
                im = Image.open(self.strCacheDir + "stanford-cars/car_data/car_data/train/" + vecLabels[i] + "/" + strPath)

                tIm = torch.tensor(np.array(im))
                if len(tIm.shape) == 2:
                    tIm = torch.stack((tIm, tIm, tIm), dim = 2)
                tIm = tTransform(torch.permute(tIm, (2, 0, 1)).unsqueeze(0))
                X[idx,...] = tIm
                Y[idx] = i
                idx += 1

        assert idx == 8144, "Expected 8144 train images but found {} instead!".format(idx)

        idx = 0
        for i in tqdm.tqdm(range(len(vecLabels))):
            for f in os.listdir(self.strCacheDir + "stanford-cars/car_data/car_data/test/" + vecLabels[i] + "/"):
                strPath = os.fsdecode(f)
                im = Image.open(self.strCacheDir + "stanford-cars/car_data/car_data/test/" + vecLabels[i] + "/" + strPath)

                tIm = torch.tensor(np.array(im))
                if len(tIm.shape) == 2:
                    tIm = torch.stack((tIm, tIm, tIm), dim = 2)
                tIm = tTransform(torch.permute(tIm, (2, 0, 1)).unsqueeze(0))
                Xt[idx,...] = tIm
                Yt[idx] = i
                idx += 1

        assert idx == 8041, "Expected 8041 train images but found {} instead!".format(idx)

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
