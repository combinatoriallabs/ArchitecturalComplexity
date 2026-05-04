import torch
import numpy as np
import zipfile
import tqdm

try:
    from Arch.Datasets.IDataset import IDataset
except:
    from Datasets.IDataset import IDataset


class OxfordPets(IDataset):
    def __init__(self, iBatchSize: int, strNormalization: str = "MeanVar", 
                 iDownSample: int = 1, dtype: torch.dtype = torch.float32, fEpsilon: float = 1e-8):
        super().__init__(iBatchSize, strNormalization, iDownSample, dtype, fEpsilon)

        self.iC = 37
        self.iN = -1
        self.strDownloadURL = "https://github.com/ml4py/dataset-iiit-pet/archive/refs/heads/master.zip"
        self.strFolder = "OxfordPets/"
        self.strDownloadPath = "master.zip"

    def Unpack(self) -> None:
        with zipfile.ZipFile(self.GetDownloadPath(), 'r') as zip:
            zip.extractall(self.strCacheDir)

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