import torch
import numpy as np
import pandas as pd
from PIL import Image
import io
import tqdm

try:
    from Arch.Datasets.IDataset import IDataset
except:
    from Datasets.IDataset import IDataset

class TinyImagenet(IDataset):
    def __init__(self, iBatchSize: int, strNormalization: str = "MeanVar", 
                 iDownSample: int = 1, dtype: torch.dtype = torch.float32, fEpsilon: float = 1e-8):
        super().__init__(iBatchSize, strNormalization, iDownSample, dtype, fEpsilon)

        self.iC = 200
        self.iN = 100000
        self.strDownloadURL = ["https://huggingface.co/datasets/zh-plus/tiny-imagenet/resolve/main/data/train-00000-of-00001-1359597a978bc4fa.parquet?download=true",
                               "https://huggingface.co/datasets/zh-plus/tiny-imagenet/resolve/main/data/valid-00000-of-00001-70d52db3c749a935.parquet?download=true"]
        self.strFolder = "TinyImagenet/"

    def Unpack(self) -> None:
        X, Y = self.ParseParquet(self.strCacheDir + "train-00000-of-00001-1359597a978bc4fa.parquet?download=true")
        Xt, Yt = self.ParseParquet(self.strCacheDir + "valid-00000-of-00001-70d52db3c749a935.parquet?download=true")

        torch.save(X, self.GetFolder() + "X_Train.pkl")
        torch.save(Xt, self.GetFolder() + "X_Test.pkl")
        
        torch.save(Y, self.GetFolder() + "Y_Train.pkl")
        torch.save(Yt, self.GetFolder() + "Y_Test.pkl")

        return
    
    def ParseParquet(self, strPath: str) -> tuple[torch.tensor, torch.tensor]:
        df = pd.read_parquet(strPath)
    
        X = torch.zeros((df.shape[0], 3, 64, 64))
        Y = torch.zeros((df.shape[0]), dtype = torch.long)
        
        cnt = 0
        
        for i in tqdm.tqdm(range(df.shape[0])):
            im = Image.open(io.BytesIO(df.iloc[i]["image"]["bytes"]))
            tim = torch.tensor(np.array(im))
            if len(tim.shape) == 2:
                tim = torch.stack((tim, tim, tim), dim = 2)
                cnt += 1
            tim = torch.permute(tim, (2, 0, 1))
            X[i,...] = tim / 255
            
            Y[i] = df.iloc[i]["label"]
        
        return X, Y
    
    def Load(self) -> None:
        self.X = torch.load(self.GetFolder() + "X_Train.pkl")
        self.Xt = torch.load(self.GetFolder() + "X_Test.pkl")
        
        self.Y = torch.load(self.GetFolder() + "Y_Train.pkl")
        self.Yt = torch.load(self.GetFolder() + "Y_Test.pkl")

        return