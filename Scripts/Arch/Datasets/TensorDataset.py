import torch
import numpy as np
import pickle
import tarfile
import tqdm

try:
    from Arch.Datasets.IDataset import IDataset
except:
    from Datasets.IDataset import IDataset


class TensorDataset(IDataset):
    def __init__(self, iBatchSize: int, X: torch.tensor, Y: torch.tensor, Xt: torch.tensor, Yt: torch.tensor, 
                 strNormalization: str = "None", iDownSample: int = 1, dtype: torch.dtype = torch.float32, fEpsilon: float = 1e-8):
        super().__init__(iBatchSize, strNormalization, iDownSample, dtype, fEpsilon)

        if len(Y.shape) == 1: self.iC = torch.max(Y) + 1
        elif len(Y.shape) == 2: self.iC = Y.shape[1]

        self.iN = Y.shape[0]

        self.strDownloadURL = None
        self.strFolder = None

        self.X = X
        self.Y = Y
        self.Xt = Xt
        self.Yt = Yt

        self.vecShape = list(X.shape)[1:]

        return

    def Unpack(self) -> None:
        return

    def Load(self) -> None:
        return