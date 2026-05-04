import torch
import numpy as np
import pickle
import tarfile
import tqdm

try:
    from Arch.Datasets.IDataset import IDataset
except:
    from Datasets.IDataset import IDataset


class CIFAR10(IDataset):
    def __init__(self, iBatchSize: int, strNormalization: str = "MeanVar", 
                 iDownSample: int = 1, dtype: torch.dtype = torch.float32, fEpsilon: float = 1e-8):
        super().__init__(iBatchSize, strNormalization, iDownSample, dtype, fEpsilon)

        self.iC = 10
        self.iN = 50000
        self.strDownloadURL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        self.strFolder = "CIFAR10/"

    def Unpack(self) -> None:
        with tarfile.open(self.GetDownloadPath(), 'r') as tar:
            tar.extractall(self.strCacheDir)

        vecPaths = [self.strCacheDir + "cifar-10-batches-py/data_batch_1", self.strCacheDir + "cifar-10-batches-py/data_batch_2", 
                self.strCacheDir + "cifar-10-batches-py/data_batch_3", self.strCacheDir + "cifar-10-batches-py/data_batch_4", 
                self.strCacheDir + "cifar-10-batches-py/data_batch_5", self.strCacheDir + "cifar-10-batches-py/test_batch"]
    
        #image unpacking
        X = torch.zeros((60000, 3, 32, 32))
        o = 0
        for path in vecPaths:
            with open(path, 'rb') as f:
                d = pickle.load(f, encoding='bytes')
            data = np.array(d[b'data'])
            for n in tqdm.tqdm(range(data.shape[0])):
                idx = o + n

                r = data[n, :1024]
                g = data[n, 1024:2048]
                b = data[n, 2048:]
            
                r = r.reshape((32, 32)).astype(np.float64)
                g = g.reshape((32, 32)).astype(np.float64)
                b = b.reshape((32, 32)).astype(np.float64)

                r /= 255
                g /= 255
                b /= 255

                X[idx, 0, ...] = torch.tensor(r)
                X[idx, 1, ...] = torch.tensor(g)
                X[idx, 2, ...] = torch.tensor(b)
            o += 10000

        #label unpacking
        nLabels = np.zeros((0), np.uint8)
        for path in vecPaths:
            with open(path, 'rb') as f:
                d = pickle.load(f, encoding='bytes')
            nLabels = np.concatenate((nLabels, np.array(d[b'labels'])))

        Y = torch.tensor(nLabels)

        #clone() is necessary to prevent weird saving behavior
        Yt = torch.clone(Y[50000:, ...])
        Y = torch.clone(Y[:50000, ...])

        Xt = torch.clone(X[50000:, ...])
        X = torch.clone(X[:50000, ...])

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