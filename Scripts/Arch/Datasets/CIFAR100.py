import torch
import numpy as np
import pickle
import tarfile
import tqdm

try:
    from Arch.Datasets.IDataset import IDataset
except:
    from Datasets.IDataset import IDataset

class CIFAR100(IDataset):
    def __init__(self, iBatchSize: int, strNormalization: str = "MeanVar", 
                 iDownSample: int = 1, dtype: torch.dtype = torch.float32, fEpsilon: float = 1e-8, bFineLabels: bool = True):
        super().__init__(iBatchSize, strNormalization, iDownSample, dtype, fEpsilon)

        self.iC = 100 if bFineLabels else 20
        self.iN = 50000
        self.strDownloadURL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        self.strFolder = "CIFAR100/"
        self.bFineLabels = bFineLabels

    def Unpack(self) -> None:
        with tarfile.open(self.GetDownloadPath(), 'r') as tar:
            tar.extractall(self.strCacheDir)

        vecPaths = [self.strCacheDir + "cifar-100-python/train", self.strCacheDir + "cifar-100-python/test"]
        
        X = torch.zeros((60000, 3, 32, 32))
        for i in range(len(vecPaths)):
            path = vecPaths[i]
            o = 0 if not i else 50000
            with open(path, 'rb') as f:
                d = pickle.load(f, encoding='bytes')
                data = np.array(d[b'data'])
                for n in tqdm.tqdm(range(data.shape[0])):
                    r = data[n, :1024]
                    g = data[n, 1024:2048]
                    b = data[n, 2048:]
                
                    r = r.reshape((32, 32)).astype(np.float64)
                    g = g.reshape((32, 32)).astype(np.float64)
                    b = b.reshape((32, 32)).astype(np.float64)

                    r /= 255
                    g /= 255
                    b /= 255

                    X[o + n, 0, ...] = torch.tensor(r)
                    X[o + n, 1, ...] = torch.tensor(g)
                    X[o + n, 2, ...] = torch.tensor(b)
                        
        #label unpacking
        coarse_labels = np.zeros((0), np.uint8)
        fine_labels = np.zeros_like(coarse_labels)
        for path in vecPaths:
            with open(path, 'rb') as f:
                d = pickle.load(f, encoding='bytes')
            coarse_labels = np.concatenate((coarse_labels, np.array(d[b'coarse_labels'])))
            fine_labels = np.concatenate((fine_labels, np.array(d[b'fine_labels'])))
        
        Yc = torch.tensor(coarse_labels)
        Yf = torch.tensor(fine_labels)

        #clone() is necessary to prevent weird saving behavior
        Xt = torch.clone(X[50000:, ...])
        X = torch.clone(X[:50000, ...])
        Yct = torch.clone(Yc[50000:, ...])
        Yc = torch.clone(Yc[:50000, ...])
        Yft = torch.clone(Yf[50000:, ...])
        Yf = torch.clone(Yf[:50000, ...])

        torch.save(X, self.GetFolder() + "X_Train.pkl")
        torch.save(Xt, self.GetFolder() + "X_Test.pkl")

        torch.save(Yc, self.GetFolder() + "YC_Train.pkl")
        torch.save(Yct, self.GetFolder() + "YC_Test.pkl")
        
        torch.save(Yf, self.GetFolder() + "YF_Train.pkl")
        torch.save(Yft, self.GetFolder() + "YF_Test.pkl")
            
        return
    
    def Load(self) -> None:
        self.X = torch.load(self.GetFolder() + "X_Train.pkl")
        self.Xt = torch.load(self.GetFolder() + "X_Test.pkl")

        if self.bFineLabels:
            self.Y = torch.load(self.GetFolder() + "YF_Train.pkl")
            self.Yt = torch.load(self.GetFolder() + "YF_Test.pkl")
        else:
            self.Y = torch.load(self.GetFolder() + "YC_Train.pkl")
            self.Yt = torch.load(self.GetFolder() + "YC_Test.pkl")

        return