import torch
import torchvision
import numpy as np
import os
import zipfile
from PIL import Image
import tqdm

try:
    from Arch.Datasets.IDataset import IDataset
except:
    from Datasets.IDataset import IDataset


class PatternNet(IDataset):
    def __init__(self, iBatchSize: int, strNormalization: str = "MeanVar", 
                 iDownSample: int = 1, dtype: torch.dtype = torch.float32, fEpsilon: float = 1e-8):
        super().__init__(iBatchSize, strNormalization, iDownSample, dtype, fEpsilon)

        self.iC = 38
        self.strDownloadURL = "https://sites.google.com/view/zhouwx/dataset"
        self.strDownloadPath = "PatternNet.zip"
        self.strFolder = "PatternNet/"

    def Download(self):
        print("Automatic downloading not supported for PatternNet! Go to {}, download .zip file, and place in ./DownloadCache/".format(self.strDownloadURL))
        input("Press any key when finished...")
        return

    def Unpack(self) -> None:
        if os.path.isdir(self.strCacheDir + "PatternNet") and len(os.listdir(self.strCacheDir + "PatternNet")) > 0:
            print("Skipping unzip operation for StanfordCars. If this is undesired behavior delete or rename DownloadCache/EuroSAT/")
        else:
            with zipfile.ZipFile(self.GetDownloadPath(), 'r') as zip:
                zip.extractall(self.strCacheDir)

        tTransform = torchvision.transforms.Resize((224, 224))

        vecLabels = []
        for f in os.listdir(self.strCacheDir + "PatternNet/images/"):
            vecLabels.append(os.fsdecode(f))
        
        assert len(vecLabels) == 38, "Expected 38 labels but found {} instead!".format(len(vecLabels))

        XA = torch.zeros((30400, 3, 224, 224))
        YA = torch.zeros((30400), dtype = torch.long)

        idx = 0
        for i in range(len(vecLabels)):
            for f in tqdm.tqdm(os.listdir(self.strCacheDir + "PatternNet/images/" + vecLabels[i])):
                strF = os.fsdecode(f)
                im = Image.open(self.strCacheDir + "PatternNet/images/" + vecLabels[i] + "/" + strF)
                tIm = torch.tensor(np.array(im))
                tIm = torch.permute(tIm, (2, 0, 1))
                XA[idx, ...] = tTransform(tIm)
                YA[idx] = i
                idx += 1

        assert idx == 30400, "Expected 30400 images but found {} instead!".format(idx)

        vecIdx = [torch.where(YA == i)[0] for i in range(self.iC)]

        X = torch.zeros((24320, 3, 224, 224))
        Y = torch.zeros((24320), dtype = torch.long)
        Xt = torch.zeros((6080, 3, 224, 224))
        Yt = torch.zeros((6080), dtype = torch.long)

        idx = 0
        idx2 = 0
        for i in range(len(vecLabels)):
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
