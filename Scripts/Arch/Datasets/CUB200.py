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


class CUB200(IDataset):
    def __init__(self, iBatchSize: int, strNormalization: str = "MeanVar", 
                 iDownSample: int = 1, dtype: torch.dtype = torch.float32, fEpsilon: float = 1e-8):
        super().__init__(iBatchSize, strNormalization, iDownSample, dtype, fEpsilon)

        self.iC = 200
        self.strDownloadURL = "https://www.kaggle.com/api/v1/datasets/download/wenewone/cub2002011"
        self.strFolder = "CUB200/"

    def Unpack(self) -> None:
        if not os.path.isdir(self.strCacheDir + "CUB200"): os.mkdir(self.strCacheDir + "CUB200")

        if len(os.listdir(self.strCacheDir + "CUB200")) > 0:
            print("Skipping unzip operation for StanfordCars. If this is undesired behavior delete or rename DownloadCache/CUB200/")
        else:
            with zipfile.ZipFile(self.GetDownloadPath(), 'r') as zip:
                zip.extractall(self.strCacheDir + "CUB200/")

        with open(self.strCacheDir + "CUB200/CUB_200_2011/train_test_split.txt", "r") as f:
            tSplit = torch.tensor([int(row[-2]) for row in f.readlines()])

        iTr = torch.sum(tSplit)
        iTs = tSplit.shape[0] - iTr

        X = torch.zeros((iTr, 3, 224, 224))
        Y = torch.zeros((iTr), dtype = torch.long)
        Xt = torch.zeros((iTs, 3, 224, 224))
        Yt = torch.zeros((iTs), dtype = torch.long)

        idxTr = 0
        idxTs = 0
        tTransform = torchvision.transforms.Resize((224, 224))

        with open(self.strCacheDir + "CUB200/CUB_200_2011/images.txt", "r") as f:
            vecPaths = [l[:-1] for l in f.readlines()]

        for i in tqdm.tqdm(range(len(vecPaths))):
            strPath = vecPaths[i].split(" ")[1]
            iCls = int(strPath[:3])
            
            im = Image.open(self.strCacheDir + "CUB200/CUB_200_2011/images/" + strPath)
            tIm = torch.tensor(np.array(im))
            
            #plt.imshow(tIm)
            #plt.show()
            #plt.close()
            
            if len(tIm.shape) == 2:
                tIm = torch.stack((tIm, tIm, tIm), dim = 2)
            tIm = tTransform(torch.permute(tIm, (2, 0, 1)).unsqueeze(0))
            if tSplit[i]:
                X[idxTr,...] = tIm
                Y[idxTr] = iCls - 1
                idxTr += 1
            else:
                Xt[idxTs,...] = tIm
                Yt[idxTs] = iCls - 1
                idxTs += 1

        if idxTr != iTr or idxTs != iTs:
            print("Something weird happened!")
            print(idxTr, iTr, idxTs, iTs)

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
