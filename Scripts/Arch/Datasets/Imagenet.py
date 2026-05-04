import torch
import torchvision
import shutil
import zipfile
import tqdm
import os

try:
    from Arch.Datasets.IDataset import IDataset
except:
    from Datasets.IDataset import IDataset

class Imagenet(IDataset):
    def __init__(self, iBatchSize: int, strNormalization: str = "None", 
                 iDownSample: int = 1, dtype: torch.dtype = torch.float32, fEpsilon: float = 1e-8):
        super().__init__(iBatchSize, strNormalization, iDownSample, dtype, fEpsilon)

        self.iC = 1000
        self.iN = 1.2e6
        self.strDownloadURL = "https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data"
        self.strFolder = "Imagenet/"
        self.strDownloadPath = "imagenet-object-localization-challenge.zip"

    def Download(self):
        print("Automatic downloading not supported for Imagenet! Go to {}, download .zip file, and place in ./DownloadCache/".format(self.strDownloadURL))
        input("Press any key when finished...")
        return

    def Unpack(self) -> None:
        if len(os.listdir(self.strCacheDir + "ILSVRC/")) > 0:
            print("Skipping unzip operation for Imagenet. If this is undesired behavior, delete or rename DownloadCache/ILSVRC/")
        else:
            with zipfile.ZipFile(self.GetDownloadPath(), 'r') as zip:
                zip.extractall(self.strCacheDir)

        import xml.etree.ElementTree as ET
        vecSplits = ["val"]

        for strSplit in vecSplits:
            strDir = self.strCacheDir + "ILSVRC/Annotations/CLS-LOC/" + strSplit + "/"
            strModifyDir = self.strCacheDir + "ILSVRC/Data/CLS-LOC/" + strSplit + "/"
            for f in tqdm.tqdm(os.listdir(strDir)):
                strF = os.fsdecode(f)
                xmlLabel = ET.parse(strDir + strF).getroot()
                strLabel = xmlLabel[5][0].text
                strPath = xmlLabel[1].text

                strLabelDir = strModifyDir + strLabel + "/"
                if not os.path.exists(strLabelDir): os.mkdir(strLabelDir)
                os.rename(strModifyDir + strPath + ".JPEG", strLabelDir + strPath + ".JPEG")

        shutil.move(self.strCacheDir + "ILSVRC/Data/CLS-LOC/train/", self.GetFolder() + "train/")
        shutil.move(self.strCacheDir + "ILSVRC/Data/CLS-LOC/val/", self.GetFolder() + "val/")

        return
            
    def Load(self) -> None:
        import multiprocessing
        iCPUs = multiprocessing.cpu_count()

        trnsNormalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
        trnsTrain = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            trnsNormalize,
            torchvision.transforms.ConvertImageDtype(self.tDT),
        ])
        trnsTest = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            trnsNormalize,
            torchvision.transforms.ConvertImageDtype(self.tDT),
        ])
        tdsTrain = torchvision.datasets.ImageFolder(self.GetFolder() + "train", transform = trnsTrain)
        tdsTest = torchvision.datasets.ImageFolder(self.GetFolder() + "val", transform = trnsTest)

        self.tdlTrain = torch.utils.data.DataLoader(tdsTrain, batch_size = self.iBatchSize, shuffle = True, num_workers = iCPUs)
        self.tdlTest = torch.utils.data.DataLoader(tdsTest, batch_size = self.iBatchSize, shuffle = True, num_workers = iCPUs)

        return

if __name__ == "__main__":
    dsIM = Imagenet(128)