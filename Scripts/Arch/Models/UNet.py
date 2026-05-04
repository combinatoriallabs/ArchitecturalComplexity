import torch

#implementation taken from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
#modified to have a linearizable structure for use with the IModel interface

def DoubleConv(iCin: int, iCout: int, iCmid: int = None) -> list[torch.nn.Module]:
    if iCmid is None: iCmid = iCout

    return [
        torch.nn.Conv2d(iCin, iCmid, kernel_size=3, padding=1, bias=False),
        torch.nn.BatchNorm2d(iCmid),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(iCmid, iCout, kernel_size=3, padding=1, bias=False),
        torch.nn.BatchNorm2d(iCout),
        torch.nn.ReLU(inplace=True)
    ]


def Down(iCin: int, iCout: int) -> torch.nn.Sequential:
    vecModules = [torch.nn.MaxPool2d(2)] + DoubleConv(iCin, iCout)
    return torch.nn.Sequential(*vecModules)


class Up(torch.nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = torch.nn.Sequential(*DoubleConv(in_channels, out_channels, in_channels // 2))
        else:
            self.up = torch.nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = torch.nn.Sequential(*DoubleConv(in_channels, out_channels))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        print(diffY, diffX)

        x1 = torch.nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

    
class UNet(torch.nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = torch.nn.Sequential(*DoubleConv(n_channels, 64))

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))

        self.outc = torch.nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    

if __name__ == "__main__":
    from Models.IModel import IModel
    from Models.ModelUtils import *
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    x = torch.randn((1, 3, 224, 224)).to(device)

    model = UNet(n_channels = 3, n_classes = 10).to(device)
    iModel = IModel(model)
    PrintModelSummary(iModel)
    print(model(x).shape)