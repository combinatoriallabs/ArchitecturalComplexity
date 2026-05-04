import torch
import numpy as np
from typing import Optional, Callable, OrderedDict
from functools import partial
import os
import math

try:
    from Arch.Models.TransformerBlocks import *
    from Arch.Models.ModelUtils import *
except:
    from TransformerBlocks import *
    from ModelUtils import *

#Path config
if os.environ["USER"] == "nickubuntu" or os.environ["USER"] == "nickubuntuworkstation":
    strModelDir = "/home/" + os.environ["USER"] + "/ImagenetPretrainedModels/"
elif os.environ["USER"] == "nicklaptop":
    strModelDir = "/home/nicklaptop/ImagenetPretrainedModels/"
else:
    strModelDir = "NOT SET UP YET!"
    print("Double check path config!")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype).unsqueeze(0)

def resize_positional_embedding_(posemb, posemb_new):
    """Rescale the grid of position embeddings in a sensible manner"""
    #Taken and modified from: https://github.com/lukemelas/PyTorch-Pretrained-ViT/blob/master/pytorch_pretrained_vit/utils.py
    from scipy.ndimage import zoom

    # Deal with class token
    ntok_new = posemb_new.shape[1]
    posemb_tok, posemb_grid = posemb[:, -1:], posemb[0, :-1]
    ntok_new -= 1

    # Get old and new grid sizes
    gs_old = int(np.sqrt(len(posemb_grid)))
    gs_new = int(np.sqrt(ntok_new))
    posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

    # Rescale grid
    zoom_factor = (gs_new / gs_old, gs_new / gs_old, 1)
    posemb_grid = zoom(posemb_grid, zoom_factor, order=1)
    posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
    posemb_grid = torch.from_numpy(posemb_grid)

    # Deal with class token and return
    posemb = torch.cat([posemb_grid, posemb_tok], dim=1)
    return posemb

class PositionEncodingLayer(torch.nn.Module):
    def __init__(self, iImSize: int, iPatchSize: int, iModelDim: int, iDim: int = 1, bClsToken: bool = True):
        super().__init__()

        if iDim == 1:
            iSz = (iImSize // iPatchSize)**2
            if bClsToken: iSz += 1
            self.tPosEmbed = torch.nn.Parameter(torch.empty(1, iSz, iModelDim).normal_(std=0.02))
        elif iDim == 2:
            self.tPosEmbed = posemb_sincos_2d(iImSize // iPatchSize, iImSize // iPatchSize, iModelDim).to(device)
            self.tPosEmbed.requires_grad = False
        else:
            print("Error! Invalid embed dimension {}!".format(iDim))
            return

        self.iDim = iDim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.iDim == 1:
            x += self.tPosEmbed
        elif self.iDim == 2:
            x[:,:-1,:] += self.tPosEmbed
        return x

class ViTInputLayer(torch.nn.Module):
    def __init__(self, iImSize: int, iPatchSize: int, iChannels: int, iModelDim: int, bClsToken: bool = True):
        super().__init__()
        self.iImSize = iImSize
        self.iPatchSize = iPatchSize
        self.iModelDim = iModelDim

        self.bGeneralizedLayer = True
        
        self.tClsToken = None
        if bClsToken: self.tClsToken = torch.nn.Parameter(torch.zeros(1, 1, self.iModelDim))
        
        self.tConv = torch.nn.Conv2d(
                in_channels=iChannels, out_channels=iModelDim, kernel_size=iPatchSize, stride=iPatchSize
            )
        
        fan_in = self.tConv.in_channels * self.tConv.kernel_size[0] * self.tConv.kernel_size[1]
        torch.nn.init.trunc_normal_(self.tConv.weight, std=math.sqrt(1 / fan_in))
        if self.tConv.bias is not None:
            torch.nn.init.zeros_(self.tConv.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, h, w = x.shape
        p = self.iPatchSize
        torch._assert(h == self.iImSize, f"Wrong image height! Expected {self.iImSize} but got {h}!")
        torch._assert(w == self.iImSize, f"Wrong image width! Expected {self.iImSize} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.tConv(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.iModelDim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)
        
        if self.tClsToken is not None:
            batch_class_token = self.tClsToken.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)

        return x

class VisionTransformer(torch.nn.Module):
    """
    Modified version of the official pytorch impl of ViTs. 
    Unfortunantely, the og version does not expose transformer layers easily, hence this version.
    This verion has a linearizable module structure
    """
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 10, #for CIFAR10
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(torch.nn.LayerNorm, eps=1e-6),
        bAvgTokens: bool = False,
        iEmbedDim: int = 1,
    ):
        super().__init__()
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.representation_size = representation_size
        self.norm_layer = norm_layer
        self.in_channels = in_channels
        self.bAvgTokens = bAvgTokens
        self.iEmbedDim = iEmbedDim

        self.tInputLayer = ViTInputLayer(self.image_size, self.patch_size, self.in_channels, self.hidden_dim)

        self.tEmbedLayer = PositionEncodingLayer(self.image_size, self.patch_size, self.hidden_dim, iDim = iEmbedDim)
        
        self.tDropout = torch.nn.Dropout(self.dropout)
        
        trLayers: OrderedDict[str, torch.nn.Module] = OrderedDict()
        for i in range(num_layers):
            trLayers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.trLayers = torch.nn.Sequential(trLayers)
        self.tLn = self.norm_layer(hidden_dim)
        self.tFlat = TransformerFlattenLayer(self.bAvgTokens)
        self.tHead = torch.nn.Linear(self.hidden_dim, self.num_classes)
        torch.nn.init.zeros_(self.tHead.weight)
        torch.nn.init.zeros_(self.tHead.bias)
        
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #Conv
        x = self.tInputLayer(x)
        #+pos embeddings
        x = self.tEmbedLayer(x)
        #dropout
        x = self.tDropout(x)
        #Actual transformer layers
        x = self.trLayers(x)
        #layer norm
        #print(x.shape) #tokens are not multiplied by channels
        x = self.tLn(x)
        
        x = self.tFlat(x)
            
        return x 
    
    def classify(self, x: torch.Tensor) -> torch.Tensor:
        #final linear layer   
        return self.tHead(x)
    
    def GetHeadGrad(self) -> torch.Tensor:
        with torch.no_grad():
            return torch.matmul(self.tHead.bias.grad.unsqueeze(0), self.tHead.weight)
        
    def LoadWeights(self, strPath: str) -> None:
        '''
        This assumes weights derived from the standard pytorch impl found at: https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py
        '''
        if not os.path.exists(strModelDir + strPath):
            print("WARNING! Unable to load weights! Double check model path: {}".format(strModelDir + strPath))
            return
        
        with open(strModelDir + strPath, "rb") as f:
            sd = torch.load(f, weights_only = False)

        #self.load_state_dict(sd) #turn this on to help w/ debug

        #we always relearn the final head
        del sd["heads.head.weight"]
        del sd["heads.head.bias"]

        #map names to this linearizable version of the ViT
        dNameMap = {
            "conv_proj.weight": "tInputLayer.tConv.weight",
            "conv_proj.bias": "tInputLayer.tConv.bias",
            "class_token": "tInputLayer.tClsToken",
            "encoder.ln.weight": "tLn.weight",
            "encoder.ln.bias": "tLn.bias",
        }
        for i in range(self.num_layers):
            strBase = "encoder.layers.encoder_layer_" + str(i) + "."
            strBaseNew = "trLayers.encoder_layer_" + str(i) + "."
            dNameMap[strBase + "ln_1.weight"] = strBaseNew + "ln_1.weight"
            dNameMap[strBase + "ln_1.bias"] = strBaseNew + "ln_1.bias"
            dNameMap[strBase + "self_attention.in_proj_weight"] = strBaseNew + "self_attention.in_proj_weight"
            dNameMap[strBase + "self_attention.in_proj_bias"] = strBaseNew + "self_attention.in_proj_bias"
            dNameMap[strBase + "self_attention.out_proj.weight"] = strBaseNew + "self_attention.out_proj.weight"
            dNameMap[strBase + "self_attention.out_proj.bias"] = strBaseNew + "self_attention.out_proj.bias"
            dNameMap[strBase + "ln_2.weight"] = strBaseNew + "ln_2.weight"
            dNameMap[strBase + "ln_2.bias"] = strBaseNew + "ln_2.bias"
            dNameMap[strBase + "mlp.linear_1.weight"] = strBaseNew + "mlp.0.weight"
            dNameMap[strBase + "mlp.linear_1.bias"] = strBaseNew + "mlp.0.bias"
            dNameMap[strBase + "mlp.linear_2.weight"] = strBaseNew + "mlp.3.weight"
            dNameMap[strBase + "mlp.linear_2.bias"] = strBaseNew + "mlp.3.bias"

        #apply the map
        for key in dNameMap.keys():
            sd[dNameMap[key]] = sd[key]
            del sd[key]

        #rescale the positional embeddings. The 2D version is not learned so need to translate that
        if self.iEmbedDim == 1:
            sd["tEmbedLayer.tPosEmbed"] = resize_positional_embedding_(sd["encoder.pos_embedding"], self.state_dict()["tEmbedLayer.tPosEmbed"])

        #rescale the input convolution layer's kernels, no rescale needed for 16
        if self.patch_size < 16:
            sd["tInputLayer.tConv.weight"] = torch.nn.functional.avg_pool2d(sd["tInputLayer.tConv.weight"], 
                                                                            kernel_size = 16 // self.patch_size, stride = 16 // self.patch_size)
        elif self.patch_size != 16:
            print("Weird patch size detected! {}".format(self.patch_size))
            return
        
        #Finally, load in the modified weights
        self.load_state_dict(sd, strict = False)

        return
  
if __name__ == "__main__":
    #t = posemb_sincos_2d(8, 8, 384)
    #print(t.shape)
    
    model = VisionTransformer(32, 4, 3, 8, 6, 192, 576, dropout=0.0, bAvgTokens = False, iEmbedDim = 1).to(device)

    #model.LoadWeights("vit_b_16.pth")

    x = torch.randn((1, 3, 32, 32)).to(device)
    PrintModelSummary(model)
    print(model(x).shape)
    print(CountParams(model))
    
    # model = TextEncTransformer(64, 100, 8, 8, 256, 768, 0.5, bAvgTokens = True).to(device)
    # x = torch.randn((1, 64, 100)).to(device)
    # f = model(x)
    # print(f.shape)
    # p = model.classify(f)
    # print(p.shape)
    