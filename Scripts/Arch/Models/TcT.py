import torch

from typing import Optional, Callable, OrderedDict
from functools import partial

try:
    from Arch.Models.TransformerBlocks import *
    from Arch.Models.ModelUtils import *
except:
    from TransformerBlocks import *
    from Arch.Models.ModelUtils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def posemb_sincos_1d(n, dim, temperature: int = 10000, dtype = torch.float32):
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    
    token_positions = torch.arange(start=0, end = n).view(-1, 1)
    dim_positions = torch.arange(start=0, end = dim).view(1, -1)
    angles = token_positions / (10000 ** ((2 * dim_positions) / dim))

    encodings = torch.zeros(1, n, dim, device = device)
    encodings[0, :, ::2] = torch.cos(angles[:, ::2])
    encodings[0, :, 1::2] = torch.sin(angles[:, 1::2])
    return encodings

class PositionEncodingLayer1D(torch.nn.Module):
    def __init__(self, iNumTokens: int, iModelDim: int):
        super().__init__()

        #self.tPosEmbed = torch.nn.Parameter(torch.empty(1, seq_length, iModelDim).normal_(std=0.02))
        self.tPosEmbed = posemb_sincos_1d(iNumTokens, iModelDim).to(device)
        self.tPosEmbed.requires_grad = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x[:,:-1,:] += self.tPosEmbed
        return x
    
class TextEncInputLayer(torch.nn.Module):
    def __init__(self, iInputDim: int, iModelDim: int):
        super().__init__()
        
        self.iModelDim = iModelDim
        self.iInputDim = iInputDim
        
        self.tClsToken = torch.nn.Parameter(torch.zeros(1, 1, self.iModelDim))
        self.tLin = torch.nn.Linear(iInputDim, iModelDim)
        
    def forward(self, x):
        n, s, e = x.shape
        torch._assert(e == self.iInputDim, "Error! Invalid token dimensionality, expected {} got {}".format(self.iInputDim, e))
        x = self.tLin(x)
        
        batch_class_token = self.tClsToken.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        return x
    
    
class TextEncTransformer(torch.nn.Module):
    """
    Modified version of the official pytorch impl of a transformer encoder. 
    Unfortunantely, the og version does not expose transformer layers easily, hence this version.
    This verion has a linearizable module structure
    """
    def __init__(
        self,
        num_tokens: int,
        input_dim: int,
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
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer
        self.bAvgTokens = bAvgTokens

        self.tInputLayer = TextEncInputLayer(self.input_dim, self.hidden_dim)
        
        self.tEmbedLayer = PositionEncodingLayer1D(self.num_tokens, self.hidden_dim)
        
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
        #lin. proj
        x = self.tInputLayer(x)
        #+pos embeddings
        x = self.tEmbedLayer(x)
        #dropout
        x = self.tDropout(x)
        #Actual transformer layers
        x = self.trLayers(x)
        #layer norm
        x = self.tLn(x)
        
        x = self.tFlat(x)     

        return x
    
    def classify(self, x: torch.Tensor) -> torch.Tensor:
        #final linear layer
        return self.tHead(x)
    
    def GetHeadGrad(self) -> torch.Tensor:
        with torch.no_grad():
            return torch.matmul(self.tHead.bias.grad.unsqueeze(0), self.tHead.weight)