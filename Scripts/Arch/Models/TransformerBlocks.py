import torch
from torchvision.models.vision_transformer import MLPBlock

from typing import Callable
from functools import partial


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TransformerFlattenLayer(torch.nn.Module):
    def __init__(self, bAvgTokens: bool = True) -> None:
        super(TransformerFlattenLayer, self).__init__()
        self.bAvgTokens = bAvgTokens
        
    def forward(self, x):
        if not self.bAvgTokens:
            x = x[:, 0]
        else:
            x = torch.mean(x, dim=1)
            
        return x

class EncoderBlock(torch.nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(torch.nn.LayerNorm, eps=1e-5),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = torch.nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = torch.nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)
        
        return

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        #print(x.shape)
        # tCheck = torch.sum(x)
        # if tCheck != tCheck:
        #     print("NaNs found after ln_1!")
        #     print(x)
        #     tCheck2 = torch.sum(input)
        #     print(input)
        #     if tCheck2 != tCheck2:
        #         print("Input was NaN!") 
        #     input()

        x, _ = self.self_attention(x, x, x, need_weights=False)
        #print(x.shape)
        x = self.dropout(x)
        x = x + input
        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y
    
    def features(self, input: torch.Tensor) -> torch.Tensor:
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input
        y = self.ln_2(x)
        f = torch.nn.Sequential(*list(self.mlp)[:-2])(y)
        y = torch.nn.Sequential(*list(self.mlp)[-2:])(f)
        return x + y, f
