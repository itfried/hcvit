import torch
torch.cuda.empty_cache()
from torch import nn
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import torch.nn.functional as F
import spectral
from torchvision.transforms import Resize


import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class HCViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, emb_dim = 1024, patch_channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        # assert channels % patch_channels == 0, 'Image channels must be divisible by the patch channels.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = patch_channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # Conv Block
        self.act = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(channels, emb_dim, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(emb_dim)
        self.conv2 = nn.Conv2d(emb_dim, emb_dim, kernel_size=3, padding=1, groups=emb_dim, bias=False)
        self.bn3 = nn.BatchNorm2d(3*emb_dim)
        self.conv3 = nn.Conv2d(3*emb_dim, patch_channels, kernel_size=1, bias=False)

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        # Conv Block
        x0 = self.conv1(img)
        x1 = self.conv2(self.act(self.bn2(x0)))
        x2 = torch.cat([x0, x1, x0], axis=1)
        x = self.conv3(self.act(self.bn3(x2)))

        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


def get_model(in_channels=3, pretrained=False, num_classes=2, **kwargs):
    c_image_size = 80  # vnir
    c_patch_size = 16
    c_num_classes = num_classes
    c_dim = 1280
    c_depth = 24
    c_heads = 16
    c_mlp_dim = 5120
    c_dropout = 0.1
    c_patch_channels = 84
    c_emb_dim = 420
    c_channels = 840  # vnir
    c_emb_dropout = 0.1

    model = HCViT(
        image_size=c_image_size,
        patch_size=c_patch_size,
        num_classes=c_num_classes,
        dim=c_dim,
        depth=c_depth,
        heads=c_heads,
        mlp_dim=c_mlp_dim,
        dropout=c_dropout,
        channels=c_channels,  # vnir
        emb_dim=c_emb_dim,
        patch_channels=c_patch_channels,
        emb_dropout=c_emb_dropout
    )

    print("model_params are: \n image_size = ", c_image_size,
          "\n patch_size = ", c_patch_size,
          "\n num_classes = ", c_num_classes,
          "\n dim = ", c_dim,
          "\n depth = ", c_depth,
          "\n heads = ", c_heads,
          "\n mlp_dim = ", c_mlp_dim,
          "\n dropout = ", c_dropout,
          "\n channels = ", c_channels,
          "\n emb_dim = ", c_emb_dim,
          "\n patch_channels = ", c_patch_channels,
          "\n emb_dropout = ", c_emb_dropout
          )

    return model


def main():
    model = get_model(pretrained=False, in_channels=256, num_classes=2)
    model.eval()
    sample_image = torch.randn(1, 1096, 80, 80)
    
    print(model(sample_image))
    probabilities = F.softmax(model(sample_image), dim=1)
    print(probabilities)


if __name__ == "__main__":
    main()















