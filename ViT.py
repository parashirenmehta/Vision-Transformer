import torch
from torch import nn
from vit_encoder import VitEncoder
from vit_blocks import VitBlocks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VisionTransformer(nn.Module):
    def __init__(self,
                 batch_size,
                 num_classes,
                 image_size,
                 patch_size,
                 in_channels,
                 embed_dim,
                 num_heads,
                 num_layers,
                 dim_feedforward=2048,  # default
                 dropout=0.1,           # default
                 activation="relu",     # default (or use gelu)
                 norm_first=False):     # default
        super().__init__()

        # N = (H * W) // (P * P) -> (image_size // patch_size) ** 2
        num_patches = (image_size // patch_size) ** 2

        # patches
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # class tokens
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim)).to(device)

        # positional embeddings (class token + num_patches)
        self.position_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim)).to(device)

        # transformer encorder layer
        encoder_layer =  VitEncoder(image_size,
                                      patch_size,
                                      embed_dim,
                                      num_heads,
                                      dim_feedforward,
                                      num_classes,
                                      batch_size)
        # transformer encorder
        self.transformer = VitBlocks(encoder_layer, num_layers).to(device)

        # mlp head
        self.mlp_head = nn.Linear(embed_dim, num_classes).to(device)

    def forward(self, x): # (B, 1, H, W)
        #print(x.shape)
        # patch embeddings
        x = self.proj(x)  # (B, E, P, P) of each patch
        #print(x.shape)
        x = x.flatten(2).transpose(1, 2)
        #print(x.shape)
        # flatten -> (B, E, P * P) transpose -> (B, P * P, E) -> (B, N, E)

        # class tokens
        class_token = self.class_token.repeat(x.shape[0], 1, 1)  # (B, 1, E)
        x = torch.cat([class_token, x], dim=1)  # (B, 1 + N, E)

        # positional embeddings
        x += self.position_embedding  # (B, 1 + N, E)

        # transformer encoder
        #x = x.transpose(0, 1)  # PyTorch transformer expects (1 + N, B, E)
        #print(x.shape)
        x = self.transformer(x)# (1 + N, B, E)
        #print('After transformer: ',x.shape)
        x = self.mlp_head(x[:,0,:])  # (B, num_classes)
        #print('After MLP head:',x.shape)

        return x  # predicted mlp_head class tokens