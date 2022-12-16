import torch
import torch.nn as nn

from .input_layer import VitInputlayer
from .encoder_block import VitEncoderBlock


class Vit(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        emb_dim: int = 384,
        num_patch_row: int = 2,
        image_size: int = 32,
        num_blocks: int = 7,
        head: int = 8,
        hidden_dim: int = 384 * 4,
        dropout: float = 0.0,
    ):

        super(Vit, self).__init__()

        self.input_layer = VitInputlayer(
            in_channels=in_channels,
            emb_dim=emb_dim,
            num_patch_row=num_patch_row,
            image_size=image_size,
        )

        self.encoder = nn.Sequential(
            *[
                VitEncoderBlock(
                    emb_dim=emb_dim,
                    head=head,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_dim), nn.Linear(emb_dim, num_classes)
        )

    def forward(self, x: torch.tensor) -> torch.Tensor:

        out = self.input_layer(x)
        out = self.encoder(out)
        print(out.shape)
        cls_token = out[:, 0]
        pred = self.mlp_head(cls_token)

        return pred
