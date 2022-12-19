import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        emb_dim: int = 384,
        head: int = 3,
        dropout: float = 0,
        attention_activation: str = "softmax",
    ):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // head
        self.sqrt_dh = self.head_dim**0.5
        self.attention_activation = attention_activation

        self.w_q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_k = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_v = nn.Linear(emb_dim, emb_dim, bias=False)

        self.attn_drop = nn.Dropout(dropout)

        self.w_o = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.Dropout(dropout)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:

        batch_size, num_patch, _ = z.size()

        q = self.w_q(z)
        k = self.w_k(z)
        v = self.w_v(z)

        q = q.view(batch_size, num_patch, self.head, self.head_dim)
        k = k.view(batch_size, num_patch, self.head, self.head_dim)
        v = v.view(batch_size, num_patch, self.head, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        k_T = k.transpose(2, 3)
        dots = (q @ k_T) / self.sqrt_dh

        if self.attention_activation == "softmax":

            attn = F.softmax(dots, dim=-1)
            attn = self.attn_drop(attn)

            out = attn @ v

        elif self.attention_activation == "rela":
            ln = nn.LayerNorm([self.head, num_patch, self.head_dim])
            attn = F.relu(dots)
            attn = self.attn_drop(attn)

            out = ln(attn @ v)

        out = out.transpose(1, 2)
        out = out.reshape(batch_size, num_patch, self.emb_dim)

        out = self.w_o(out)

        return out
