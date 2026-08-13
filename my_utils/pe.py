# positional encoding
import torch
import torch.nn as nn

class PositionalEncoding1D(nn.Module):
    def __init__(self, max_len, emb_dim, dropout_p: float = 0.1):
        super(PositionalEncoding1D, self).__init__()
        self.dropout = nn.Dropout(p=dropout_p)

        pos = torch.arange(max_len).unsqueeze(1)
        den = torch.pow(10000, torch.arange(0, emb_dim, 2) / emb_dim)

        pe = torch.zeros(1, max_len, emb_dim)
        pe[0, :, 0::2] = torch.sin(pos / den)
        pe[0, :, 1::2] = torch.cos(pos / den)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x.shape = [batch_size, sec_len, emb_dim]
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class PositionalEncoding2D(nn.Module):
    def __init__(self, num_channels, max_height, max_width, dropout_p: float = 0.1):
        super(PositionalEncoding2D, self).__init__()
        self.dropout = nn.Dropout(p=dropout_p)

        pos_h = torch.arange(max_height).unsqueeze(1)
        pos_w = torch.arange(max_width).unsqueeze(1)
        den = torch.pow(10000, torch.arange(0, num_channels // 2, 2) / num_channels)

        pe = torch.zeros(1, max_height, max_width, num_channels)
        pe[0, :, :, 0 : num_channels // 2 : 2] = (
            torch.sin(pos_w / den).unsqueeze(0).repeat(max_height, 1, 1)
        )
        pe[0, :, :, 1 : num_channels // 2 : 2] = (
            torch.cos(pos_w / den).unsqueeze(0).repeat(max_height, 1, 1)
        )
        pe[0, :, :, num_channels // 2 :: 2] = (
            torch.sin(pos_h / den).unsqueeze(1).repeat(1, max_width, 1)
        )
        pe[0, :, :, (num_channels // 2) + 1 :: 2] = (
            torch.cos(pos_h / den).unsqueeze(1).repeat(1, max_width, 1)
        )
        pe = pe.permute(0, 3, 1, 2).contiguous()
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x.shape = [batch_size, num_channels, h, w]
        x = x + self.pe[:, :, : x.size(2), : x.size(3)]
        return self.dropout(x)