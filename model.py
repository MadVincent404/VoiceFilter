"""
VoiceFilter Model - Architecture complète du papier
Paper: https://arxiv.org/abs/1810.04826

Architecture:
  - 8 couches CNN avec dilatations sur l'axe temporel
  - 1 couche LSTM (400 nodes)
  - FC1 (600) + FC2 (600) + sigmoid -> masque
"""

import torch
import torch.nn as nn
import torch.nn.functional as F_func


class VoiceFilter(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_fft_bins  = config.n_fft // 2 + 1   # 257
        self.d_vec_size  = config.d_vec_size         # 192
        self.lstm_hidden = config.lstm_hidden        # 400
        self.fc_hidden   = config.fc_hidden          # 600


        # Bloc CNN — traite le spectrogramme magnitude (B, 1, T, F)
        # Les couches CNN opèrent sur (time, freq)

        # CNN 1 : width (time=1, freq=7), dilation (1,1), 64 filtres
        # CNN 2 : width (time=7, freq=1), dilation (1,1), 64 filtres
        # CNN 3 : width (time=5, freq=5), dilation (1,1), 64 filtres
        # CNN 4 : width (time=5, freq=5), dilation (2,1), 64 filtres
        # CNN 5 : width (time=5, freq=5), dilation (4,1), 64 filtres
        # CNN 6 : width (time=5, freq=5), dilation (8,1), 64 filtres
        # CNN 7 : width (time=5, freq=5), dilation (16,1), 64 filtres
        # CNN 8 : width (time=1, freq=1), dilation (1,1), 8 filtres

        def conv_block(in_ch, out_ch, kt, kf, dt, df):
            """Conv2d + BatchNorm + ReLU avec padding 'same'."""
            pt = (kt - 1) * dt // 2
            pf = (kf - 1) * df // 2
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch,
                          kernel_size=(kt, kf),
                          dilation=(dt, df),
                          padding=(pt, pf)),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.cnn = nn.Sequential(
            conv_block(1,  64, kt=1,  kf=7,  dt=1,  df=1),   # CNN 1
            conv_block(64, 64, kt=7,  kf=1,  dt=1,  df=1),   # CNN 2
            conv_block(64, 64, kt=5,  kf=5,  dt=1,  df=1),   # CNN 3
            conv_block(64, 64, kt=5,  kf=5,  dt=2,  df=1),   # CNN 4
            conv_block(64, 64, kt=5,  kf=5,  dt=4,  df=1),   # CNN 5
            conv_block(64, 64, kt=5,  kf=5,  dt=8,  df=1),   # CNN 6
            conv_block(64, 64, kt=5,  kf=5,  dt=16, df=1),   # CNN 7
            conv_block(64, 8,  kt=1,  kf=1,  dt=1,  df=1),   # CNN 8
        )

        # Après CNN : (B, 8, T, F) -> reshape -> (B, T, 8*F)
        cnn_out_size = 8 * self.n_fft_bins   # 8 * 257 = 2056

        # Input LSTM = CNN features + d-vector
        lstm_input_size = cnn_out_size + self.d_vec_size   # 2056 + 192 = 2248

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=self.lstm_hidden,   # 400
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )

        # FC1 (600) + FC2 (600) + sigmoid
        self.fc1 = nn.Linear(self.lstm_hidden, self.fc_hidden)   # 400 -> 600
        self.fc2 = nn.Linear(self.fc_hidden,   self.fc_hidden)   # 600 -> 600
        self.fc3 = nn.Linear(self.fc_hidden,   self.n_fft_bins)  # 600 -> 257

    def forward(self, mixed_mag: torch.Tensor, d_vec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mixed_mag : (B, T, F)  magnitude spectrogram
            d_vec     : (B, D)     speaker d-vector normalisé

        Returns:
            mask      : (B, T, F)  soft mask in [0, 1]
        """
        B, T, F = mixed_mag.shape

        # --- CNN block ---
        # (B, T, F) -> (B, 1, T, F)
        x = mixed_mag.unsqueeze(1)
        x = self.cnn(x)                    # (B, 8, T, F)
        # (B, 8, T, F) -> (B, T, 8*F)
        x = x.permute(0, 2, 1, 3).reshape(B, T, -1)

        # --- Concat d-vector ---
        d = d_vec.unsqueeze(1).expand(-1, T, -1)   # (B, T, D)
        x = torch.cat([x, d], dim=-1)              # (B, T, 8*F + D)

        # --- LSTM ---
        x, _ = self.lstm(x)                        # (B, T, 400)

        # --- FC layers ---
        x    = F_func.relu(self.fc1(x))            # (B, T, 600)
        x    = F_func.relu(self.fc2(x))            # (B, T, 600)
        mask = torch.sigmoid(self.fc3(x))          # (B, T, 257)

        return mask