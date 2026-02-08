import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel-wise feature recalibration."""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SleepBiLSTM_Energy(nn.Module):
    """
    Dual-stream architecture integrating Log Mel-spectrograms (CNN)
    and energy profiles (MLP) fused via a BiLSTM for temporal modeling.
    """
    def __init__(self, num_classes=3, d_model=256, rnn_layers=2):
        super().__init__()

        # Feature Extractor: 4-layer VGG-style CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # Pooling only freq dim
            SEBlock(256, reduction=16)
        )

        self.cnn_feat_dim = 256 * 8

        # Energy Branch
        self.energy_proj = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64)
        )

        # Fusion Layer
        self.fusion_proj = nn.Linear(self.cnn_feat_dim + 64, d_model)
        self.proj_norm = nn.LayerNorm(d_model)

        # Temporal Modeling
        self.rnn = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if rnn_layers > 1 else 0
        )

        self.head_dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, energy):
        # 1. Spectrogram Encoding
        x = self.cnn(x)
        B, C, F_prime, T_prime = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T_prime, C * F_prime)

        # 2. Energy Encoding
        energy = energy.unsqueeze(1)
        energy = F.adaptive_avg_pool1d(energy, T_prime)
        energy = energy.permute(0, 2, 1)
        e_feat = self.energy_proj(energy)

        # 3. Fusion
        combined = torch.cat([x, e_feat], dim=-1)
        x = self.fusion_proj(combined)
        x = self.proj_norm(x)

        # 4. Temporal Classification
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)
        x = self.head_dropout(x)

        return self.classifier(x)