"""Seq2SeqLSTM 模型定义 — 日度气象特征 → 月度径流预测"""
import torch
import torch.nn as nn

# 特征列: 8个 ERA5 原始特征 + 2个日历循环编码 = 10 个特征
FEATURE_COLS_RAW = ['discharge', 'smlt', 'ssrd', 'e', 'u10', 'v10', 'sp', 'skt']
FEATURE_COLS_FINAL = FEATURE_COLS_RAW + ['day_of_year_sin', 'day_of_year_cos']


class Seq2SeqLSTM(nn.Module):
    """Encoder-Decoder LSTM: 365天日度序列 → 12月月度径流"""

    def __init__(self, input_size, hidden_size, num_layers, output_steps):
        super(Seq2SeqLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_steps = output_steps

        self.encoder = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True, dropout=0.2)
        self.decoder_fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.decoder_fc2 = nn.Linear(hidden_size, output_steps)

    def forward(self, x):
        encoder_out, (h_n, c_n) = self.encoder(x)
        context_vector = h_n[-1, :, :]
        dec_out = self.decoder_fc1(context_vector)
        dec_out = self.relu(dec_out)
        dec_out = self.dropout(dec_out)
        predictions = self.decoder_fc2(dec_out)
        return predictions
