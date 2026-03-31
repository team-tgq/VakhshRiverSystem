import torch.nn as nn

FEATURE_COLS_FINAL = [
    "smlt",
    "ssrd",
    "e",
    "u10",
    "v10",
    "sp",
    "skt",
    "day_of_year_sin",
    "day_of_year_cos",
]


class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_steps: int):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.decoder_fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.decoder_fc2 = nn.Linear(hidden_size, output_steps)

    def forward(self, x):
        _, (h_n, _) = self.encoder(x)
        context_vector = h_n[-1, :, :]
        dec_out = self.decoder_fc1(context_vector)
        dec_out = self.relu(dec_out)
        dec_out = self.dropout(dec_out)
        return self.decoder_fc2(dec_out)
