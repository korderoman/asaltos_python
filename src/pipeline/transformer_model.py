import torch
import torch.nn as nn

from src.pipeline import PositionalEncoding


class TransformerModel(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers=4, nhead=4, dim_feedforward=256, dropout=0.3):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(num_features, dropout)
        encoder_layers = nn.TransformerEncoderLayer(num_features, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.fc_dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, src):
        # Asegurarse de que src tiene las dimensiones correctas
        # print(f"Entrada src shape: {src.shape}")  # [batch_size, sequence_length, num_features]

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)

        # Imprimir la forma de output después del TransformerEncoder
        # print(f"Output después del TransformerEncoder shape: {output.shape}")

        # Verificar si la salida tiene 3 dimensiones antes de indexar
        if output.dim() == 3:
            output = output[:, -1, :]  # Tomar la última salida de la secuencia
        else:
            output = output  # Si solo tiene 2 dimensiones, continuar sin indexar

        output = self.fc_dropout(output)  # Aplicar dropout antes de la capa final
        output = self.fc(output)
        return output