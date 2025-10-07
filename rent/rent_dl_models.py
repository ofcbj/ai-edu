"""
Neural Network Models for Rent Prediction
Contains two model architectures:
1. RentPredictionNet - Basic deep learning model with embeddings
2. RentPredictionNetWithAttention - Advanced model with attention mechanism
"""

import torch
import torch.nn as nn


class RentPredictionNet(nn.Module):
    """Basic deep learning model for rent prediction with embeddings"""

    def __init__(self, num_wards, num_structures, num_types,
                 embedding_dim=16, hidden_dims=[256, 128, 64]):
        super(RentPredictionNet, self).__init__()

        # Embedding layers (learn ward characteristics)
        self.ward_embedding     = nn.Embedding(num_wards, embedding_dim)
        self.structure_embedding= nn.Embedding(num_structures, embedding_dim // 2)
        self.type_embedding     = nn.Embedding(num_types, embedding_dim // 2)

        # Calculate input dimensions
        # 3 numeric features + ward embedding + structure embedding + type embedding + ward average price
        input_dim = 3 + embedding_dim + (embedding_dim // 2) * 2 + 1

        # Main network
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.1)

    def forward(self, ward_idx, structure_idx, type_idx, numeric_features, ward_avg_price):
        # Embeddings
        ward_emb        = self.ward_embedding(ward_idx)
        structure_emb   = self.structure_embedding(structure_idx)
        type_emb        = self.type_embedding(type_idx)

        # Concatenate all features
        features = torch.cat([
            numeric_features,
            ward_emb,
            structure_emb,
            type_emb,
            ward_avg_price.unsqueeze(1)
        ], dim=1)

        # Pass through network
        output = self.network(features)

        return output


class RentPredictionNetWithAttention(nn.Module):
    """Advanced deep learning model with attention mechanism for rent prediction"""

    def __init__(self, num_wards, num_structures, num_types,
                 embedding_dim=32, hidden_dims=[512, 256, 128]):
        super(RentPredictionNetWithAttention, self).__init__()

        self.num_wards      = num_wards
        self.embedding_dim  = embedding_dim
        self.hidden_dims    = hidden_dims

        # Embedding layers
        self.ward_embedding     = nn.Embedding(num_wards, embedding_dim)
        self.structure_embedding= nn.Embedding(num_structures, embedding_dim // 2)
        self.type_embedding     = nn.Embedding(num_types, embedding_dim // 2)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.Tanh(),
            nn.Linear(embedding_dim // 2, 1),
            nn.Sigmoid()
        )

        # Interaction layers
        self.ward_room_interaction      = nn.Linear(embedding_dim + 1, embedding_dim)
        self.ward_station_interaction   = nn.Linear(embedding_dim + 1, embedding_dim)

        # Input dimensions
        input_dim = 3 + embedding_dim * 3 + (embedding_dim // 2) * 2 + 1

        # Main network
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.25)
            ])
            prev_dim = hidden_dim

        # Skip connection
        self.skip_connection = nn.Linear(input_dim, hidden_dims[-1])

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, 1)
        )

        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, ward_idx, structure_idx, type_idx, numeric_features, ward_avg_price):
        # Generate embeddings
        ward_emb            = self.ward_embedding(ward_idx)
        structure_emb       = self.structure_embedding(structure_idx)
        type_emb            = self.type_embedding(type_idx)

        # Apply attention
        attention_weights   = self.attention(ward_emb)
        ward_emb_attended   = ward_emb * attention_weights

        # Generate interaction features
        room_size           = numeric_features[:, 0:1]
        station_dist        = numeric_features[:, 1:2]

        ward_room_feat      = self.ward_room_interaction(torch.cat([ward_emb, room_size], dim=1))
        ward_station_feat   = self.ward_station_interaction(torch.cat([ward_emb, station_dist], dim=1))

        # Concatenate all features
        features = torch.cat([
            numeric_features,
            ward_emb_attended,
            ward_room_feat,
            ward_station_feat,
            structure_emb,
            type_emb,
            ward_avg_price.unsqueeze(1) if ward_avg_price.dim() == 1 else ward_avg_price
        ], dim=1)

        # Forward pass with skip connection
        main_output = self.network(features)
        skip_output = self.skip_connection(features)
        combined = main_output + skip_output * 0.1

        output = self.output_layer(combined)

        # Return additional information for visualization when not training
        if self.training:
            return output
        else:
            return output, attention_weights, ward_emb
