"""
Data preprocessing and dataset classes for Tokyo rent prediction
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder


class RentDataPreprocessor:
    """Handles data preprocessing and transformation for rent data"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.ward_avg_prices = {}
        self.ward_avg_mean = None
        self.ward_avg_std = None
        self.num_wards = None

    def fit_transform(self, df):
        """データ前処理と変換"""
        df_processed = df.copy()

        # 1. 区（ward）埋め込みのためのインデックス生成
        self.label_encoders['区'] = LabelEncoder()
        df_processed['区_encoded'] = self.label_encoders['区'].fit_transform(df['区'])
        self.num_wards = len(df['区'].unique())

        # 2. カテゴリ変数エンコーディング
        for col in ['建物構造', '建物タイプ']:
            self.label_encoders[col] = LabelEncoder()
            df_processed[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])

        # 3. 数値型変数の正規化
        numeric_cols = ['部屋サイズ_m2', '駅距離_分', '築年数_年']
        df_processed[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])

        # 4. 区別平均価格（補助情報）
        ward_avg_price = df.groupby('区')['家賃_円'].mean()
        self.ward_avg_prices = ward_avg_price.to_dict()
        df_processed['区_avg_price'] = df['区'].map(ward_avg_price)

        # 区平均価格の正規化統計を保存
        self.ward_avg_mean = df_processed['区_avg_price'].mean()
        self.ward_avg_std = df_processed['区_avg_price'].std()

        df_processed['区_avg_price'] = (df_processed['区_avg_price'] - self.ward_avg_mean) / self.ward_avg_std

        return df_processed

    def transform(self, df):
        """学習済み前処理器で変換"""
        df_processed = df.copy()

        # 区エンコーディング
        df_processed['区_encoded'] = self.label_encoders['区'].transform(df['区'])

        # カテゴリ変数エンコーディング
        for col in ['建物構造', '建物タイプ']:
            df_processed[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])

        # 数値型変数の正規化
        numeric_cols = ['部屋サイズ_m2', '駅距離_分', '築年数_年']
        df_processed[numeric_cols] = self.scaler.transform(df[numeric_cols])

        # 区別平均価格（学習済み統計を使用）
        df_processed['区_avg_price'] = df['区'].map(self.ward_avg_prices)
        df_processed['区_avg_price'] = (df_processed['区_avg_price'] - self.ward_avg_mean) / self.ward_avg_std

        return df_processed


class TokyoRentDataset(Dataset):
    """PyTorch Dataset wrapper for Tokyo rent data"""

    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
