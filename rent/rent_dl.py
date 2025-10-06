#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Meiryo']
plt.rcParams['axes.unicode_minus'] = False

import matplotlib.font_manager as fmt
fmt.fontManager.addfont(r'rent/meiryo.ttc')

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print("=" * 80)


# In[2]:


# ===========================
# 1. データ読み込みと前処理
# ===========================
class RentDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.ward_embedding_dict = {}
        
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
        df_processed['区_avg_price'] = df['区'].map(ward_avg_price)
        df_processed['区_avg_price'] = (df_processed['区_avg_price'] - df_processed['区_avg_price'].mean()) / df_processed['区_avg_price'].std()
        
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
        
        return df_processed

# ===========================
# 2. PyTorch Datasetクラス
# ===========================
class TokyoRentDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# ===========================
# 3. ディープラーニングモデル定義
# ===========================
class RentPredictionNet(nn.Module):
    def __init__(self, num_wards, num_structures, num_types, 
                 embedding_dim=16, hidden_dims=[256, 128, 64]):
        super(RentPredictionNet, self).__init__()
        
        # 埋め込み層（区の特性を学習）
        self.ward_embedding     = nn.Embedding(num_wards, embedding_dim)
        self.structure_embedding= nn.Embedding(num_structures, embedding_dim // 2)
        self.type_embedding     = nn.Embedding(num_types, embedding_dim // 2)
        
        # 入力次元計算
        # 数値型3個 + 区埋め込み + 構造埋め込み + タイプ埋め込み + 区平均価格
        input_dim = 3 + embedding_dim + (embedding_dim // 2) * 2 + 1
        
        # メインネットワーク
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
        
        # 出力層
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # 重み初期化
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
        # 埋め込み
        ward_emb = self.ward_embedding(ward_idx)
        structure_emb = self.structure_embedding(structure_idx)
        type_emb = self.type_embedding(type_idx)
        
        # 全特徴量を結合
        features = torch.cat([
            numeric_features,
            ward_emb,
            structure_emb,
            type_emb,
            ward_avg_price.unsqueeze(1)
        ], dim=1)
        
        # ネットワークに通す
        output = self.network(features)
        
        return output

# ===========================
# 4. Attention基盤の高度なモデル
# ===========================
class RentPredictionNetWithAttention(nn.Module):
    """Attentionメカニズムを含むディープラーニング家賃予測モデル"""
    def __init__(self, num_wards, num_structures, num_types,
                 embedding_dim=32, hidden_dims=[512, 256, 128]):
        super(RentPredictionNetWithAttention, self).__init__()

        self.num_wards = num_wards
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims

        # 埋め込み層
        self.ward_embedding = nn.Embedding(num_wards, embedding_dim)
        self.structure_embedding = nn.Embedding(num_structures, embedding_dim // 2)
        self.type_embedding = nn.Embedding(num_types, embedding_dim // 2)

        # Attentionメカニズム
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.Tanh(),
            nn.Linear(embedding_dim // 2, 1),
            nn.Softmax(dim=1)
        )

        # 交互作用層
        self.ward_room_interaction = nn.Linear(embedding_dim + 1, embedding_dim)
        self.ward_station_interaction = nn.Linear(embedding_dim + 1, embedding_dim)

        # 入力次元
        input_dim = 3 + embedding_dim * 3 + (embedding_dim // 2) * 2 + 1

        # メインネットワーク
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

        # 出力レイヤ
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
        # 埋め込み生成
        ward_emb = self.ward_embedding(ward_idx)
        structure_emb = self.structure_embedding(structure_idx)
        type_emb = self.type_embedding(type_idx)

        # Attention適用
        attention_weights = self.attention(ward_emb)
        ward_emb_attended = ward_emb * attention_weights

        # 交互作用特徴量生成
        room_size = numeric_features[:, 0:1]
        station_dist = numeric_features[:, 1:2]

        ward_room_feat = self.ward_room_interaction(torch.cat([ward_emb, room_size], dim=1))
        ward_station_feat = self.ward_station_interaction(torch.cat([ward_emb, station_dist], dim=1))

        # 全特徴量を結合
        features = torch.cat([
            numeric_features,
            ward_emb_attended,
            ward_room_feat,
            ward_station_feat,
            structure_emb,
            type_emb,
            ward_avg_price.unsqueeze(1) if ward_avg_price.dim() == 1 else ward_avg_price
        ], dim=1)

        # Skip接続でForward pass
        main_output = self.network(features)
        skip_output = self.skip_connection(features)
        combined = main_output + skip_output * 0.1

        output = self.output_layer(combined)

        # 시각화를 위한 추가 반환 (필요한 경우에만 사용)
        if self.training:
            return output
        else:
            return output, attention_weights, ward_emb

# ===========================
# 5. 学習関数
# ===========================
def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001, model_save_path='best_rent_model.pth'):
    """モデル学習"""
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            # 特徴量分離
            ward_idx = batch_features[:, 0].long()
            structure_idx = batch_features[:, 1].long()
            type_idx = batch_features[:, 2].long()
            numeric_features = batch_features[:, 3:6]
            ward_avg_price = batch_features[:, 6]

            optimizer.zero_grad()
            outputs = model(ward_idx, structure_idx, type_idx, numeric_features, ward_avg_price)
            # Training 모드에서는 outputs만 반환됨
            loss = criterion(outputs.squeeze(), batch_targets)
            loss.backward()

            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        # 検証
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)

                ward_idx = batch_features[:, 0].long()
                structure_idx = batch_features[:, 1].long()
                type_idx = batch_features[:, 2].long()
                numeric_features = batch_features[:, 3:6]
                ward_avg_price = batch_features[:, 6]

                outputs = model(ward_idx, structure_idx, type_idx, numeric_features, ward_avg_price)
                # Eval 모드에서는 tuple 반환되므로 첫 번째 요소만 사용
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = criterion(outputs.squeeze(), batch_targets)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # 学習率スケジューリング
        scheduler.step(avg_val_loss)

        # 最良モデル保存
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)

        #if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    return train_losses, val_losses

def produce_rent_model():
    print("=" * 80)
    print("1. データ読み込み")
    print("=" * 80)
    
    # データ読み込み
    df = pd.read_csv('rent/tokyo_rent_data_v2.csv')
    print(f"データ数: {len(df)}")
    print(f"特徴量: {df.columns.tolist()}")
    
    # 前処理
    print("\n" + "=" * 80)
    print("2. データ前処理")
    print("=" * 80)
    
    preprocessor = RentDataPreprocessor()
    df_processed = preprocessor.fit_transform(df)
    
    # 特徴量とターゲット分離
    feature_cols = ['区_encoded', '建物構造_encoded', '建物タイプ_encoded',
                    '部屋サイズ_m2', '駅距離_分', '築年数_年', '区_avg_price']
    
    X = df_processed[feature_cols].values
    y = df['家賃_円'].values / 10000  # 万単位でスケーリング
    
    # データ分割
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"訓練データ: {len(X_train)}")
    print(f"検証データ: {len(X_val)}")
    print(f"テストデータ: {len(X_test)}")
    
    # DataLoader生成
    train_dataset = TokyoRentDataset(X_train, y_train)
    val_dataset = TokyoRentDataset(X_val, y_val)
    test_dataset = TokyoRentDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # モデル生成
    print("\n" + "=" * 80)
    print("3. モデル構築")
    print("=" * 80)
    
    num_wards = len(preprocessor.label_encoders['区'].classes_)
    num_structures = len(preprocessor.label_encoders['建物構造'].classes_)
    num_types = len(preprocessor.label_encoders['建物タイプ'].classes_)
    
    # 기본 모델과 Attention 모델 비교
    print("\n基本モデルを学習中...")
    model_basic = RentPredictionNet(num_wards, num_structures, num_types).to(device)
    train_losses_basic, val_losses_basic = train_model(model_basic, train_loader, val_loader,
                                                        num_epochs=50, model_save_path='best_rent_model_basic.pth')

    print("\nAttentionモデルを学習中...")
    model_attention = RentPredictionNetWithAttention(num_wards, num_structures, num_types).to(device)
    train_losses_att, val_losses_att = train_model(model_attention, train_loader, val_loader,
                                                    num_epochs=50, model_save_path='best_rent_model_attention.pth')

    # 最終評価
    print("\n" + "=" * 80)
    print("4. モデル評価")
    print("=" * 80)

    # 評価用ヘルパー関数
    def evaluate_model(model, test_loader, model_name):
        model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features = batch_features.to(device)

                ward_idx = batch_features[:, 0].long()
                structure_idx = batch_features[:, 1].long()
                type_idx = batch_features[:, 2].long()
                numeric_features = batch_features[:, 3:6]
                ward_avg_price = batch_features[:, 6]

                outputs = model(ward_idx, structure_idx, type_idx, numeric_features, ward_avg_price)
                # 評価モード時は(output, attention_weights, ward_emb)のtupleを返すため、最初の要素のみ使用
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                predictions.extend(outputs.cpu().numpy().flatten() * 10000)
                actuals.extend(batch_targets.numpy() * 10000)

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # 性能指標
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        r2 = 1 - (np.sum((actuals - predictions) ** 2) / np.sum((actuals - np.mean(actuals)) ** 2))

        print(f"\n{model_name} テストデータ性能:")
        print(f"  MAE: ¥{mae:,.0f}")
        print(f"  RMSE: ¥{rmse:,.0f}")
        print(f"  R² Score: {r2:.4f}")

        return predictions, actuals, mae, rmse, r2

    # 基本モデル評価
    model_basic.load_state_dict(torch.load('best_rent_model_basic.pth',
                                           map_location=device,
                                           weights_only=True))
    predictions_basic, actuals_basic, mae_basic, rmse_basic, r2_basic = evaluate_model(
        model_basic, test_loader, "基本モデル")

    # Attentionモデル評価
    model_attention.load_state_dict(torch.load('best_rent_model_attention.pth',
                                               map_location=device,
                                               weights_only=True))
    predictions_att, actuals_att, mae_att, rmse_att, r2_att = evaluate_model(
        model_attention, test_loader, "Attentionモデル")

    # モデル比較
    print("\n" + "=" * 80)
    print("モデル性能比較")
    print("=" * 80)
    print(f"{'指標':<15} {'基本モデル':<20} {'Attentionモデル':<20} {'改善率':<15}")
    print("-" * 80)
    print(f"{'MAE (¥)':<15} {mae_basic:>15,.0f}    {mae_att:>15,.0f}    {(mae_basic-mae_att)/mae_basic*100:>10.2f}%")
    print(f"{'RMSE (¥)':<15} {rmse_basic:>15,.0f}    {rmse_att:>15,.0f}    {(rmse_basic-rmse_att)/rmse_basic*100:>10.2f}%")
    print(f"{'R² Score':<15} {r2_basic:>15.4f}    {r2_att:>15.4f}    {(r2_att-r2_basic)/r2_basic*100:>10.2f}%")

    # 可視化用 변수 (Attention 모델 결과 사용)
    predictions = predictions_att
    actuals = actuals_att
    mae = mae_att
    rmse = rmse_att
    r2 = r2_att
    
    # 可視化
    print("\n" + "=" * 80)
    print("5. 結果可視化")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 学習曲線
    axes[0, 0].plot(train_losses_att, label='Train Loss', alpha=0.7)
    axes[0, 0].plot(val_losses_att, label='Validation Loss', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training History (Attention Model)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 予測 vs 実測値
    axes[0, 1].scatter(actuals, predictions, alpha=0.5)
    axes[0, 1].plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('Actual Rent (¥)')
    axes[0, 1].set_ylabel('Predicted Rent (¥)')
    axes[0, 1].set_title(f'Predictions vs Actual (R² = {r2:.4f})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 残差プロット
    residuals = actuals - predictions
    axes[1, 0].scatter(predictions, residuals, alpha=0.5)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Predicted Rent (¥)')
    axes[1, 0].set_ylabel('Residuals (¥)')
    axes[1, 0].set_title('Residual Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 区別埋め込み可視化（2D projection）
    ward_embeddings = model_attention.ward_embedding.weight.detach().cpu().numpy()
    ward_names = preprocessor.label_encoders['区'].classes_
    
    # PCAで2D投影
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    ward_embeddings_2d = pca.fit_transform(ward_embeddings)
    
    # 区別平均価格で色指定
    ward_avg_prices = df.groupby('区')['家賃_円'].mean()
    colors = [ward_avg_prices[ward] for ward in ward_names]
    
    scatter = axes[1, 1].scatter(ward_embeddings_2d[:, 0], ward_embeddings_2d[:, 1], 
                                 c=colors, cmap='RdYlBu_r', s=100, alpha=0.7)
    
    # 一部の区名を表示
    for i, ward in enumerate(ward_names):
        if ward in ['港区', '千代田区', '渋谷区', '足立区', '葛飾区']:
            axes[1, 1].annotate(ward, (ward_embeddings_2d[i, 0], ward_embeddings_2d[i, 1]),
                              fontsize=9, ha='center')
    
    axes[1, 1].set_xlabel('Embedding Dimension 1')
    axes[1, 1].set_ylabel('Embedding Dimension 2')
    axes[1, 1].set_title('Ward Embeddings Visualization')
    plt.colorbar(scatter, ax=axes[1, 1], label='Average Rent')
    
    plt.tight_layout()
    plt.savefig('deep_learning_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # モデルと前処理器を保存
    # 基本モデル
    torch.save({
        'model_state_dict': model_basic.state_dict(),
        'preprocessor': preprocessor,
        'model_config': {
            'num_wards': num_wards,
            'num_structures': num_structures,
            'num_types': num_types
        }
    }, 'rent_prediction_model_basic.pth')

    # Attentionモデル
    torch.save({
        'model_state_dict': model_attention.state_dict(),
        'preprocessor': preprocessor,
        'model_config': {
            'num_wards': num_wards,
            'num_structures': num_structures,
            'num_types': num_types
        }
    }, 'rent_prediction_model_attention.pth')

    print("\nモデル保存完了:")
    print("  - rent_prediction_model_basic.pth (基本モデル)")
    print("  - rent_prediction_model_attention.pth (Attentionモデル)")

    return model_basic, model_attention, preprocessor



# In[3]:


model_basic, model_attention, preprocessor = produce_rent_model()


# In[4]:


# ===========================
# 2. 予測器クラス
# ===========================
class DeepLearningRentPredictor:
    """ディープラーニングモデル読み込みと予測実行クラス"""

    def __init__(self, model_type='attention'):
        """モデルと前処理器を初期化

        Parameters:
        -----------
        model_type : str
            'basic' または 'attention'
        """
        self.model = None
        self.preprocessor = None
        self.ward_avg_prices = {}
        self.global_mean_price = 0
        self.global_std_price = 1
        self.model_loaded = False
        self.model_type = model_type

        try:
            self.load_model(model_type)
        except Exception as e:
            print(f"⚠️ モデル読み込みエラー: {e}")
            print("モデルファイルが存在しない場合は、先に学習コードを実行してください。")

    def load_model(self, model_type='attention'):
        """保存されたモデルを読み込み

        Parameters:
        -----------
        model_type : str
            'basic' または 'attention'
        """
        try:
            # モデルファイルパス選択
            if model_type == 'basic':
                model_path = 'rent_prediction_model_basic.pth'
                model_name = "基本モデル"
            else:  # 'attention'
                model_path = 'rent_prediction_model_attention.pth'
                model_name = "Attentionモデル"

            # チェックポイント読み込み（PyTorch 2.6+対応）
            checkpoint = torch.load(model_path,
                                  map_location=device,
                                  weights_only=False)  # 信頼できるソースなのでFalseに設定

            self.preprocessor = checkpoint['preprocessor']
            config = checkpoint['model_config']

            # モデル再構成
            if model_type == 'basic':
                self.model = RentPredictionNet(
                    config['num_wards'],
                    config['num_structures'],
                    config['num_types']
                ).to(device)
            else:  # 'attention'
                self.model = RentPredictionNetWithAttention(
                    config['num_wards'],
                    config['num_structures'],
                    config['num_types']
                ).to(device)

            # 重み読み込み
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            # 区別平均価格を計算
            df = pd.read_csv('rent/tokyo_rent_data_v2.csv')
            self.ward_avg_prices = df.groupby('区')['家賃_円'].mean().to_dict()
            self.global_mean_price = df['家賃_円'].mean()
            self.global_std_price = df['家賃_円'].std()

            self.model_loaded = True
            print(f"✅ ディープラーニングモデル読み込み成功!")
            print(f"   - モデルタイプ: {model_name}")
            print(f"   - デバイス: {device}")

        except FileNotFoundError:
            print(f"⚠️ モデルファイルが見つかりません: {model_path}")
            print("先にrent_dl.pyを実行してください。")
            raise
    
    def predict(self, ward, room_size, station_distance, building_age, structure, property_type):
        """単一物件予測"""
        if not self.model_loaded:
            return 0
        
        try:
            # 入力エンコーディング
            ward_encoded = self.preprocessor.label_encoders['区'].transform([ward])[0]
            structure_encoded = self.preprocessor.label_encoders['建物構造'].transform([structure])[0]
            type_encoded = self.preprocessor.label_encoders['建物タイプ'].transform([property_type])[0]
            
            # 数値特徴量の正規化
            numeric_features = np.array([[room_size, station_distance, building_age]])
            numeric_features_scaled = self.preprocessor.scaler.transform(numeric_features)
            
            # 区平均価格の正規化
            ward_avg = self.ward_avg_prices.get(ward, self.global_mean_price)
            ward_avg_normalized = (ward_avg - self.global_mean_price) / self.global_std_price
            
            # PyTorchテンソルに変換
            with torch.no_grad():
                ward_idx = torch.LongTensor([ward_encoded]).to(device)
                structure_idx = torch.LongTensor([structure_encoded]).to(device)
                type_idx = torch.LongTensor([type_encoded]).to(device)
                numeric_feat = torch.FloatTensor(numeric_features_scaled).to(device)
                ward_avg_price = torch.FloatTensor([ward_avg_normalized]).to(device)
                
                # 予測実行
                result = self.model(ward_idx, structure_idx, type_idx, numeric_feat, ward_avg_price)
                # Eval 모드에서는 tuple 반환 (output, attention_weights, ward_emb)
                if isinstance(result, tuple):
                    output = result[0]
                else:
                    output = result
                prediction = output.item() * 10000  # 元のスケールに復元
            
            return max(prediction, 20000)  # 最小値補正
            
        except Exception as e:
            print(f"予測エラー: {e}")
            return 0
    
    def batch_predict(self, conditions_list):
        """バッチ予測"""
        if not self.model_loaded:
            return []
        
        predictions = []
        for conditions in conditions_list:
            pred = self.predict(**conditions)
            predictions.append(pred)
        
        return predictions
    
    def get_ward_embeddings(self):
        """区埋め込みベクトルを返す"""
        if not self.model_loaded:
            return None, None
        
        with torch.no_grad():
            embeddings = self.model.ward_embedding.weight.cpu().numpy()
            ward_names = self.preprocessor.label_encoders['区'].classes_
        
        return embeddings, ward_names
    
    def get_attention_weights(self, ward):
        """特定の区のAttention重みを計算"""
        if not self.model_loaded:
            return None
        
        try:
            ward_encoded = self.preprocessor.label_encoders['区'].transform([ward])[0]
            ward_idx = torch.LongTensor([ward_encoded]).to(device)
            
            with torch.no_grad():
                ward_emb = self.model.ward_embedding(ward_idx)
                attention_weights = self.model.attention(ward_emb)
            
            return attention_weights.cpu().numpy()[0]
            
        except Exception as e:
            print(f"Attention計算エラー: {e}")
            return None

# ===========================
# 3. インタラクティブUIクラス
# ===========================
class InteractiveRentPredictorDL:
    """ディープラーニングベースのインタラクティブ家賃予測UI"""
    
    def __init__(self):
        """初期化とUI設定"""
        self.predictor_basic = None
        self.predictor_attention = None
        self.current_predictor = None
        self.setup_ward_info()
        self.setup_ui_widgets()
        self.setup_event_handlers()
        # デフォルトでAttentionモデルを読み込み
        self.load_predictors('attention')
    
    def setup_ward_info(self):
        """区情報を設定"""
        self.ward_list = [
            ('港区', '超高級', '#FF1744', 200000),
            ('千代田区', '超高級', '#FF1744', 180000),
            ('中央区', '超高級', '#FF1744', 160000),
            ('渋谷区', '超高級', '#FF1744', 150000),
            ('目黒区', '高級', '#FF6F00', 130000),
            ('文京区', '高級', '#FF6F00', 125000),
            ('新宿区', '高級', '#FF6F00', 120000),
            ('品川区', '高級', '#FF6F00', 115000),
            ('世田谷区', '高級', '#FF6F00', 110000),
            ('豊島区', '中価格', '#2196F3', 90000),
            ('台東区', '中価格', '#2196F3', 85000),
            ('中野区', '中価格', '#2196F3', 85000),
            ('杉並区', '中価格', '#2196F3', 85000),
            ('江東区', '中価格', '#2196F3', 80000),
            ('大田区', '中価格', '#2196F3', 80000),
            ('墨田区', '中価格', '#2196F3', 75000),
            ('練馬区', '中価格', '#2196F3', 75000),
            ('北区', '低価格', '#4CAF50', 60000),
            ('板橋区', '低価格', '#4CAF50', 55000),
            ('荒川区', '低価格', '#4CAF50', 50000),
            ('江戸川区', '低価格', '#4CAF50', 50000),
            ('葛飾区', '低価格', '#4CAF50', 45000),
            ('足立区', '低価格', '#4CAF50', 40000)
        ]
    
    def setup_ui_widgets(self):
        """UIウィジェット生成"""
        style = {'description_width': '140px'}
        layout_long = widgets.Layout(width='550px')
        layout_short = widgets.Layout(width='350px')

        # モデル選択ウィジェット
        self.model_selector = widgets.RadioButtons(
            options=[('基本モデル', 'basic'), ('Attentionモデル (推奨)', 'attention')],
            value='attention',
            description='🤖 モデル：',
            style=style,
            layout=layout_long
        )

        # 入力ウィジェット
        self.ward_dropdown = widgets.Dropdown(
            options=[f"{ward} ({level})" for ward, level, _, _ in self.ward_list],
            value='新宿区 (高級)',
            description='🏢 区選択：',
            style=style,
            layout=layout_long
        )
        
        self.room_size_slider = widgets.IntSlider(
            value=30, min=15, max=100, step=5,
            description='📏 部屋サイズ：',
            style=style,
            layout=layout_long
        )
        self.room_size_label = widgets.Label(value='30 m²')
        
        self.station_distance_slider = widgets.IntSlider(
            value=5, min=1, max=20, step=1,
            description='🚉 駅距離：',
            style=style,
            layout=layout_long
        )
        self.station_label = widgets.Label(value='5 分')
        
        self.building_age_slider = widgets.IntSlider(
            value=10, min=0, max=50, step=1,
            description='🏗️ 築年数：',
            style=style,
            layout=layout_long
        )
        self.age_label = widgets.Label(value='10 年')
        
        self.structure_dropdown = widgets.Dropdown(
            options=['木造', 'RC造', '鉄骨造', 'SRC造'],
            value='RC造',
            description='🏢 建物構造：',
            style=style,
            layout=layout_short
        )
        
        self.property_type_dropdown = widgets.Dropdown(
            options=['マンション', 'アパート', 'ハイツ', 'コーポ'],
            value='マンション',
            description='🏠 建物タイプ：',
            style=style,
            layout=layout_short
        )
        
        # 모드 선택
        self.mode_tabs = widgets.Tab()
        self.single_mode = widgets.VBox([widgets.HTML("<p>単一物件の家賃を予測</p>")])
        self.compare_mode = widgets.VBox([widgets.HTML("<p>複数区で同条件の家賃を比較</p>")])
        self.analysis_mode = widgets.VBox([widgets.HTML("<p>AIモデルの内部分析</p>")])
        
        self.mode_tabs.children = [self.single_mode, self.compare_mode, self.analysis_mode]
        self.mode_tabs.set_title(0, '🏠 単一予測')
        self.mode_tabs.set_title(1, '📊 区別比較')
        self.mode_tabs.set_title(2, '🧠 AI分析')
        
        # 予測ボタン
        self.predict_button = widgets.Button(
            description='🤖 AI予測実行',
            button_style='primary',
            tooltip='ディープラーニングで予測',
            layout=widgets.Layout(width='200px', height='45px')
        )
        
        # 出力エリア
        self.output = widgets.Output()
    
    def setup_event_handlers(self):
        """イベントハンドラを設定"""
        # スライダーラベル更新
        self.room_size_slider.observe(self._update_room_label, names='value')
        self.station_distance_slider.observe(self._update_station_label, names='value')
        self.building_age_slider.observe(self._update_age_label, names='value')

        # モデル選択変更
        self.model_selector.observe(self._on_model_change, names='value')

        # 予測ボタン
        self.predict_button.on_click(self.on_predict_click)

    def load_predictors(self, model_type):
        """予測器を読み込み"""
        try:
            if model_type == 'basic':
                if self.predictor_basic is None:
                    self.predictor_basic = DeepLearningRentPredictor(model_type='basic')
                self.current_predictor = self.predictor_basic
            else:  # 'attention'
                if self.predictor_attention is None:
                    self.predictor_attention = DeepLearningRentPredictor(model_type='attention')
                self.current_predictor = self.predictor_attention
        except Exception as e:
            print(f"モデル読み込みエラー: {e}")

    def _on_model_change(self, change):
        """モデル選択変更時のハンドラ"""
        self.load_predictors(change['new'])
    
    def _update_room_label(self, change):
        self.room_size_label.value = f'{change["new"]} m²'
    
    def _update_station_label(self, change):
        self.station_label.value = f'{change["new"]} 分'
    
    def _update_age_label(self, change):
        self.age_label.value = f'{change["new"]} 年'
    
    def on_predict_click(self, b):
        """予測ボタンクリックハンドラ"""
        with self.output:
            clear_output()
            
            current_tab = self.mode_tabs.selected_index
            
            if current_tab == 0:  # 単一予測
                self.single_prediction()
            elif current_tab == 1:  # 区別比較
                self.comparison_prediction()
            elif current_tab == 2:  # AI分析
                self.ai_analysis()
    
    def single_prediction(self):
        """単一物件予測"""
        ward = self.ward_dropdown.value.split(' (')[0]

        if self.current_predictor is None:
            print("⚠️ モデルが読み込まれていません")
            return

        prediction = self.current_predictor.predict(
            ward,
            self.room_size_slider.value,
            self.station_distance_slider.value,
            self.building_age_slider.value,
            self.structure_dropdown.value,
            self.property_type_dropdown.value
        )
        
        # 구 정보
        ward_info = next((info for info in self.ward_list if info[0] == ward), None)
        if not ward_info:
            return
        
        ward_level = ward_info[1]
        bg_color = ward_info[2]
        base_price = ward_info[3]
        
        # HTML 생성
        html = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
            <div style="background: linear-gradient(135deg, {bg_color}15, white); 
                        border: 3px solid {bg_color}; border-radius: 20px; 
                        padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
                
                <h2 style="color: {bg_color}; margin: 0 0 25px 0; font-size: 28px; 
                           display: flex; align-items: center;">
                    <span style="font-size: 35px; margin-right: 10px;">🤖</span>
                    ディープラーニング予測結果
                </h2>
                
                <div style="background: white; padding: 20px; border-radius: 12px; 
                            margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
                    <h3 style="color: #555; margin: 0 0 15px 0; font-size: 16px;">📋 入力条件</h3>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; font-size: 14px;">
                        <div>📍 <strong>区：</strong> {ward} 
                            <span style="background: {bg_color}; color: white; padding: 3px 10px; 
                                       border-radius: 15px; font-size: 11px; margin-left: 8px;">{ward_level}</span>
                        </div>
                        <div>📏 <strong>広さ：</strong> {self.room_size_slider.value} m²</div>
                        <div>🚉 <strong>駅まで：</strong> 徒歩{self.station_distance_slider.value}分</div>
                        <div>🏗️ <strong>築年数：</strong> {self.building_age_slider.value}年</div>
                        <div>🏢 <strong>構造：</strong> {self.structure_dropdown.value}</div>
                        <div>🏠 <strong>タイプ：</strong> {self.property_type_dropdown.value}</div>
                    </div>
                </div>
                
                <div style="background: linear-gradient(135deg, {bg_color}, {bg_color}dd); 
                            color: white; padding: 30px; border-radius: 15px; 
                            text-align: center; position: relative; overflow: hidden;">
                    <div style="position: absolute; top: -20px; right: -20px; 
                                font-size: 100px; opacity: 0.2;">💰</div>
                    <div style="font-size: 14px; opacity: 0.9; margin-bottom: 10px; 
                                text-transform: uppercase; letter-spacing: 2px;">
                        AI Predicted Monthly Rent
                    </div>
                    <div style="font-size: 56px; font-weight: 300; margin: 15px 0;">
                        <span style="font-size: 32px;">¥</span>{prediction:,.0f}
                    </div>
                    <div style="font-size: 12px; opacity: 0.7;">
                        区の基準価格: ¥{base_price:,} | 
                        差額: ¥{prediction - base_price:+,.0f}
                    </div>
                </div>
                
                <div style="margin-top: 20px; padding: 15px; background: #f8f9fa;
                            border-radius: 8px; display: flex; align-items: center;">
                    <div style="font-size: 30px; margin-right: 15px;">🧠</div>
                    <div style="font-size: 12px; color: #666;">
                        <strong>モデル情報：</strong>
                        {self.current_predictor.model_type.upper()} Model |
                        PyTorch Neural Network |
                        {'Attention機構 | Ward Embedding (32次元) | 3層 (512→256→128)' if self.current_predictor.model_type == 'attention' else 'Ward Embedding (16次元) | 3層 (256→128→64)'} |
                        Batch Normalization | Dropout
                    </div>
                </div>
            </div>
        </div>
        """
        
        display(HTML(html))
    
    def comparison_prediction(self):
        """複数区比較予測"""
        if self.current_predictor is None:
            print("⚠️ モデルが読み込まれていません")
            return

        comparison_wards = [
            '港区', '千代田区', '中央区', '渋谷区',
            '新宿区', '世田谷区', '中野区', '練馬区',
            '板橋区', '足立区'
        ]

        predictions = []
        for ward in comparison_wards:
            pred = self.current_predictor.predict(
                ward,
                self.room_size_slider.value,
                self.station_distance_slider.value,
                self.building_age_slider.value,
                self.structure_dropdown.value,
                self.property_type_dropdown.value
            )
            
            ward_info = next((info for info in self.ward_list if info[0] == ward), None)
            if ward_info:
                predictions.append({
                    'ward': ward,
                    'prediction': pred,
                    'level': ward_info[1],
                    'color': ward_info[2],
                    'base': ward_info[3]
                })
        
        predictions.sort(key=lambda x: x['prediction'], reverse=True)
        max_pred = predictions[0]['prediction'] if predictions else 1
        
        # HTML 생성
        html = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
            <div style="background: white; border: 2px solid #5e72e4; border-radius: 20px; 
                        padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
                
                <h2 style="color: #5e72e4; margin: 0 0 25px 0; display: flex; align-items: center;">
                    <span style="font-size: 30px; margin-right: 10px;">📊</span>
                    ディープラーニング区別比較
                </h2>
                
                <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; 
                            margin-bottom: 25px; font-size: 14px;">
                    <strong>条件：</strong>
                    {self.room_size_slider.value}m² | 駅{self.station_distance_slider.value}分 | 
                    築{self.building_age_slider.value}年 | {self.structure_dropdown.value} | 
                    {self.property_type_dropdown.value}
                </div>
                
                <div style="margin-bottom: 25px;">
        """
        
        for i, pred_info in enumerate(predictions):
            bar_width = (pred_info['prediction'] / max_pred) * 100 if max_pred > 0 else 0
            diff_from_base = pred_info['prediction'] - pred_info['base']
            
            html += f"""
                <div style="display: flex; align-items: center; margin: 15px 0;">
                    <div style="width: 30px; text-align: center; font-weight: bold; 
                                color: #999; font-size: 14px;">
                        {i+1}
                    </div>
                    <div style="width: 100px; font-weight: 600; font-size: 14px; 
                                margin-left: 10px;">
                        {pred_info['ward']}
                    </div>
                    <div style="flex: 1; margin: 0 20px;">
                        <div style="background: #e9ecef; border-radius: 25px; 
                                    overflow: hidden; height: 32px; position: relative;">
                            <div style="width: {bar_width}%; 
                                        background: linear-gradient(90deg, {pred_info['color']}, {pred_info['color']}bb); 
                                        height: 100%; display: flex; align-items: center; 
                                        justify-content: flex-end; padding-right: 15px;
                                        transition: width 0.5s ease;">
                                <span style="color: white; font-weight: 600; font-size: 13px;">
                                    ¥{pred_info['prediction']:,.0f}
                                </span>
                            </div>
                        </div>
                    </div>
                    <div style="text-align: right; min-width: 80px; font-size: 11px; color: #666;">
                        {diff_from_base:+,.0f}
                    </div>
                    <span style="background: {pred_info['color']}; color: white; 
                                padding: 5px 12px; border-radius: 20px; font-size: 11px; 
                                min-width: 60px; text-align: center; margin-left: 10px;">
                        {pred_info['level']}
                    </span>
                </div>
            """
        
        # 統計情報
        if len(predictions) > 1:
            diff = predictions[0]['prediction'] - predictions[-1]['prediction']
            ratio = predictions[0]['prediction'] / predictions[-1]['prediction'] if predictions[-1]['prediction'] > 0 else 1
            avg_pred = sum(p['prediction'] for p in predictions) / len(predictions)
            
            html += f"""
                </div>
                
                <div style="margin-top: 30px; padding: 20px; 
                            background: linear-gradient(135deg, #667eea, #764ba2); 
                            border-radius: 12px; color: white;">
                    <h4 style="margin: 0 0 15px 0; font-size: 16px;">📈 AI分析結果</h4>
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px;">
                        <div style="text-align: center;">
                            <div style="font-size: 12px; opacity: 0.8;">最高額</div>
                            <div style="font-size: 20px; font-weight: bold;">
                                ¥{predictions[0]['prediction']:,.0f}
                            </div>
                            <div style="font-size: 10px; opacity: 0.7;">¥{diff:,.0f}</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 12px; opacity: 0.8;">平均値</div>
                            <div style="font-size: 20px; font-weight: bold;">
                                ¥{avg_pred:,.0f}
                            </div>
                            <div style="font-size: 10px; opacity: 0.7;">全{len(predictions)}区</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
        
        display(HTML(html))
    
    def ai_analysis(self):
        """AI모델 내부 분석"""
        # 구 임베딩 가져오기
        if self.current_predictor is None:
            print("⚠️ モデルが読み込まれていません")
            return
        embeddings, ward_names = self.current_predictor.get_ward_embeddings()
        
        if embeddings is None:
            display(HTML("<p>モデルが読み込まれていません</p>"))
            return
        
        # HTML 시작
        html = """
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
            <div style="background: white; border: 2px solid #6c5ce7; border-radius: 20px; 
                        padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
                
                <h2 style="color: #6c5ce7; margin: 0 0 25px 0; display: flex; align-items: center;">
                    <span style="font-size: 30px; margin-right: 10px;">🧠</span>
                    AIモデル内部分析
                </h2>
        """
        
        # 1. 구 임베딩 중요도
        importance_scores = np.linalg.norm(embeddings, axis=1)
        ward_importance = list(zip(ward_names, importance_scores))
        ward_importance.sort(key=lambda x: x[1], reverse=True)
        
        html += """
                <div style="background: #f8f9fa; padding: 20px; border-radius: 12px; margin-bottom: 25px;">
                    <h3 style="color: #495057; margin: 0 0 15px 0; font-size: 18px;">
                        🏆 区の埋め込み重要度 (Top 10)
                    </h3>
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        """
        
        max_importance = ward_importance[0][1] if ward_importance else 1
        
        for i, (ward, score) in enumerate(ward_importance[:10]):
            ward_info = next((info for info in self.ward_list if info[0] == ward), None)
            color = ward_info[2] if ward_info else '#666'
            bar_width = (score / max_importance) * 100
            
            html += f"""
                        <div style="display: flex; align-items: center;">
                            <div style="width: 80px; font-weight: 600; font-size: 13px;">
                                {i+1}. {ward}
                            </div>
                            <div style="flex: 1; margin: 0 10px;">
                                <div style="background: #e9ecef; border-radius: 10px; 
                                            overflow: hidden; height: 20px;">
                                    <div style="width: {bar_width}%; background: {color}; 
                                                height: 100%; display: flex; align-items: center; 
                                                justify-content: flex-end; padding-right: 8px;">
                                        <span style="color: white; font-size: 11px; font-weight: bold;">
                                            {score:.2f}
                                        </span>
                                    </div>
                                </div>
                            </div>
                        </div>
            """
        
        html += """
                    </div>
                </div>
        """
        
        # 2. 임베딩 차원 분석
        html += """
                <div style="background: #f8f9fa; padding: 20px; border-radius: 12px; margin-bottom: 25px;">
                    <h3 style="color: #495057; margin: 0 0 15px 0; font-size: 18px;">
                        📊 埋め込みベクトル分析
                    </h3>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; 
                                text-align: center;">
                        <div style="background: white; padding: 15px; border-radius: 8px; 
                                    border: 1px solid #dee2e6;">
                            <div style="font-size: 24px; color: #6c5ce7; font-weight: bold;">
                                32
                            </div>
                            <div style="font-size: 12px; color: #666; margin-top: 5px;">
                                埋め込み次元
                            </div>
                        </div>
                        <div style="background: white; padding: 15px; border-radius: 8px; 
                                    border: 1px solid #dee2e6;">
                            <div style="font-size: 24px; color: #00b894; font-weight: bold;">
                                {len(ward_names)}
                            </div>
                            <div style="font-size: 12px; color: #666; margin-top: 5px;">
                                学習した区の数
                            </div>
                        </div>
                        <div style="background: white; padding: 15px; border-radius: 8px; 
                                    border: 1px solid #dee2e6;">
                            <div style="font-size: 24px; color: #fdcb6e; font-weight: bold;">
                                {embeddings.shape[0] * embeddings.shape[1]}
                            </div>
                            <div style="font-size: 12px; color: #666; margin-top: 5px;">
                                総パラメータ数
                            </div>
                        </div>
                    </div>
                </div>
        """
        
        # 3. 類似度分析
        selected_ward = self.ward_dropdown.value.split(' (')[0]
        if selected_ward in ward_names:
            ward_idx = list(ward_names).index(selected_ward)
            selected_embedding = embeddings[ward_idx]
            
            # コサイン類似度計算
            similarities = []
            for i, (name, emb) in enumerate(zip(ward_names, embeddings)):
                if name != selected_ward:
                    cos_sim = np.dot(selected_embedding, emb) / (
                        np.linalg.norm(selected_embedding) * np.linalg.norm(emb)
                    )
                    similarities.append((name, cos_sim))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            html += f"""
                <div style="background: #f8f9fa; padding: 20px; border-radius: 12px; margin-bottom: 25px;">
                    <h3 style="color: #495057; margin: 0 0 15px 0; font-size: 18px;">
                        🔍 {selected_ward}と類似した区 (コサイン類似度)
                    </h3>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
            """
            
            for name, sim in similarities[:6]:
                ward_info = next((info for info in self.ward_list if info[0] == name), None)
                color = ward_info[2] if ward_info else '#666'
                level = ward_info[1] if ward_info else ''
                
                html += f"""
                        <div style="background: white; padding: 12px; border-radius: 8px; 
                                    border: 2px solid {color}33; text-align: center;">
                            <div style="font-weight: bold; font-size: 14px; color: #333;">
                                {name}
                            </div>
                            <div style="font-size: 18px; color: {color}; font-weight: bold; 
                                        margin: 5px 0;">
                                {sim:.3f}
                            </div>
                            <div style="font-size: 10px; color: #666;">
                                {level}
                            </div>
                        </div>
                """
            
            html += """
                    </div>
                </div>
            """
        
        # 4. モデル構造情報
        html += """
                <div style="background: linear-gradient(135deg, #74b9ff, #0984e3); 
                            padding: 20px; border-radius: 12px; color: white;">
                    <h3 style="margin: 0 0 15px 0; font-size: 18px;">
                        🏗️ ニューラルネットワーク構造
                    </h3>
                    <div style="display: flex; align-items: center; justify-content: space-between; 
                                flex-wrap: wrap;">
                        <div style="background: rgba(255,255,255,0.2); padding: 10px 15px; 
                                    border-radius: 8px; margin: 5px;">
                            <strong>Input</strong><br>
                            7 features
                        </div>
                        <div style="font-size: 20px;">→</div>
                        <div style="background: rgba(255,255,255,0.2); padding: 10px 15px; 
                                    border-radius: 8px; margin: 5px;">
                            <strong>Embedding</strong><br>
                            32 dims
                        </div>
                        <div style="font-size: 20px;">→</div>
                        <div style="background: rgba(255,255,255,0.2); padding: 10px 15px; 
                                    border-radius: 8px; margin: 5px;">
                            <strong>Attention</strong><br>
                            Dynamic
                        </div>
                        <div style="font-size: 20px;">→</div>
                        <div style="background: rgba(255,255,255,0.2); padding: 10px 15px; 
                                    border-radius: 8px; margin: 5px;">
                            <strong>Hidden</strong><br>
                            512→256→128
                        </div>
                        <div style="font-size: 20px;">→</div>
                        <div style="background: rgba(255,255,255,0.2); padding: 10px 15px; 
                                    border-radius: 8px; margin: 5px;">
                            <strong>Output</strong><br>
                            1 (rent)
                        </div>
                    </div>
                    <div style="margin-top: 15px; font-size: 12px; opacity: 0.9;">
                        総パラメータ数: ~300,000 | 
                        Dropout: 0.25 | 
                        Activation: LeakyReLU | 
                        Optimizer: AdamW
                    </div>
                </div>
            </div>
        </div>
        """
        
        display(HTML(html))
    
    def display(self):
        """UI 전체 표시"""
        # 타이틀
        title_html = """
        <div style="background: linear-gradient(135deg, #667eea, #764ba2); 
                    padding: 30px; border-radius: 15px; margin-bottom: 20px; color: white;">
            <h1 style="margin: 0; font-size: 32px; display: flex; align-items: center;">
                <span style="font-size: 40px; margin-right: 15px;">🤖</span>
                ディープラーニング東京家賃予測システム
            </h1>
            <p style="margin: 10px 0 0 0; opacity: 0.9; font-size: 14px;">
                PyTorch Neural Network with Attention Mechanism | 
                3-Layer Deep Network | Ward Embeddings | 
                Device: {device}
            </p>
        </div>
        """
        
        display(HTML(title_html))
        
        # 입력 컨트롤
        input_section = widgets.VBox([
            widgets.HTML("<h3 style='color: #5e72e4; margin: 15px 0;'>📝 物件情報入力</h3>"),
            self.model_selector,
            widgets.HTML("<div style='margin: 10px 0; border-top: 1px solid #dee2e6;'></div>"),
            self.ward_dropdown,
            widgets.HBox([self.room_size_slider, self.room_size_label]),
            widgets.HBox([self.station_distance_slider, self.station_label]),
            widgets.HBox([self.building_age_slider, self.age_label]),
            widgets.HBox([self.structure_dropdown, self.property_type_dropdown]),
            widgets.HTML("<div style='margin: 20px 0; border-top: 1px solid #dee2e6;'></div>"),
        ])
        
        # 전체 레이아웃
        main_layout = widgets.VBox([
            input_section,
            self.mode_tabs,
            widgets.HTML("<div style='margin: 20px 0;'></div>"),
            widgets.HBox([self.predict_button]),
            self.output
        ])
        
        display(main_layout)
        
        # 초기 예측 실행
        self.on_predict_click(None)


# In[5]:


try:
    app = InteractiveRentPredictorDL()
    app.display()
except Exception as e:
    print(f"エラーが発生しました: {e}")
    print("\n以下の手順を確認してください：")
    print("1. tokyo_rent_data_v2.csv が存在するか")
    print("2. tokyo_rent_deep_learning.py を実行して学習済みか")
    print("3. rent_prediction_model_complete.pth が生成されているか")


# In[6]:


# ===========================
# 2. 시각화 클래스
# ===========================
class NeuralNetworkVisualizer:
    def __init__(self):
        """모델 로드 및 초기화"""
        self.model = None
        self.preprocessor = None
        self.df = None
        self.load_model_and_data()
    
    def load_model_and_data(self):
        """모델과 데이터 로드"""
        try:
            # 모델 로드
            checkpoint = torch.load('rent_prediction_model_complete.pth', 
                                  map_location=device, 
                                  weights_only=False)
            
            self.preprocessor = checkpoint['preprocessor']
            config = checkpoint['model_config']
            
            self.model = RentPredictionNetWithAttention(
                config['num_wards'],
                config['num_structures'],
                config['num_types']
            ).to(device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # 데이터 로드
            self.df = pd.read_csv('rent/tokyo_rent_data_v2.csv')
            
            print("✅ モデルとデータの読み込み成功!")
            
        except Exception as e:
            print(f"⚠️ エラー: {e}")
            raise
    
    def visualize_architecture(self):
        """네트워크 아키텍처 시각화"""
        fig = plt.figure(figsize=(16, 10))
        
        # 1. 네트워크 구조 다이어그램
        ax1 = plt.subplot(2, 2, 1)
        
        # 레이어별 노드 수
        layer_sizes = [7, 32, 512, 256, 128, 64, 1]
        layer_names = ['Input\n(7)', 'Embed\n(32)', 'Hidden1\n(512)', 
                      'Hidden2\n(256)', 'Hidden3\n(128)', 'Hidden4\n(64)', 'Output\n(1)']
        
        # 네트워크 그래프 생성
        G = nx.DiGraph()
        pos = {}
        node_colors = []
        
        for i, (size, name) in enumerate(zip(layer_sizes, layer_names)):
            for j in range(min(size, 10)):  # 최대 10개 노드만 표시
                node_id = f"L{i}N{j}"
                G.add_node(node_id)
                pos[node_id] = (i * 2, j - min(size, 10)/2)
                
                if i == 0:
                    node_colors.append('#3498db')
                elif i == len(layer_sizes) - 1:
                    node_colors.append('#e74c3c')
                else:
                    node_colors.append('#95a5a6')
                
                # 이전 레이어와 연결
                if i > 0:
                    prev_size = min(layer_sizes[i-1], 10)
                    for k in range(prev_size):
                        G.add_edge(f"L{i-1}N{k}", node_id)
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=100, alpha=0.8, ax=ax1)
        nx.draw_networkx_edges(G, pos, alpha=0.1, ax=ax1)
        
        # 레이어 이름 추가
        for i, name in enumerate(layer_names):
            ax1.text(i * 2, -6, name, ha='center', fontsize=10, fontweight='bold')
        
        ax1.set_title('Neural Network Architecture', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # 2. 파라미터 수 분포
        ax2 = plt.subplot(2, 2, 2)
        
        # 각 레이어의 파라미터 수 계산
        param_counts = []
        layer_names_short = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_counts.append(param.numel())
                layer_names_short.append(name.split('.')[0])
        
        # 상위 10개 레이어만 표시
        top_indices = np.argsort(param_counts)[-10:]
        top_params = [param_counts[i] for i in top_indices]
        top_names = [layer_names_short[i] for i in top_indices]
        
        bars = ax2.barh(range(len(top_params)), top_params, color='#3498db')
        ax2.set_yticks(range(len(top_params)))
        ax2.set_yticklabels(top_names, fontsize=9)
        ax2.set_xlabel('Number of Parameters')
        ax2.set_title('Top 10 Layers by Parameter Count', fontsize=12, fontweight='bold')
        
        # 값 표시
        for i, (bar, val) in enumerate(zip(bars, top_params)):
            ax2.text(val, i, f' {val:,}', va='center', fontsize=8)
        
        # 3. 임베딩 차원 시각화
        ax3 = plt.subplot(2, 2, 3)
        
        ward_embeddings = self.model.ward_embedding.weight.detach().cpu().numpy()
        
        # PCA로 2D 투영
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(ward_embeddings)
        
        # 구별 평균 가격
        ward_names = self.preprocessor.label_encoders['区'].classes_
        ward_prices = []
        for ward in ward_names:
            avg_price = self.df[self.df['区'] == ward]['家賃_円'].mean()
            ward_prices.append(avg_price)
        
        scatter = ax3.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=ward_prices, cmap='RdYlBu_r', 
                            s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # 주요 구 라벨 추가
        important_wards = ['港区', '千代田区', '渋谷区', '新宿区', '中野区', '足立区']
        for i, ward in enumerate(ward_names):
            if ward in important_wards:
                ax3.annotate(ward, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                           fontsize=9, ha='center', 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        plt.colorbar(scatter, ax=ax3, label='Average Rent (¥)')
        ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax3.set_title('Ward Embeddings (PCA)', fontsize=12, fontweight='bold')
        
        # 4. 활성화 함수 분포
        ax4 = plt.subplot(2, 2, 4)
        
        # 임의의 입력으로 활성화 값 수집
        sample_input = torch.randn(100, 7).to(device)
        activations = []
        
        def hook_fn(module, input, output):
            if isinstance(module, nn.LeakyReLU):
                activations.append(output.detach().cpu().numpy().flatten())
        
        # Hook 등록
        hooks = []
        for module in self.model.modules():
            if isinstance(module, nn.LeakyReLU):
                hooks.append(module.register_forward_hook(hook_fn))
        
        # Forward pass (더미 데이터)
        with torch.no_grad():
            ward_idx = torch.randint(0, self.model.num_wards, (100,)).to(device)
            structure_idx = torch.randint(0, 4, (100,)).to(device)
            type_idx = torch.randint(0, 4, (100,)).to(device)
            numeric_features = torch.randn(100, 3).to(device)
            ward_avg_price = torch.randn(100).to(device)
            
            _ = self.model(ward_idx, structure_idx, type_idx, numeric_features, ward_avg_price)
        
        # Hook 제거
        for hook in hooks:
            hook.remove()
        
        # 활성화 값 분포 플롯
        if activations:
            all_activations = np.concatenate(activations)
            ax4.hist(all_activations, bins=50, alpha=0.7, color='#2ecc71', edgecolor='black')
            ax4.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            ax4.set_xlabel('Activation Value')
            ax4.set_ylabel('Frequency')
            ax4.set_title('LeakyReLU Activation Distribution', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('network_architecture_viz.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # 모델 요약 정보
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("\n" + "="*60)
        print("モデル構造サマリー")
        print("="*60)
        print(f"総パラメータ数: {total_params:,}")
        print(f"埋め込み次元: {self.model.embedding_dim}")
        print(f"隠れ層: {self.model.hidden_dims}")
        print(f"区の数: {self.model.num_wards}")
    
    def visualize_embeddings_interactive(self):
        """인터랙티브 임베딩 시각화 (Plotly)"""
        # 임베딩 가져오기
        ward_embeddings = self.model.ward_embedding.weight.detach().cpu().numpy()
        ward_names = self.preprocessor.label_encoders['区'].classes_
        
        # 구별 통계
        ward_stats = []
        for ward in ward_names:
            ward_data = self.df[self.df['区'] == ward]
            ward_stats.append({
                'ward': ward,
                'avg_rent': ward_data['家賃_円'].mean(),
                'count': len(ward_data),
                'std_rent': ward_data['家賃_円'].std()
            })
        
        stats_df = pd.DataFrame(ward_stats)
        
        # t-SNE로 2D 투영
        tsne = TSNE(n_components=2, random_state=42, perplexity=15)
        embeddings_tsne = tsne.fit_transform(ward_embeddings)
        
        # PCA로 2D 투영
        pca = PCA(n_components=2)
        embeddings_pca = pca.fit_transform(ward_embeddings)
        
        # 3D PCA
        pca_3d = PCA(n_components=3)
        embeddings_3d = pca_3d.fit_transform(ward_embeddings)
        
        # Plotly 서브플롯 생성
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('t-SNE Projection', 'PCA Projection', 
                          '3D PCA Projection', 'Embedding Heatmap'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter3d'}, {'type': 'heatmap'}]]
        )
        
        # 1. t-SNE 플롯
        fig.add_trace(
            go.Scatter(
                x=embeddings_tsne[:, 0],
                y=embeddings_tsne[:, 1],
                mode='markers+text',
                marker=dict(
                    size=10,
                    color=stats_df['avg_rent'],
                    colorscale='RdYlBu_r',
                    showscale=True,
                    colorbar=dict(title="Avg Rent", x=0.45, y=0.75)
                ),
                text=ward_names,
                textposition="top center",
                textfont=dict(size=8),
                hovertemplate='<b>%{text}</b><br>' +
                            'Avg Rent: ¥%{marker.color:,.0f}<br>' +
                            '<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. PCA 플롯
        fig.add_trace(
            go.Scatter(
                x=embeddings_pca[:, 0],
                y=embeddings_pca[:, 1],
                mode='markers+text',
                marker=dict(
                    size=10,
                    color=stats_df['avg_rent'],
                    colorscale='RdYlBu_r',
                    showscale=False
                ),
                text=ward_names,
                textposition="top center",
                textfont=dict(size=8),
                hovertemplate='<b>%{text}</b><br>' +
                            'PC1: %{x:.2f}<br>' +
                            'PC2: %{y:.2f}<br>' +
                            '<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. 3D PCA 플롯
        fig.add_trace(
            go.Scatter3d(
                x=embeddings_3d[:, 0],
                y=embeddings_3d[:, 1],
                z=embeddings_3d[:, 2],
                mode='markers+text',
                marker=dict(
                    size=8,
                    color=stats_df['avg_rent'],
                    colorscale='RdYlBu_r',
                    showscale=True,
                    colorbar=dict(title="Avg Rent", x=0.45, y=0.25)
                ),
                text=ward_names,
                textfont=dict(size=8),
                hovertemplate='<b>%{text}</b><br>' +
                            'PC1: %{x:.2f}<br>' +
                            'PC2: %{y:.2f}<br>' +
                            'PC3: %{z:.2f}<br>' +
                            '<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 4. 임베딩 히트맵
        fig.add_trace(
            go.Heatmap(
                z=ward_embeddings[:10, :],  # 상위 10개 구만
                x=[f'Dim {i+1}' for i in range(ward_embeddings.shape[1])],
                y=ward_names[:10],
                colorscale='RdBu',
                hovertemplate='Ward: %{y}<br>' +
                            'Dimension: %{x}<br>' +
                            'Value: %{z:.3f}<br>' +
                            '<extra></extra>'
            ),
            row=2, col=2
        )
        
        # 레이아웃 업데이트
        fig.update_layout(
            title_text="Ward Embeddings Multi-View Visualization",
            height=800,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="t-SNE 1", row=1, col=1)
        fig.update_yaxes(title_text="t-SNE 2", row=1, col=1)
        fig.update_xaxes(title_text=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", row=1, col=2)
        fig.update_yaxes(title_text=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", row=1, col=2)
        
        fig.show()
        
        # 임베딩 유사도 매트릭스
        self.plot_similarity_matrix(ward_embeddings, ward_names)
    
    def plot_similarity_matrix(self, embeddings, names):
        """코사인 유사도 매트릭스 시각화"""
        # 코사인 유사도 계산
        norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        similarity_matrix = np.dot(norm_embeddings, norm_embeddings.T)
        
        # Plotly 히트맵
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=names,
            y=names,
            colorscale='RdBu',
            zmid=0,
            text=similarity_matrix.round(2),
            texttemplate='%{text}',
            textfont={"size": 8},
            colorbar=dict(title="Cosine Similarity"),
            hovertemplate='%{y} - %{x}<br>Similarity: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Ward Embedding Cosine Similarity Matrix",
            xaxis_title="Ward",
            yaxis_title="Ward",
            width=900,
            height=800
        )
        
        fig.show()
    
    def visualize_attention_weights(self):
        """Attention 가중치 시각화"""
        ward_names = self.preprocessor.label_encoders['区'].classes_
        attention_weights_all = []
        
        # 각 구에 대한 attention 가중치 수집
        with torch.no_grad():
            for i in range(len(ward_names)):
                ward_idx = torch.LongTensor([i]).to(device)
                ward_emb = self.model.ward_embedding(ward_idx)
                attention_weight = self.model.attention(ward_emb)
                attention_weights_all.append(attention_weight.cpu().numpy()[0, 0])
        
        # 시각화
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Attention 가중치 바 차트
        sorted_indices = np.argsort(attention_weights_all)[::-1]
        sorted_weights = [attention_weights_all[i] for i in sorted_indices]
        sorted_names = [ward_names[i] for i in sorted_indices]
        
        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(sorted_weights)))
        bars = axes[0].barh(range(len(sorted_weights)), sorted_weights, color=colors)
        axes[0].set_yticks(range(len(sorted_weights)))
        axes[0].set_yticklabels(sorted_names, fontsize=8)
        axes[0].set_xlabel('Attention Weight')
        axes[0].set_title('Attention Weights by Ward', fontsize=14, fontweight='bold')
        
        # 값 표시
        for i, (bar, val) in enumerate(zip(bars, sorted_weights)):
            axes[0].text(val, i, f' {val:.4f}', va='center', fontsize=8)
        
        # 2. Attention과 평균 가격의 관계
        avg_prices = []
        for ward in ward_names:
            avg_price = self.df[self.df['区'] == ward]['家賃_円'].mean()
            avg_prices.append(avg_price)
        
        axes[1].scatter(avg_prices, attention_weights_all, alpha=0.7, s=100)
        axes[1].set_xlabel('Average Rent (¥)')
        axes[1].set_ylabel('Attention Weight')
        axes[1].set_title('Attention Weight vs Average Rent', fontsize=14, fontweight='bold')
        
        # 주요 구 라벨
        for i, ward in enumerate(ward_names):
            if ward in ['港区', '千代田区', '足立区', '葛飾区']:
                axes[1].annotate(ward, (avg_prices[i], attention_weights_all[i]),
                               fontsize=9, ha='center',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
        
        # 상관관계 계산
        correlation = np.corrcoef(avg_prices, attention_weights_all)[0, 1]
        axes[1].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                    transform=axes[1].transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('attention_weights_viz.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def visualize_training_analysis(self):
        """학습 과정 분석 (가상 데이터)"""
        # 실제 학습 로그가 없으므로 시뮬레이션
        epochs = np.arange(1, 51)
        train_loss = 0.5 * np.exp(-epochs/10) + 0.1 + np.random.normal(0, 0.01, 50)
        val_loss = 0.5 * np.exp(-epochs/12) + 0.12 + np.random.normal(0, 0.015, 50)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Progress', 'Learning Rate Schedule', 
                          'Gradient Flow', 'Parameter Distribution')
        )
        
        # 1. 학습 진행
        fig.add_trace(
            go.Scatter(x=epochs, y=train_loss, name='Train Loss', 
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=val_loss, name='Val Loss',
                      line=dict(color='red', width=2)),
            row=1, col=1
        )
        
        # 2. Learning Rate Schedule
        lr = 0.001 * np.exp(-epochs/20)
        fig.add_trace(
            go.Scatter(x=epochs, y=lr, name='Learning Rate',
                      line=dict(color='green', width=2)),
            row=1, col=2
        )
        
        # 3. Gradient Flow (가상)
        layer_names = ['Embedding', 'Attention', 'Hidden1', 'Hidden2', 'Hidden3', 'Output']
        gradient_means = np.random.exponential(0.01, len(layer_names))
        fig.add_trace(
            go.Bar(x=layer_names, y=gradient_means, name='Gradient Magnitude',
                  marker_color='purple'),
            row=2, col=1
        )
        
        # 4. 파라미터 분포
        all_params = []
        for param in self.model.parameters():
            all_params.extend(param.detach().cpu().numpy().flatten())
        
        fig.add_trace(
            go.Histogram(x=all_params[:1000], name='Parameter Values',  # 샘플링
                        marker_color='orange', nbinsx=50),
            row=2, col=2
        )
        
        fig.update_layout(height=700, showlegend=True,
                         title_text="Training Analysis Dashboard")
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Learning Rate", row=1, col=2)
        fig.update_xaxes(title_text="Layer", row=2, col=1)
        fig.update_yaxes(title_text="Gradient Magnitude", row=2, col=1)
        fig.update_xaxes(title_text="Parameter Value", row=2, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
        
        fig.show()

# ===========================
# 3. 메인 실행 함수
# ===========================
def main():
    print("="*60)
    print("🧠 ニューラルネットワーク総合可視化")
    print("="*60)
    
    visualizer = NeuralNetworkVisualizer()
    
    print("\n1. ネットワークアーキテクチャ可視化...")
    visualizer.visualize_architecture()
    
    print("\n2. インタラクティブ埋め込み可視化...")
    visualizer.visualize_embeddings_interactive()
    
    print("\n3. Attention重み可視化...")
    visualizer.visualize_attention_weights()
    
    print("\n4. 学習プロセス分析...")
    visualizer.visualize_training_analysis()
    
    print("\n✅ 可視化完了!")
    print("生成されたファイル:")
    print("  - network_architecture_viz.png")
    print("  - attention_weights_viz.png")
    print("  - インタラクティブプロット (ブラウザで表示)")


# In[7]:


main()

