"""
Tokyo Rent Prediction - Prediction utilities and model loading
"""

import pandas as pd
import numpy as np
import torch


class DeepLearningRentPredictor:
    """Deep Learning model loading and prediction execution class"""

    def __init__(self, model_type='attention', device=None):
        """Initialize model and preprocessor

        Parameters:
        -----------
        model_type : str
            'basic' or 'attention'
        device : torch.device
            Device to use for inference
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        """Load saved model"""
        try:
            from rent_dl_models import RentPredictionNet, RentPredictionNetWithAttention

            if model_type == 'basic':
                model_path = 'rent_prediction_model_basic.pth'
                model_name = "基本モデル"
            else:
                model_path = 'rent_prediction_model_attention.pth'
                model_name = "Attentionモデル"

            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            self.preprocessor = checkpoint['preprocessor']
            config = checkpoint['model_config']

            if model_type == 'basic':
                self.model = RentPredictionNet(
                    config['num_wards'],
                    config['num_structures'],
                    config['num_types']
                ).to(self.device)
            else:
                self.model = RentPredictionNetWithAttention(
                    config['num_wards'],
                    config['num_structures'],
                    config['num_types']
                ).to(self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            df = pd.read_csv('tokyo_rent_data_v2.csv')
            self.ward_avg_prices = df.groupby('区')['家賃_円'].mean().to_dict()
            self.global_mean_price = df['家賃_円'].mean()
            self.global_std_price = df['家賃_円'].std()

            self.model_loaded = True
            print(f"✅ ディープラーニングモデル読み込み成功!")
            print(f"   - モデルタイプ: {model_name}")
            print(f"   - デバイス: {self.device}")

        except FileNotFoundError:
            print(f"⚠️ モデルファイルが見つかりません: {model_path}")
            raise

    def predict(self, ward, room_size, station_distance, building_age, structure, property_type):
        """Single property prediction"""
        if not self.model_loaded:
            return 0

        try:
            ward_encoded = self.preprocessor.label_encoders['区'].transform([ward])[0]
            structure_encoded = self.preprocessor.label_encoders['建物構造'].transform([structure])[0]
            type_encoded = self.preprocessor.label_encoders['建物タイプ'].transform([property_type])[0]

            numeric_features = np.array([[room_size, station_distance, building_age]])
            numeric_features_scaled = self.preprocessor.scaler.transform(numeric_features)

            ward_avg = self.ward_avg_prices.get(ward, self.global_mean_price)
            ward_avg_normalized = (ward_avg - self.global_mean_price) / self.global_std_price

            with torch.no_grad():
                ward_idx = torch.LongTensor([ward_encoded]).to(self.device)
                structure_idx = torch.LongTensor([structure_encoded]).to(self.device)
                type_idx = torch.LongTensor([type_encoded]).to(self.device)
                numeric_feat = torch.FloatTensor(numeric_features_scaled).to(self.device)
                ward_avg_price = torch.FloatTensor([ward_avg_normalized]).to(self.device)

                result = self.model(ward_idx, structure_idx, type_idx, numeric_feat, ward_avg_price)
                if isinstance(result, tuple):
                    output = result[0]
                else:
                    output = result
                prediction = output.item() * 10000

            return max(prediction, 20000)

        except Exception as e:
            print(f"予測エラー: {e}")
            return 0

    def batch_predict(self, conditions_list):
        """Batch prediction"""
        if not self.model_loaded:
            return []

        predictions = []
        for conditions in conditions_list:
            pred = self.predict(**conditions)
            predictions.append(pred)

        return predictions

    def get_ward_embeddings(self):
        """Return ward embedding vectors"""
        if not self.model_loaded:
            return None, None

        with torch.no_grad():
            embeddings = self.model.ward_embedding.weight.cpu().numpy()
            ward_names = self.preprocessor.label_encoders['区'].classes_

        return embeddings, ward_names

    def get_attention_weights(self, ward):
        """Calculate Attention weights for a specific ward"""
        if not self.model_loaded or self.model_type != 'attention':
            return None

        try:
            ward_encoded = self.preprocessor.label_encoders['区'].transform([ward])[0]
            ward_idx = torch.LongTensor([ward_encoded]).to(self.device)

            with torch.no_grad():
                ward_emb = self.model.ward_embedding(ward_idx)
                attention_weights = self.model.attention(ward_emb)

            return attention_weights.cpu().numpy()[0]

        except Exception as e:
            print(f"Attention計算エラー: {e}")
            return None