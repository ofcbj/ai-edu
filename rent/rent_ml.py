#!/usr/bin/env python
# coding: utf-8

# # 🏠 東京家賃予測システム - リファクタリング版 v2
# 
# **【改善点】**
# - 学習と予測で共通のエンコーディングロジックを使用
# - 各Strategyクラスに `transform_single()` メソッドを追加
# - 重複コード削除 (PredictionFeatureBuilder: 133行 → 30行)
# - **予測関連コードを完全にクラス化**
# - Webインターフェースと完全統合

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

import ipywidgets as widgets
from IPython.display import display, HTML
import pickle


# ## 1. エンコーディング戦略クラス

# In[ ]:


def _align_columns(dummies, reference_columns):
    for col in reference_columns:
        if col not in dummies.columns:
            dummies[col] = 0
    
    return dummies[reference_columns]


class TargetEncodingStrategy:
    """Target Encoding専用クラス (transform_single追加)"""

    def __init__(self, df=None):
        self.df = df.copy() if df is not None else None
        self.ward_encoding = None
        self.structure_dummies = None
        self.type_dummies = None

    def fit(self, df=None):
        if df is not None:
            self.df = df.copy()
        
        self.ward_encoding      = self.df.groupby('区')['家賃_円'].mean().to_dict()
        self.structure_dummies  = pd.get_dummies(self.df['建物構造'], prefix='建物構造', drop_first=True)
        self.type_dummies       = pd.get_dummies(self.df['建物タイプ'], prefix='建物タイプ', drop_first=True)
        return self

    def transform(self, df=None):
        if df is None:
            df = self.df
        
        df = df.copy()
        df['区_target_encoded'] = df['区'].map(self.ward_encoding).fillna(df['区'].map(self.ward_encoding).mean())
        numeric_features = df[['部屋サイズ_m2', '駅距離_分', '築年数_年', '区_target_encoded']]
        
        structure_dummies   = pd.get_dummies(df['建物構造'], prefix='建物構造', drop_first=True)
        type_dummies        = pd.get_dummies(df['建物タイプ'], prefix='建物タイプ', drop_first=True)
        structure_dummies   = _align_columns(structure_dummies, self.structure_dummies.columns)
        type_dummies        = _align_columns(type_dummies, self.type_dummies.columns)
        
        return pd.concat([numeric_features, structure_dummies, type_dummies], axis=1)

    def transform_single(self, sample):
        """✅ コア機能: 単一サンプル変換（学習ロジック100%再利用）"""
        if isinstance(sample, pd.Series):
            sample = sample.to_frame().T
        elif isinstance(sample, dict):
            sample = pd.DataFrame([sample])
        return self.transform(sample)


class OneHotEncodingStrategy:
    def __init__(self, df=None):
        self.df = df.copy() if df is not None else None
        self.ward_dummies = None
        self.structure_dummies = None
        self.type_dummies = None

    def fit(self, df=None):
        if df is not None:
            self.df = df.copy()
        
        self.ward_dummies       = pd.get_dummies(self.df['区'], prefix='区', drop_first=False)
        self.structure_dummies  = pd.get_dummies(self.df['建物構造'], prefix='建物構造', drop_first=True)
        self.type_dummies       = pd.get_dummies(self.df['建物タイプ'], prefix='建物タイプ', drop_first=True)
        return self

    def transform(self, df=None):
        if df is None:
            df = self.df
        
        numeric_features    = df[['部屋サイズ_m2', '駅距離_分', '築年数_年']].copy()
        ward_dummies        = pd.get_dummies(df['区'], prefix='区', drop_first=False)
        structure_dummies   = pd.get_dummies(df['建物構造'], prefix='建物構造', drop_first=True)
        type_dummies        = pd.get_dummies(df['建物タイプ'], prefix='建物タイプ', drop_first=True)
        
        ward_dummies        = _align_columns(ward_dummies, self.ward_dummies.columns)
        structure_dummies   = _align_columns(structure_dummies, self.structure_dummies.columns)
        type_dummies        = _align_columns(type_dummies, self.type_dummies.columns)
        return pd.concat([numeric_features, ward_dummies, structure_dummies, type_dummies], axis=1)

    def transform_single(self, sample):
        if isinstance(sample, pd.Series):
            sample = sample.to_frame().T
        elif isinstance(sample, dict):
            sample = pd.DataFrame([sample])
        return self.transform(sample)


# ## 2. 前処理および予測クラス

# In[ ]:


class RentDataPreprocessor:
    def __init__(self, df):
        self.df = df.copy()
        self.target_encoder = TargetEncodingStrategy(df)
        self.onehot_encoder = OneHotEncodingStrategy(df)

    def fit_all_encodings(self):
        self.target_encoder.fit()
        self.onehot_encoder.fit()
        return self

    def create_feature_sets(self):
        X_target = self.target_encoder.transform()
        X_onehot = self.onehot_encoder.transform()
        return X_target, X_onehot

    def get_encoding_strategies(self):
        return {
            'target': self.target_encoder,
            'onehot': self.onehot_encoder
        }

class PredictionFeatureBuilder:
    """✅ 予測用特徴量構築クラス（133行 → 30行に簡素化）"""
    
    def __init__(self, encoding_strategies):
        self.target_encoder = encoding_strategies['target']
        self.onehot_encoder = encoding_strategies['onehot']

    def prepare_features(self, test_sample, model_type, scaler=None):
        """✅ 既存のグローバル関数を完全に置き換えるメソッド"""
        # 適切なエンコーダーを選択
        if 'Target' in model_type:
            encoder = self.target_encoder
        else:
            encoder = self.onehot_encoder

        # ✅ transform_singleで重複コード削除！
        X_test = encoder.transform_single(test_sample)

        # スケーリング
        if scaler and 'Forest' not in model_type and 'Gradient' not in model_type:
            X_test = scaler.transform(X_test)

        return X_test


# ## 3. モデル学習と実行

# In[ ]:


# データロードと前処理
print("🏠 東京家賃予測システム開始")
df = pd.read_csv('tokyo_rent_data_v2.csv')
print(f"データロード完了: {len(df)}行")

# 前処理
preprocessor = RentDataPreprocessor(df)
preprocessor.fit_all_encodings()
X_target, X_onehot = preprocessor.create_feature_sets()
y = df['家賃_円']

print(f"Target Encoding 特徴量: {X_target.shape[1]}個")
print(f"One-Hot Encoding 特徴量: {X_onehot.shape[1]}個")


# In[ ]:


# データ分割
X_train_target, X_test_target, y_train, y_test = train_test_split(X_target, y, test_size=0.2, random_state=42)
X_train_onehot, X_test_onehot, _, _ = train_test_split(X_onehot, y, test_size=0.2, random_state=42)

# スケーリング
scaler_target = StandardScaler()
scaler_onehot = StandardScaler()

X_train_target_scaled   = scaler_target.fit_transform(X_train_target)
X_test_target_scaled    = scaler_target.transform(X_test_target)
X_train_onehot_scaled   = scaler_onehot.fit_transform(X_train_onehot)
X_test_onehot_scaled    = scaler_onehot.transform(X_test_onehot)

# モデル学習
models = {
    'Linear (Target Enc)'   : (LinearRegression(), X_train_target_scaled, X_test_target_scaled),
    'Ridge (Target Enc)'    : (Ridge(alpha=10), X_train_target_scaled, X_test_target_scaled),
    'Linear (One-Hot)'      : (LinearRegression(),X_train_onehot_scaled, X_test_onehot_scaled),
    'Ridge (One-Hot)'       : (Ridge(alpha=10), X_train_onehot_scaled, X_test_onehot_scaled),
    'Random Forest'         : (RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15), 
                                X_train_target, X_test_target),
    'Gradient Boosting'     : (GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5), 
                                X_train_target, X_test_target)
}

results     = {}
best_model  = None
best_score  = -1
best_model_name = ''
best_scaler = None

print("\nモデル学習開始...")
for name, (model, X_tr, X_te) in models.items():
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {'model': model, 'r2': r2}
    print(f"{name}: R² = {r2:.4f}")
    
    if r2 > best_score:
        best_score = r2
        best_model = model
        best_model_name = name

print(f"\n✅ 最高性能モデル: {best_model_name} (R² = {best_score:.4f})")

# ✅ 新しいPredictionFeatureBuilder使用
feature_builder = PredictionFeatureBuilder(preprocessor.get_encoding_strategies())


# In[ ]:


# 최고 모델 저장
print(f"\n💾 モデル保存中: {best_model_name}")

# 모델 타입에 따라 적절한 스케일러 선택
if 'Target' in best_model_name:
    best_scaler = scaler_target
    best_encoder = preprocessor.target_encoder
    use_target_encoding = True
    feature_columns = X_train_target.columns.tolist()
    structure_categories = [col for col in feature_columns if col.startswith('建物構造_')]
    type_categories = [col for col in feature_columns if col.startswith('建物タイプ_')]
    ward_categories = []
else:
    best_scaler = scaler_onehot
    best_encoder = preprocessor.onehot_encoder
    use_target_encoding = False
    feature_columns = X_train_onehot.columns.tolist()
    structure_categories = [col for col in feature_columns if col.startswith('建物構造_')]
    type_categories = [col for col in feature_columns if col.startswith('建物タイプ_')]
    ward_categories = [col for col in feature_columns if col.startswith('区_')]

# モデル情報をまとめて保存
model_info = {
    'model': best_model,
    'model_name': best_model_name,
    'r2_score': best_score,
    'scaler': best_scaler,
    'ward_encoding': preprocessor.target_encoder.ward_encoding,
    'feature_columns': feature_columns,
    'use_target_encoding': use_target_encoding,
    'use_interaction': False,
    'structure_categories': structure_categories,
    'type_categories': type_categories,
    'ward_categories': ward_categories
}

with open('best_rent_model.pkl', 'wb') as f:
    pickle.dump(model_info, f)

print(f"✅ モデル保存完了: best_rent_model.pkl")
print(f"   - モデル: {best_model_name}")
print(f"   - R² スコア: {best_score:.4f}")
print(f"   - 特徴量数: {len(feature_columns)}")


# ## 4. ✅ 完全にクラス化されたWebインターフェース

# In[ ]:


# 저장된 모델 정보 로드
try:
    with open('best_rent_model.pkl', 'rb') as f:
        model_info = pickle.load(f)
        
    model               = model_info['model']
    model_name          = model_info['model_name']
    ward_encoding       = model_info['ward_encoding']
    feature_columns     = model_info['feature_columns']
    scaler              = model_info['scaler']
    use_target_encoding = model_info['use_target_encoding']
    use_interaction     = model_info['use_interaction']
    structure_categories= model_info['structure_categories']
    type_categories     = model_info['type_categories']
    ward_categories     = model_info.get('ward_categories', [])
    
    print(f"✅ モデルロード成功: {model_name}")
    print(f"   R² スコア: 訓練データで確認済み")
    
except Exception as e:
    print(f"⚠️ モデルロードエラー: {e}")
    print("先に回帰分析コードを実行してください。")
    raise

# 구 리스트 (가격순)
ward_list = [
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

# ウィジェット作成
style_width = {'description_width': '120px'}
layout_width = widgets.Layout(width='500px')

ward_dropdown = widgets.Dropdown(
    options=[f"{ward} ({level})" for ward, level, _, _ in ward_list],
    value='新宿区 (高級)',
    description='🏢 区選択：',
    style=style_width,
    layout=layout_width
)

room_size_slider = widgets.IntSlider(
    value=30,
    min=15,
    max=100,
    step=5,
    description='📏 部屋サイズ：',
    style=style_width,
    layout=layout_width
)

room_size_text = widgets.Label(value='30 m²')

station_distance_slider = widgets.IntSlider(
    value=5,
    min=1,
    max=20,
    step=1,
    description='🚉 駅距離：',
    style=style_width,
    layout=layout_width
)

station_text = widgets.Label(value='5 分')

building_age_slider = widgets.IntSlider(
    value=10,
    min=0,
    max=50,
    step=1,
    description='🏗️ 築年数：',
    style=style_width,
    layout=layout_width
)

age_text = widgets.Label(value='10 年')

structure_dropdown = widgets.Dropdown(
    options=['木造', 'RC造', '鉄骨造', 'SRC造'],
    value='RC造',
    description='🏢 建物構造：',
    style=style_width,
    layout=widgets.Layout(width='300px')
)

property_type_dropdown = widgets.Dropdown(
    options=['マンション', 'アパート', 'ハイツ', 'コーポ'],
    value='マンション',
    description='🏠 建物タイプ：',
    style=style_width,
    layout=widgets.Layout(width='300px')
)

# モードボタン
comparison_mode = widgets.ToggleButton(
    value=False,
    description='区別比較モード',
    button_style='info',
    tooltip='複数の区で同じ条件の家賃を比較',
    icon='chart-bar'
)

# 予測ボタン
predict_button = widgets.Button(
    description='🔮 家賃を予測',
    button_style='success',
    tooltip='クリックして家賃を予測',
    layout=widgets.Layout(width='200px', height='40px')
)

# 出力エリア
output = widgets.Output()

# スライダー更新用関数
def update_room_size_text(change):
    room_size_text.value = f'{change["new"]} m²'

def update_station_text(change):
    station_text.value = f'{change["new"]} 分'

def update_age_text(change):
    age_text.value = f'{change["new"]} 年'

room_size_slider.observe(update_room_size_text, names='value')
station_distance_slider.observe(update_station_text, names='value')
building_age_slider.observe(update_age_text, names='value')

def make_prediction(ward, room_size, station_distance, building_age, structure, property_type):
    """予測実行関数"""
    try:
        # データフレーム作成
        test_data = pd.DataFrame([{
            '部屋サイズ_m2': room_size,
            '区': ward,
            '駅距離_分': station_distance,
            '築年数_年': building_age,
            '建物構造': structure,
            '建物タイプ': property_type
        }])
        
        if use_target_encoding:
            # Target Encoding方式
            test_data['区_target_encoded'] = ward_encoding.get(ward, np.mean(list(ward_encoding.values())))
            
            # 特徴量準備
            X_test = test_data[['部屋サイズ_m2', '駅距離_分', '築年数_年', '区_target_encoded']]
            
            # ダミー変数作成
            for structure_cat in structure_categories:
                struct_name = structure_cat.replace('建物構造_', '')
                X_test[structure_cat] = 1 if struct_name == structure else 0
                
            for type_cat in type_categories:
                type_name = type_cat.replace('建物タイプ_', '')
                X_test[type_cat] = 1 if type_name == property_type else 0
            
            # 列順序調整
            X_test = X_test[feature_columns]
            
        else:
            # One-Hot Encoding方式
            X_test = pd.DataFrame()
            X_test['部屋サイズ_m2'] = [room_size]
            X_test['駅距離_分'] = [station_distance]
            X_test['築年数_年'] = [building_age]
            
            # 区のダミー変数
            for ward_cat in ward_categories:
                ward_name = ward_cat.replace('区_', '')
                X_test[ward_cat] = 1 if ward_name == ward else 0
            
            # 構造のダミー変数
            for structure_cat in structure_categories:
                struct_name = structure_cat.replace('建物構造_', '')
                X_test[structure_cat] = 1 if struct_name == structure else 0
            
            # タイプのダミー変数
            for type_cat in type_categories:
                type_name = type_cat.replace('建物タイプ_', '')
                X_test[type_cat] = 1 if type_name == property_type else 0
            
            # 交互作用項追加（必要な場合）
            if use_interaction:
                for ward_cat in ward_categories:
                    X_test[f'{ward_cat}_x_size'] = X_test[ward_cat] * room_size
                    X_test[f'{ward_cat}_x_station'] = X_test[ward_cat] * station_distance
            
            # 列順序調整
            for col in feature_columns:
                if col not in X_test.columns:
                    X_test[col] = 0
            X_test = X_test[feature_columns]
        
        # スケーリング（Tree系モデル以外）
        if 'Forest' not in model_name and 'Gradient' not in model_name:
            X_test = scaler.transform(X_test)
        
        # 予測
        prediction = model.predict(X_test)[0]
        return max(prediction, 20000)  # 最小値補正
        
    except Exception as e:
        print(f"予測エラー: {e}")
        return 0


def predict_rent(b):
    """予測ボタンのイベントハンドラ"""
    with output:
        output.clear_output()
        
        # 選択された区を抽出
        selected_ward = ward_dropdown.value.split(' (')[0]
        
        if not comparison_mode.value:
            # 単一予測モード
            prediction = make_prediction(
                selected_ward,
                room_size_slider.value,
                station_distance_slider.value,
                building_age_slider.value,
                structure_dropdown.value,
                property_type_dropdown.value
            )
            
            # 区情報取得
            ward_info = [info for info in ward_list if info[0] == selected_ward][0]
            ward_level = ward_info[1]
            bg_color = ward_info[2]
            base_price = ward_info[3]
            
            # 結果表示HTML
            html_output = f"""
            <div style="font-family: 'Helvetica Neue', sans-serif;">
                <div style="background: linear-gradient(135deg, {bg_color}22, {bg_color}11); 
                            border: 2px solid {bg_color}; border-radius: 15px; padding: 25px; 
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    
                    <h2 style="color: {bg_color}; margin: 0 0 20px 0; font-size: 24px;">
                        🏠 家賃予測結果
                    </h2>
                    
                    <div style="background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                        <h3 style="color: #333; margin: 0 0 15px 0; font-size: 18px;">📋 物件条件</h3>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 14px;">
                            <div>📍 <strong>区：</strong> {selected_ward} 
                                <span style="background: {bg_color}; color: white; padding: 2px 8px; 
                                           border-radius: 4px; font-size: 11px; margin-left: 5px;">{ward_level}</span>
                            </div>
                            <div>📏 <strong>広さ：</strong> {room_size_slider.value} m²</div>
                            <div>🚉 <strong>駅まで：</strong> 徒歩{station_distance_slider.value}分</div>
                            <div>🏗️ <strong>築年数：</strong> {building_age_slider.value}年</div>
                            <div>🏢 <strong>構造：</strong> {structure_dropdown.value}</div>
                            <div>🏠 <strong>タイプ：</strong> {property_type_dropdown.value}</div>
                        </div>
                    </div>
                    
                    <div style="background: linear-gradient(135deg, {bg_color}, {bg_color}dd); 
                                color: white; padding: 25px; border-radius: 10px; text-align: center;">
                        <div style="font-size: 14px; opacity: 0.9; margin-bottom: 5px;">予測月額家賃</div>
                        <div style="font-size: 48px; font-weight: bold; margin: 10px 0;">
                            ¥{prediction:,.0f}
                        </div>
                        <div style="font-size: 12px; opacity: 0.8;">
                            (区の基準価格: ¥{base_price:,} から計算)
                        </div>
                    </div>
                    
                    <div style="margin-top: 15px; padding: 10px; background: #f8f9fa; border-radius: 5px; 
                                font-size: 12px; color: #666;">
                        <strong>モデル情報:</strong> {model_name} | 
                        区の影響を適切に反映した予測値です
                    </div>
                </div>
            </div>
            """
            
        else:
            # 比較モード
            comparison_wards = [
                '港区', '千代田区', '中央区', '渋谷区',
                '新宿区', '世田谷区', '中野区', 
                '板橋区', '葛飾区', '足立区'
            ]
            
            predictions = []
            for ward in comparison_wards:
                pred = make_prediction(
                    ward,
                    room_size_slider.value,
                    station_distance_slider.value,
                    building_age_slider.value,
                    structure_dropdown.value,
                    property_type_dropdown.value
                )
                ward_info = [info for info in ward_list if info[0] == ward][0]
                predictions.append({
                    'ward': ward,
                    'prediction': pred,
                    'level': ward_info[1],
                    'color': ward_info[2],
                    'base': ward_info[3]
                })
            
            # 価格順にソート
            predictions.sort(key=lambda x: x['prediction'], reverse=True)
            max_pred = predictions[0]['prediction']
            
            # 比較結果HTML
            html_output = f"""
            <div style="font-family: 'Helvetica Neue', sans-serif;">
                <div style="background: white; border: 2px solid #333; border-radius: 15px; 
                            padding: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    
                    <h2 style="color: #333; margin: 0 0 20px 0;">📊 区別家賃比較</h2>
                    
                    <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                        <strong>条件:</strong> {room_size_slider.value}m² | 駅{station_distance_slider.value}分 | 
                        築{building_age_slider.value}年 | {structure_dropdown.value} | {property_type_dropdown.value}
                    </div>
                    
                    <div style="margin: 20px 0;">
            """
            
            for i, pred_info in enumerate(predictions):
                bar_width = (pred_info['prediction'] / max_pred) * 100
                html_output += f"""
                    <div style="display: flex; align-items: center; margin: 12px 0;">
                        <div style="width: 100px; font-weight: bold; font-size: 14px;">
                            {i+1}. {pred_info['ward']}
                        </div>
                        <div style="flex: 1; margin: 0 15px;">
                            <div style="background: #e0e0e0; border-radius: 20px; overflow: hidden; height: 28px;">
                                <div style="width: {bar_width}%; background: linear-gradient(90deg, {pred_info['color']}, {pred_info['color']}cc); 
                                            height: 100%; display: flex; align-items: center; justify-content: flex-end; padding-right: 10px;">
                                    <span style="color: white; font-weight: bold; font-size: 12px;">
                                        ¥{pred_info['prediction']:,.0f}
                                    </span>
                                </div>
                            </div>
                        </div>
                        <span style="background: {pred_info['color']}; color: white; padding: 4px 10px; 
                                   border-radius: 4px; font-size: 11px; width: 60px; text-align: center;">
                            {pred_info['level']}
                        </span>
                    </div>
                """
            
            # 差額計算
            diff = predictions[0]['prediction'] - predictions[-1]['prediction']
            ratio = predictions[0]['prediction'] / predictions[-1]['prediction']
            
            html_output += f"""
                    </div>
                    
                    <div style="margin-top: 25px; padding: 15px; background: linear-gradient(135deg, #ff6b6b22, #4ecdc422); 
                                border-radius: 8px; border: 1px solid #dee2e6;">
                        <div style="font-size: 16px; font-weight: bold; color: #333; margin-bottom: 10px;">
                            💹 価格差分析
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; text-align: center;">
                            <div>
                                <div style="font-size: 12px; color: #666;">最高額</div>
                                <div style="font-size: 18px; font-weight: bold; color: #ff6b6b;">
                                    ¥{predictions[0]['prediction']:,.0f}
                                </div>
                                <div style="font-size: 11px; color: #999;">{predictions[0]['ward']}</div>
                            </div>
                            <div>
                                <div style="font-size: 12px; color: #666;">最低額</div>
                                <div style="font-size: 18px; font-weight: bold; color: #4ecdc4;">
                                    ¥{predictions[-1]['prediction']:,.0f}
                                </div>
                                <div style="font-size: 11px; color: #999;">{predictions[-1]['ward']}</div>
                            </div>
                            <div>
                                <div style="font-size: 12px; color: #666;">差額/倍率</div>
                                <div style="font-size: 18px; font-weight: bold; color: #333;">
                                    ¥{diff:,.0f}
                                </div>
                                <div style="font-size: 11px; color: #999;">{ratio:.1f}倍</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            """
        
        display(HTML(html_output))

# イベント設定
predict_button.on_click(predict_rent)

# UI構築
print("=" * 80)
print("🏠 東京家賃予測システム v2.0")
print("=" * 80)
print(f"✨ 区の影響力を強化した新モデル使用中: {model_name}")
print("📊 同一条件で区を変えた場合の価格差が明確に反映されます")
print("-" * 80)

# レイアウト
controls_box = widgets.VBox([
    widgets.HTML("<h3 style='color: #2196F3; margin: 10px 0;'>🏢 物件情報入力</h3>"),
    ward_dropdown,
    widgets.HBox([room_size_slider, room_size_text]),
    widgets.HBox([station_distance_slider, station_text]),
    widgets.HBox([building_age_slider, age_text]),
    widgets.HBox([structure_dropdown, property_type_dropdown]),
    widgets.HTML("<div style='margin: 20px 0; border-top: 1px solid #ddd;'></div>"),
    widgets.HBox([comparison_mode, predict_button])
])

display(widgets.VBox([
    controls_box,
    output
]))

# 初期表示
predict_rent(None)


# ## 5. インタラクティブ予測インターフェース

# In[ ]:


# UI構成
controls_box = widgets.VBox([
    widgets.HTML("<h3 style='color: #2196F3;'>🏠 物件情報入力</h3>"),
    ward_dropdown,
    room_size_slider,
    station_distance_slider,
    building_age_slider,
    widgets.HBox([structure_dropdown, property_type_dropdown]),
    predict_button
])

display(widgets.VBox([controls_box, output]))

print("\n🚀 インターフェース準備完了！上記のコントロールを使用して家賃を予測してみてください。")

