# 東京家賃予測 - Deep Learning リファクタリング版

`rent_dl.ipynb`を機能別に3つのノートブックと共通モジュールに分離しました。

## 📁 ファイル構成

### 🔧 共通モジュール (.py)

#### 1. `rent_config.py`
設定と定数の管理
- デバイス設定 (CUDA/CPU)
- ファイルパス
- ハイパーパラメータ
- 東京23区の情報
- 特徴量カラム定義

#### 2. `rent_models.py`
ニューラルネットワークモデル
- `RentPredictionNet` - 基本モデル（Embedding + FC層）
- `RentPredictionNetWithAttention` - Attention機構付きモデル

#### 3. `rent_data.py`
データ処理クラス
- `RentDataPreprocessor` - 前処理・エンコーディング・正規化
- `TokyoRentDataset` - PyTorch Dataset

#### 4. `rent_train.py`
訓練ユーティリティ
- `train_model()` - 訓練ループ（学習率スケジューリング、早期停止）
- `evaluate_model()` - モデル評価（MAE, RMSE, R²）

#### 5. `rent_predict.py`
予測ユーティリティ
- `DeepLearningRentPredictor` - モデル読み込みと予測実行
  - 単一予測
  - バッチ予測
  - 区の埋め込み取得
  - Attentionウェイト取得

#### 6. `rent_visualize.py`
可視化ツール
- `NeuralNetworkVisualizer` - 包括的な可視化
  - ネットワーク構造
  - 埋め込みの3D可視化（Plotly）
  - 類似度マトリクス
  - Attentionウェイト
  - 訓練プロセス分析

#### 7. `rent_utils.py`
ユーティリティ関数
- `setup_japanese_font()` - 日本語フォント自動設定
  - Meiryoフォント優先
  - システムフォント自動検出
  - フォールバック対応
- `list_available_japanese_fonts()` - 利用可能フォント一覧

---

### 📓 Jupyter Notebooks

#### 1. `rent_dl_train.ipynb` - モデル訓練
**目的:** モデルの訓練と保存

**セクション:**
1. ライブラリのインポート
2. データ読み込みと前処理
3. モデルの作成（基本 + Attention）
4. モデルの訓練
5. モデルの評価と比較
6. 基本的な可視化
7. モデル保存

**出力:**
- `best_rent_model_basic.pth`
- `best_rent_model_attention.pth`
- `rent_prediction_model_basic.pth`
- `rent_prediction_model_attention.pth`
- `training_results.png`
- `ward_embeddings_2d.png`

---

#### 2. `rent_dl_predict.ipynb` - 予測と対話型UI
**目的:** 訓練済みモデルで予測を実行

**セクション:**
1. ライブラリのインポート
2. モデル読み込み
3. 単一予測の例
4. バッチ予測の例
5. モデル比較（基本 vs Attention）
6. **対話型予測UI** (`InteractiveRentPredictorDL`)
   - 3つのモード:
     - 🏠 単一予測
     - 📊 区別比較
     - 🧠 AI分析
   - ipywidgets使用
   - HTML形式のリッチな出力
7. 区の分析（埋め込み、Attention）

**特徴:**
- スライダーとドロップダウンで簡単操作
- リアルタイム予測
- 美しいビジュアル出力
- 区別の詳細比較

---

#### 3. `rent_dl_visualize.ipynb` - 高度な可視化
**目的:** モデルの詳細分析と可視化

**セクション:**
1. ライブラリのインポート
2. ビジュアライザー初期化
3. ネットワークアーキテクチャ
4. 区の埋め込み（インタラクティブ3D）
5. 区の類似度分析
6. Attention機構の可視化
7. t-SNE による埋め込み
8. 訓練プロセスの分析
9. 予測誤差の分析

**出力:**
- `network_architecture_viz.png`
- `ward_embeddings_3d.html` (インタラクティブ)
- `ward_similarity_matrix.html`
- `attention_weights_viz.html`
- `attention_analysis.png`
- `ward_embeddings_tsne.png`
- `error_analysis.png`

---

## 🚀 使用方法

### 1. 訓練
```bash
jupyter notebook rent_dl_train.ipynb
```
すべてのセルを実行してモデルを訓練・保存

### 2. 予測
```bash
jupyter notebook rent_dl_predict.ipynb
```
訓練済みモデルで予測を実行。対話型UIで簡単に操作可能

### 3. 可視化
```bash
jupyter notebook rent_dl_visualize.ipynb
```
モデルの内部構造と性能を詳細に分析

---

## 📊 モデル比較

| モデル | パラメータ数 | 埋め込み次元 | 層構成 | 特徴 |
|--------|------------|------------|--------|------|
| **基本モデル** | ~50,000 | 16次元 | 256→128→64 | シンプル、高速 |
| **Attentionモデル** | ~300,000 | 32次元 | 512→256→128 | Attention機構、Skip接続、高精度 |

---

## 🔄 元のノートブックとの違い

### 元: `rent_dl.ipynb` (660KB, 1つの巨大なノートブック)
- すべての機能が1ファイルに混在
- 再利用が困難
- 保守性が低い

### 新: 分離された構成
- **3つのノートブック** (訓練、予測、可視化)
- **6つのPythonモジュール** (再利用可能)
- 関心の分離
- 保守性の向上
- コードの重複削減

---

## 💡 推奨ワークフロー

1. **初回実行:**
   - `rent_dl_train.ipynb` でモデルを訓練

2. **予測と実験:**
   - `rent_dl_predict.ipynb` で対話的に予測

3. **分析:**
   - `rent_dl_visualize.ipynb` でモデルを深く理解

4. **カスタマイズ:**
   - 各モジュール (.py) を編集して機能拡張

---

## 📦 必要なライブラリ

```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn plotly ipywidgets networkx
```

---

## 🎯 今後の拡張案

- [ ] モデルのハイパーパラメータ最適化
- [ ] アンサンブルモデル
- [ ] 時系列予測への拡張
- [ ] Web APIの作成
- [ ] Streamlitダッシュボード

---

## ❓ FAQ（よくある質問）

### Q1: グラフに日本語が表示されない

**A:** 日本語フォント設定が原因です。以下の方法で解決できます:

1. **自動設定を使用（推奨）:**
   ```python
   from rent_utils import setup_japanese_font
   setup_japanese_font()
   ```
   すべてのノートブックの最初のセルに既に含まれています。

2. **利用可能なフォントを確認:**
   ```python
   from rent_utils import list_available_japanese_fonts
   list_available_japanese_fonts()
   ```

3. **Meiryoフォントファイルを確認:**
   `meiryo.ttc`が同じディレクトリにあることを確認してください。

### Q2: モデルの訓練にどのくらい時間がかかる？

**A:** 環境によりますが:
- **CPU:** 50エポック で約 10-15分
- **GPU:** 50エポック で約 2-5分

### Q3: メモリ不足エラーが出る

**A:** バッチサイズを減らしてください:
```python
# rent_config.py を編集
BATCH_SIZE = 16  # デフォルトは32
```

### Q4: 元の`rent_dl.ipynb`とどちらを使うべき？

**A:** 用途によります:
- **学習・実験:** 新しい分離版（保守性が高い）
- **デモ・プレゼン:** 元の統合版（1ファイルで完結）

### Q5: カスタムモデルを追加したい

**A:** `rent_models.py`に新しいクラスを追加:
```python
class CustomRentModel(nn.Module):
    def __init__(self, ...):
        # カスタム実装
```

その後、`rent_dl_train.ipynb`でインポートして使用できます。

---

## 🐛 トラブルシューティング

### エラー: `ModuleNotFoundError: No module named 'rent_xxx'`

**原因:** Pythonモジュールが見つかりません。

**解決策:**
```bash
# ノートブックと同じディレクトリで実行していることを確認
pwd  # /workspace/ai-edu/rent が表示されるべき

# または、パスを追加
import sys
sys.path.append('/workspace/ai-edu/rent')
```

### エラー: `FileNotFoundError: tokyo_rent_data_v2.csv`

**原因:** データファイルが見つかりません。

**解決策:** CSVファイルが同じディレクトリにあることを確認してください。

### 日本語の□□□（tofu）が表示される

**原因:** フォントが正しく読み込まれていません。

**解決策:**
```python
# ノートブックの最初のセルで実行
from rent_utils import setup_japanese_font
setup_japanese_font()

# キャッシュをクリア
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fm._rebuild()
```

---

## 📝 ライセンス

このプロジェクトは教育目的で作成されました。
