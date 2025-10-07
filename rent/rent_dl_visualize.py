"""
Neural Network Visualizer for Rent Prediction Models

This module provides visualization tools for analyzing deep learning rent prediction models,
including architecture visualization, embedding analysis, and attention weight inspection.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.font_manager as fmt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import warnings
import os

# 日本語フォント設定
try:
    font_path = 'meiryo.ttc'
    if os.path.exists(font_path):
        fmt.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = ['Meiryo']
    else:
        # フォントファイルがない場合はシステムの日本語フォントを使用
        plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
except Exception as e:
    print(f"Warning: 日本語フォント設定エラー: {e}")
    plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']

plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NeuralNetworkVisualizer:
    """Visualizer for neural network rent prediction models

    This class provides comprehensive visualization tools for analyzing
    rent prediction models including architecture, embeddings, and attention weights.
    """

    def __init__(self, model_type='basic', model_path='rent_prediction_model_basic.pth',
                 data_path='tokyo_rent_data_v2.csv'):
        """Initialize the visualizer with model and data

        Parameters:
        -----------
        model_type : str
            'basic' (RentPredictionNet) or 'attention' (RentPredictionNetWithAttention)
        model_path : str
            Path to the saved model file (default: 'rent_prediction_model_basic.pth')
        data_path : str
            Path to the CSV data file (default: 'tokyo_rent_data_v2.csv')
        """
        self.model = None
        self.preprocessor = None
        self.df = None
        self.model_type = model_type
        self.data_path = data_path
        self.load_model_and_data(model_path)

    def _ensure_japanese_font(self):
        """日本語フォントが設定されていることを確認"""
        try:
            if os.path.exists('meiryo.ttc'):
                plt.rcParams['font.family'] = ['Meiryo']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            pass

    def load_model_and_data(self, model_path='rent_prediction_model_basic.pth'):
        """Load model and data

        Parameters:
        -----------
        model_path : str
            Path to the saved model checkpoint
        """
        # Import model classes (must be available in the environment)
        try:
            from rent_dl_models import RentPredictionNet, RentPredictionNetWithAttention
        except ImportError:
            raise ImportError(
                "Could not import model classes. Make sure rent_dl.py is in the same directory "
                "or the model classes are available in your environment."
            )

        # Load model checkpoint
        checkpoint = torch.load(model_path,
                              map_location=device,
                              weights_only=False)

        self.preprocessor = checkpoint['preprocessor']
        config = checkpoint['model_config']

        # Load appropriate model based on type
        if self.model_type == 'basic':
            self.model = RentPredictionNet(
                config['num_wards'],
                config['num_structures'],
                config['num_types']
            ).to(device)
        elif self.model_type == 'attention':
            self.model = RentPredictionNetWithAttention(
                config['num_wards'],
                config['num_structures'],
                config['num_types']
            ).to(device)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}. Use 'basic' or 'attention'.")

        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Load data
        self.df = pd.read_csv(self.data_path)

    def visualize_architecture(self):
        """Visualize network architecture

        Creates a comprehensive visualization showing:
        1. Network structure diagram
        2. Parameter count distribution
        3. Ward embedding visualization
        4. Activation function distribution
        """
        # 日本語フォント再設定
        self._ensure_japanese_font()

        fig = plt.figure(figsize=(16, 10))

        # 1. Network structure diagram
        ax1 = plt.subplot(2, 2, 1)

        # Layer sizes based on model type
        if self.model_type == 'basic':
            layer_sizes = [7, 16, 256, 128, 64, 1]
            layer_names = ['Input\n(7)', 'Embed\n(16)', 'Hidden1\n(256)',
                          'Hidden2\n(128)', 'Hidden3\n(64)', 'Output\n(1)']
        else:  # attention
            layer_sizes = [7, 32, 512, 256, 128, 64, 1]
            layer_names = ['Input\n(7)', 'Embed\n(32)', 'Hidden1\n(512)',
                          'Hidden2\n(256)', 'Hidden3\n(128)', 'Hidden4\n(64)', 'Output\n(1)']

        # Create network graph
        G = nx.DiGraph()
        pos = {}
        node_colors = []

        for i, (size, name) in enumerate(zip(layer_sizes, layer_names)):
            for j in range(min(size, 10)):  # Show max 10 nodes per layer
                node_id = f"L{i}N{j}"
                G.add_node(node_id)
                pos[node_id] = (i * 2, j - min(size, 10)/2)

                if i == 0:
                    node_colors.append('#3498db')
                elif i == len(layer_sizes) - 1:
                    node_colors.append('#e74c3c')
                else:
                    node_colors.append('#95a5a6')

                # Connect to previous layer
                if i > 0:
                    prev_size = min(layer_sizes[i-1], 10)
                    for k in range(prev_size):
                        G.add_edge(f"L{i-1}N{k}", node_id)

        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                              node_size=100, alpha=0.8, ax=ax1)
        nx.draw_networkx_edges(G, pos, alpha=0.1, ax=ax1)

        # Add layer names
        for i, name in enumerate(layer_names):
            ax1.text(i * 2, -6, name, ha='center', fontsize=10, fontweight='bold')

        ax1.set_title('Neural Network Architecture', fontsize=14, fontweight='bold')
        ax1.axis('off')

        # 2. Parameter count distribution
        ax2 = plt.subplot(2, 2, 2)

        # Calculate parameter counts per layer
        param_counts = []
        layer_names_short = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_counts.append(param.numel())
                layer_names_short.append(name.split('.')[0])

        # Show top 10 layers by parameter count
        top_indices = np.argsort(param_counts)[-10:]
        top_params = [param_counts[i] for i in top_indices]
        top_names = [layer_names_short[i] for i in top_indices]

        bars = ax2.barh(range(len(top_params)), top_params, color='#3498db')
        ax2.set_yticks(range(len(top_params)))
        ax2.set_yticklabels(top_names, fontsize=9)
        ax2.set_xlabel('Number of Parameters')
        ax2.set_title('Top 10 Layers by Parameter Count', fontsize=12, fontweight='bold')

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_params)):
            ax2.text(val, i, f' {val:,}', va='center', fontsize=8)

        # 3. Embedding dimension visualization
        ax3 = plt.subplot(2, 2, 3)

        ward_embeddings = self.model.ward_embedding.weight.detach().cpu().numpy()

        # Project to 2D using PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(ward_embeddings)

        # Ward average prices
        ward_names = self.preprocessor.label_encoders['区'].classes_
        ward_prices = []
        for ward in ward_names:
            avg_price = self.df[self.df['区'] == ward]['家賃_円'].mean()
            ward_prices.append(avg_price)

        scatter = ax3.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                c=ward_prices, cmap='RdYlBu_r',
                                s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

        # Add labels for important wards
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

        # 4. Activation function distribution
        ax4 = plt.subplot(2, 2, 4)

        # Collect activation values with random input
        sample_input = torch.randn(100, 7).to(device)
        activations = []

        def hook_fn(module, input, output):
            activation_type = nn.LeakyReLU if self.model_type == 'attention' else nn.ReLU
            if isinstance(module, activation_type):
                activations.append(output.detach().cpu().numpy().flatten())

        # Register hooks
        hooks = []
        for module in self.model.modules():
            activation_type = nn.LeakyReLU if self.model_type == 'attention' else nn.ReLU
            if isinstance(module, activation_type):
                hooks.append(module.register_forward_hook(hook_fn))

        # Forward pass with dummy data
        with torch.no_grad():
            ward_idx = torch.randint(0, self.model.num_wards, (100,)).to(device)
            structure_idx = torch.randint(0, 4, (100,)).to(device)
            type_idx = torch.randint(0, 4, (100,)).to(device)
            numeric_features = torch.randn(100, 3).to(device)
            ward_avg_price = torch.randn(100).to(device)

            _ = self.model(ward_idx, structure_idx, type_idx, numeric_features, ward_avg_price)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Plot activation distribution
        if activations:
            all_activations = np.concatenate(activations)
            ax4.hist(all_activations, bins=50, alpha=0.7, color='#2ecc71', edgecolor='black')
            ax4.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            ax4.set_xlabel('Activation Value')
            ax4.set_ylabel('Frequency')
            activation_name = 'LeakyReLU' if self.model_type == 'attention' else 'ReLU'
            ax4.set_title(f'{activation_name} Activation Distribution', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig('network_architecture_viz.png', dpi=150, bbox_inches='tight')
        plt.show()

        # Print model summary
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("\n" + "="*60)
        print("Model Structure Summary")
        print("="*60)
        print(f"Total Parameters: {total_params:,}")
        if hasattr(self.model, 'embedding_dim'):
            print(f"Embedding Dimension: {self.model.embedding_dim}")
        if hasattr(self.model, 'hidden_dims'):
            print(f"Hidden Layers: {self.model.hidden_dims}")
        if hasattr(self.model, 'num_wards'):
            print(f"Number of Wards: {self.model.num_wards}")

    def visualize_embeddings_interactive(self):
        """Interactive embedding visualization using Plotly

        Creates interactive visualizations showing:
        - t-SNE projection
        - PCA projection
        - 3D PCA projection
        - Embedding heatmap
        """
        # 日本語フォント再設定
        self._ensure_japanese_font()

        # Get embeddings
        ward_embeddings = self.model.ward_embedding.weight.detach().cpu().numpy()
        ward_names = self.preprocessor.label_encoders['区'].classes_

        # Ward statistics
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

        # t-SNE 2D projection
        tsne = TSNE(n_components=2, random_state=42, perplexity=15)
        embeddings_tsne = tsne.fit_transform(ward_embeddings)

        # PCA 2D projection
        pca = PCA(n_components=2)
        embeddings_pca = pca.fit_transform(ward_embeddings)

        # PCA 3D projection
        pca_3d = PCA(n_components=3)
        embeddings_3d = pca_3d.fit_transform(ward_embeddings)

        # Create Plotly subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('t-SNE Projection', 'PCA Projection',
                          '3D PCA Projection', 'Embedding Heatmap'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter3d'}, {'type': 'heatmap'}]]
        )

        # 1. t-SNE plot
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

        # 2. PCA plot
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

        # 3. 3D PCA plot
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

        # 4. Embedding heatmap
        fig.add_trace(
            go.Heatmap(
                z=ward_embeddings[:10, :],  # Top 10 wards only
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

        # Update layout
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

        # Show similarity matrix
        self.plot_similarity_matrix(ward_embeddings, ward_names)

    def plot_similarity_matrix(self, embeddings, names):
        """Plot cosine similarity matrix

        Parameters:
        -----------
        embeddings : np.ndarray
            Embedding vectors
        names : list
            Names corresponding to each embedding
        """
        # 日本語フォント再設定
        self._ensure_japanese_font()

        # Calculate cosine similarity
        norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        similarity_matrix = np.dot(norm_embeddings, norm_embeddings.T)

        # Plotly heatmap
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
        """Visualize attention weights

        Note: Only available for attention-based models
        """
        # 日本語フォント再設定
        self._ensure_japanese_font()

        # Skip if basic model
        if self.model_type == 'basic':
            print("⚠️ Basic model does not have attention mechanism.")
            return

        ward_names = self.preprocessor.label_encoders['区'].classes_
        attention_weights_all = []

        # Collect attention weights for each ward
        with torch.no_grad():
            for i in range(len(ward_names)):
                ward_idx = torch.LongTensor([i]).to(device)
                ward_emb = self.model.ward_embedding(ward_idx)
                attention_weight = self.model.attention(ward_emb)
                attention_weights_all.append(attention_weight.cpu().numpy()[0, 0])

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # 1. Attention weights bar chart
        sorted_indices = np.argsort(attention_weights_all)[::-1]
        sorted_weights = [attention_weights_all[i] for i in sorted_indices]
        sorted_names = [ward_names[i] for i in sorted_indices]

        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(sorted_weights)))
        bars = axes[0].barh(range(len(sorted_weights)), sorted_weights, color=colors)
        axes[0].set_yticks(range(len(sorted_weights)))
        axes[0].set_yticklabels(sorted_names, fontsize=8)
        axes[0].set_xlabel('Attention Weight')
        axes[0].set_title('Attention Weights by Ward', fontsize=14, fontweight='bold')

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, sorted_weights)):
            axes[0].text(val, i, f' {val:.4f}', va='center', fontsize=8)

        # 2. Attention vs average price relationship
        avg_prices = []
        for ward in ward_names:
            avg_price = self.df[self.df['区'] == ward]['家賃_円'].mean()
            avg_prices.append(avg_price)

        axes[1].scatter(avg_prices, attention_weights_all, alpha=0.7, s=100)
        axes[1].set_xlabel('Average Rent (¥)')
        axes[1].set_ylabel('Attention Weight')
        axes[1].set_title('Attention Weight vs Average Rent', fontsize=14, fontweight='bold')

        # Label important wards
        for i, ward in enumerate(ward_names):
            if ward in ['港区', '千代田区', '足立区', '葛飾区']:
                axes[1].annotate(ward, (avg_prices[i], attention_weights_all[i]),
                               fontsize=9, ha='center',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

        # Calculate correlation
        correlation = np.corrcoef(avg_prices, attention_weights_all)[0, 1]
        axes[1].text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                    transform=axes[1].transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig('attention_weights_viz.png', dpi=150, bbox_inches='tight')
        plt.show()

    def visualize_training_analysis(self):
        """Visualize training analysis (simulated data)

        Note: This uses simulated data as actual training logs are not saved
        """
        # 日本語フォント再設定
        self._ensure_japanese_font()

        # Simulate training data (as actual logs are not available)
        epochs = np.arange(1, 51)
        train_loss = 0.5 * np.exp(-epochs/10) + 0.1 + np.random.normal(0, 0.01, 50)
        val_loss = 0.5 * np.exp(-epochs/12) + 0.12 + np.random.normal(0, 0.015, 50)

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Progress', 'Learning Rate Schedule',
                          'Gradient Flow', 'Parameter Distribution')
        )

        # 1. Training progress
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

        # 2. Learning rate schedule
        lr = 0.001 * np.exp(-epochs/20)
        fig.add_trace(
            go.Scatter(x=epochs, y=lr, name='Learning Rate',
                      line=dict(color='green', width=2)),
            row=1, col=2
        )

        # 3. Gradient flow (simulated)
        layer_names = ['Embedding', 'Attention', 'Hidden1', 'Hidden2', 'Hidden3', 'Output']
        gradient_means = np.random.exponential(0.01, len(layer_names))
        fig.add_trace(
            go.Bar(x=layer_names, y=gradient_means, name='Gradient Magnitude',
                  marker_color='purple'),
            row=2, col=1
        )

        # 4. Parameter distribution
        all_params = []
        for param in self.model.parameters():
            all_params.extend(param.detach().cpu().numpy().flatten())

        fig.add_trace(
            go.Histogram(x=all_params[:1000], name='Parameter Values',
                        marker_color='orange', nbinsx=50),
            row=2, col=2
        )

        fig.update_layout(
            height=700,
            showlegend=True,
            title_text="Training Analysis Dashboard"
        )
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Learning Rate", row=1, col=2)
        fig.update_xaxes(title_text="Layer", row=2, col=1)
        fig.update_yaxes(title_text="Gradient Magnitude", row=2, col=1)
        fig.update_xaxes(title_text="Parameter Value", row=2, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)

        fig.show()


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("Neural Network Comprehensive Visualization")
    print("="*60)

    # Create visualizer (adjust paths as needed)
    visualizer = NeuralNetworkVisualizer(
        model_type='attention',
        model_path='rent_prediction_model_attention.pth',
        data_path='tokyo_rent_data_v2.csv'
    )

    print("\n1. Network Architecture Visualization...")
    visualizer.visualize_architecture()

    print("\n2. Interactive Embedding Visualization...")
    visualizer.visualize_embeddings_interactive()

    print("\n3. Attention Weights Visualization...")
    visualizer.visualize_attention_weights()

    print("\n4. Training Process Analysis...")
    visualizer.visualize_training_analysis()

    print("\n✅ Visualization Complete!")
    print("Generated files:")
    print("  - network_architecture_viz.png")
    print("  - attention_weights_viz.png")
    print("  - Interactive plots (displayed in browser)")
