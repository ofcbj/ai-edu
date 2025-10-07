"""
Training utilities for Tokyo Rent Prediction Model
"""

import torch
import torch.nn as nn
import torch.optim as optim


def train_model(model, train_loader, val_loader, device, num_epochs=100,
                learning_rate=0.001, model_save_path='best_rent_model.pth'):
    """
    Train the rent prediction model with validation

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: torch.device to use for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        model_save_path: Path to save the best model

    Returns:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
    """
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

            # Feature separation
            ward_idx        = batch_features[:, 0].long()
            structure_idx   = batch_features[:, 1].long()
            type_idx        = batch_features[:, 2].long()
            numeric_features= batch_features[:, 3:6]
            ward_avg_price  = batch_features[:, 6]

            optimizer.zero_grad()
            outputs = model(ward_idx, structure_idx, type_idx, numeric_features, ward_avg_price)
            loss = criterion(outputs.squeeze(), batch_targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)

                ward_idx        = batch_features[:, 0].long()
                structure_idx   = batch_features[:, 1].long()
                type_idx        = batch_features[:, 2].long()
                numeric_features= batch_features[:, 3:6]
                ward_avg_price  = batch_features[:, 6]

                outputs = model(ward_idx, structure_idx, type_idx, numeric_features, ward_avg_price)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = criterion(outputs.squeeze(), batch_targets)
                val_loss += loss.item()

        avg_train_loss  = train_loss / len(train_loader)
        avg_val_loss    = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    return train_losses, val_losses


def evaluate_model(model, test_loader, device, model_name="Model"):
    """
    Evaluate model on test data

    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: torch.device
        model_name: Name for display

    Returns:
        predictions: Array of predictions
        actuals: Array of actual values
        mae: Mean Absolute Error
        rmse: Root Mean Squared Error
        r2: R-squared score
    """
    import numpy as np

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
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            predictions.extend(outputs.cpu().numpy().flatten() * 10000)
            actuals.extend(batch_targets.numpy() * 10000)

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    r2 = 1 - (np.sum((actuals - predictions) ** 2) / np.sum((actuals - np.mean(actuals)) ** 2))

    print(f"\n{model_name} テストデータ性能:")
    print(f"  MAE: ¥{mae:,.0f}")
    print(f"  RMSE: ¥{rmse:,.0f}")
    print(f"  R² Score: {r2:.4f}")

    return predictions, actuals, mae, rmse, r2
