import argparse
import torch
import torch.optim as optim
import os
import joblib
import numpy as np
import pandas as pd
import json
from flowMatching.data import prepare_data, get_traditional_features
from flowMatching.model import VelocityPredictor
from flowMatching.postprocess import clean_samples

def train(epochs=50, batch_size=128, lr=1e-3, device='cpu', output_dir='output_dir',
          weight_decay=0.01, dropout=0.1, hidden_dim=512, time_emb_dim=32,
          num_blocks=4, pct_start=0.1, random_state=42,
          df_train=None, df_val=None, df_test=None, all_features=None):

    print(f"--- Starting Training ---")

    import torch
    import numpy as np
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_loader, val_loader, _, scaler, feature_names, transforms_dict = prepare_data(
        batch_size=batch_size, df_train=df_train, df_val=df_val, df_test=df_test, all_features_ext=all_features
    )

    transform_path = os.path.join(output_dir, 'fm_transform_params.pkl')
    if isinstance(transforms_dict, dict):
        joblib.dump(transforms_dict, transform_path)
        print(f"Saved transform bundle to {transform_path}")
    else:
        transforms_dict.save(transform_path)

    sample_x = next(iter(train_loader))[0]
    output_dim = sample_x.shape[1]
    input_dim = len(feature_names)

    print(f"Input dim: {input_dim}, Output dim: {output_dim}")

    model = VelocityPredictor(input_dim, hidden_dim=hidden_dim, time_emb_dim=time_emb_dim,
                dropout=dropout, num_blocks=num_blocks).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=pct_start,
        div_factor=25.0,
        final_div_factor=1e4
    )

    best_metric = float('inf')
    history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch[0].to(device)

            optimizer.zero_grad()
            loss = model.compute_loss(x)
            loss.backward()
            optimizer.step()

            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_losses_batch = []
        val_clean_losses_batch = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                loss = model.compute_loss(x)
                val_losses_batch.append(loss.item())

        val_mean = np.mean(val_losses_batch)
        val_std = np.std(val_losses_batch)

        metric = val_mean + val_std

        print(f"Epoch {epoch+1}/{epochs} | " +
              f"Tr_Loss: {train_loss:.4f} | " +
              f"Val_Loss: {val_mean:.4f} | " +
              f"Val_Std: {val_std:.4f}")

        history.append({
            'Epoch': epoch + 1,
            'Train Loss': train_loss,
            'Val Loss Mean': val_mean,
            'Val Std': val_std
        })

        if metric < best_metric:
            best_metric = metric
            torch.save(model.state_dict(), os.path.join(output_dir, 'flow_matching.pth'))

    print(f"Training Complete. Best Clean Metric (Mean+Std): {best_metric:.4f}")

    log_df = pd.DataFrame(history)
    log_path = os.path.join(output_dir, 'training_log.csv')
    log_df.to_csv(log_path, index=False)
    print(f"Saved training log to {log_path}")

    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")

    return model, scaler, feature_names
