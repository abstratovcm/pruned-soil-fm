import os
import shutil
import sys
import time
import torch
import numpy as np
import pandas as pd
import joblib
import argparse
import json
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Make the repository root importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowMatching.train import train
from flowMatching.data import get_traditional_features
from flowMatching.postprocess import clean_samples
from paths import prepared_dataset

def generate_batch(model, scalers, feature_names, traditional_features, trad_indices, standard_scaler, total_samples, transform_params_path, device='cpu', seed=42):
    model.to(device)
    torch.manual_seed(seed)
    np.random.seed(seed)

    with torch.no_grad():
        # Unconditional sampling without 'c'
        samples = model.sample(total_samples, steps=50, device=device, seed=seed)
        X_gen = samples.cpu().numpy()

    X_inv = scalers.inverse_transform(X_gen)

    X_inv = clean_samples(X_inv, feature_names, transform_params_path=transform_params_path)

    X_syn_uncond = X_inv.copy()
    if hasattr(standard_scaler, 'feature_names_in_'):
        X_trad_df = pd.DataFrame(X_syn_uncond[:, trad_indices], columns=traditional_features)
        X_syn_uncond[:, trad_indices] = standard_scaler.transform(X_trad_df)
    else:
        X_syn_uncond[:, trad_indices] = standard_scaler.transform(X_syn_uncond[:, trad_indices])

    return X_syn_uncond

def calculate_auc(X_real, X_syn, n_real, seed, ratio=1, verbose=False):
    rng = np.random.RandomState(seed)

    n_target = n_real * ratio
    n_keep = min(n_target, len(X_syn))
    if n_keep < n_target and verbose:
        print(f"    [calculate_auc] WARNING: requested ratio={ratio} "
              f"({n_target} samples) but only {len(X_syn)} available. "
              f"Achieved ratio = {n_keep / n_real:.2f}x")
    idx_syn = rng.choice(len(X_syn), n_keep, replace=False)
    X_syn_used = X_syn[idx_syn]

    achieved_ratio = len(X_syn_used) / n_real
    if verbose:
        print(f"    [calculate_auc] real={n_real}, synthetic_used={len(X_syn_used)} "
              f"({achieved_ratio:.2f}x), requested_ratio={ratio}")

    X_comb = np.vstack([X_real, X_syn_used])
    y_comb = np.hstack([np.zeros(n_real), np.ones(len(X_syn_used))])

    n_pos = np.sum(y_comb == 1)
    n_neg = np.sum(y_comb == 0)
    spw = n_neg / n_pos

    clf = XGBClassifier(eval_metric='logloss', random_state=seed, n_jobs=-1,
                         scale_pos_weight=spw)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    y_prob = cross_val_predict(clf, X_comb, y_comb, cv=cv, method='predict_proba')[:, 1]
    return roc_auc_score(y_comb, y_prob)

def run_experiment(mode, N, output_dir):
    max_depth = 9
    thresholds = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]

    fm_epochs = 50
    fm_batch_size = 128
    fm_learning_rate = 1e-3
    fm_weight_decay = 0.01
    fm_dropout = 0.1
    fm_hidden_dim = 512
    fm_time_emb_dim = 32
    fm_num_blocks = 4
    fm_pct_start = 0.1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Simplified Validation Pipeline (N={N}) on device: {device}")

    dataset_csv = prepared_dataset(mode)

    if not os.path.exists(dataset_csv):
        print(f"Error: Dataset not found at {dataset_csv}. Run prepare_datasets.py first.")
        return

    print(f"Loading data from {dataset_csv}...")
    df_full = pd.read_csv(dataset_csv)

    traditional_features = get_traditional_features(mode)

    raw_spectral_features = []
    if 'nospectral' not in mode:
        for col in df_full.columns:
            if col not in traditional_features and col != 'Point_ID' and col != 'cluster' and col != 'geometry':
                try:
                    float(col)
                    raw_spectral_features.append(col)
                except ValueError:
                    continue

    val_results = { 'Pure': [] }
    test_results = { 'Pure': [] }
    test_results_10x = { 'Pure': [] }
    test_times = { 'Pure': [] }
    val_rejection_rates = { th: [] for th in thresholds }
    test_rejection_rates = { th: [] for th in thresholds }
    test_rejection_rates_10x = { th: [] for th in thresholds }
    for th in thresholds:
        val_results[f'Pruned_{th}'] = []
        test_results[f'Pruned_{th}'] = []
        test_results_10x[f'Pruned_{th}'] = []
        test_times[f'Pruned_{th}'] = []

    train_time_rows = []

    start_time = time.time()

    for i in range(N):
        print(f"\n{'='*50}\nIteration {i+1}/{N}\n{'='*50}")
        seed = 42 + i

        from flowMatching.data import split_and_scale_data
        X_train, X_val, X_test, all_features, standard_scaler, spectral_pca_fitted, spectral_scaler_fitted = split_and_scale_data(
            df_full, traditional_features, raw_spectral_features, seed=seed
        )

        X_real_train_df = X_train.copy()
        X_real_val_df = X_val.copy()
        X_real_test_df = X_test.copy()

        if hasattr(standard_scaler, 'feature_names_in_'):
            X_real_train_df[traditional_features] = standard_scaler.transform(X_real_train_df[traditional_features])
            X_real_val_df[traditional_features] = standard_scaler.transform(X_real_val_df[traditional_features])
            X_real_test_df[traditional_features] = standard_scaler.transform(X_real_test_df[traditional_features])
        else:
            X_real_train_df[traditional_features] = standard_scaler.transform(X_real_train_df[traditional_features].values)
            X_real_val_df[traditional_features] = standard_scaler.transform(X_real_val_df[traditional_features].values)
            X_real_test_df[traditional_features] = standard_scaler.transform(X_real_test_df[traditional_features].values)

        X_real_train = X_real_train_df[all_features].values
        X_real_val = X_real_val_df[all_features].values
        X_real_test = X_real_test_df[all_features].values

        n_train = len(X_real_train)
        n_val = len(X_real_val)
        n_test = len(X_real_test)

        trad_indices = [i for i, f in enumerate(all_features) if f in traditional_features]

        temp_dir = f"temp_output_iter_{i}"
        os.makedirs(temp_dir, exist_ok=True)

        train_start_time = time.time()

        print(f"Training VelocityPredictor model for iteration {i+1}...")

        model, scaler, feature_names = train(
            epochs=fm_epochs, batch_size=fm_batch_size, lr=fm_learning_rate,
            device=device, output_dir=temp_dir,
            weight_decay=fm_weight_decay, dropout=fm_dropout, hidden_dim=fm_hidden_dim,
            time_emb_dim=fm_time_emb_dim, num_blocks=fm_num_blocks,
            pct_start=fm_pct_start, random_state=seed,
            df_train=X_train, df_val=X_val, df_test=X_test,
            all_features=all_features
        )

        transform_params_path = None
        files = [f for f in os.listdir(temp_dir) if f.startswith('fm_transform_params') and f.endswith('.pkl')]
        if files:
            transform_params_path = os.path.join(temp_dir, files[0])

        print(f"Running Phase 1 (Validation)...")
        print(f" -> Generating initial pure batch to train the Pruning Decision Tree...")

        X_pure_train = generate_batch(model, scaler, feature_names, traditional_features, trad_indices, standard_scaler, n_train, transform_params_path, device=device, seed=seed)

        idx_pure = np.random.choice(len(X_pure_train), min(n_train, len(X_pure_train)), replace=False)
        X_pure_sampled = X_pure_train[idx_pure]

        X_prune_train = np.vstack([X_real_train[:len(idx_pure)], X_pure_sampled])
        y_prune_train = np.hstack([np.zeros(len(idx_pure)), np.ones(len(idx_pure))])

        dt_prune = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
        dt_prune.fit(X_prune_train, y_prune_train)

        leaf_indices_prune = dt_prune.apply(X_prune_train)
        unique_leaves_prune = np.unique(leaf_indices_prune)
        bad_leaves_dict = {th: {} for th in thresholds}

        for leaf in unique_leaves_prune:
            leaf_mask = (leaf_indices_prune == leaf)
            y_true_leaf = y_prune_train[leaf_mask]
            total_samples_leaf = len(y_true_leaf)
            if total_samples_leaf == 0: continue
            synth_ratio = np.sum(y_true_leaf == 1) / total_samples_leaf

            for th in thresholds:
                if synth_ratio >= th:
                    drop_prob = max(0.0, (2.0 * synth_ratio - 1.0) / synth_ratio) if synth_ratio > 0 else 0.0
                    bad_leaves_dict[th][leaf] = drop_prob

        print(f" -> Generating and filtering continuous batches for Validation thresholds...")
        accepted_data = {th: [] for th in thresholds}
        accepted_counts = {th: 0 for th in thresholds}
        total_generated_for_th = {th: 0 for th in thresholds}
        pure_accumulated = []
        pure_count = 0
        batch_counter = 0

        val_generation_seed = seed + 100
        drop_rng = np.random.RandomState(val_generation_seed)
        batch_gen_size = n_val * 3

        while True:
            batch_counter += 1
            print(f"    - Generating Validation Batch {batch_counter}...", end=" ", flush=True)
            val_generation_seed += 1
            X_batch = generate_batch(model, scaler, feature_names, traditional_features, trad_indices, standard_scaler, batch_gen_size, transform_params_path, device=device, seed=val_generation_seed)
            batch_leaves = dt_prune.apply(X_batch)
            batch_size = len(X_batch)
            print("Done. Filtering...")

            if pure_count < n_val:
                pure_accumulated.append(X_batch)
                pure_count += batch_size

            u = drop_rng.uniform(0, 1, batch_size)

            for th in thresholds:
                if accepted_counts[th] < n_val:
                    total_generated_for_th[th] += batch_size
                    drop_probs = np.array([bad_leaves_dict[th].get(leaf, 0.0) for leaf in batch_leaves])
                    valid_mask = u >= drop_probs
                    valid_samples = X_batch[valid_mask]
                    accepted_data[th].append(valid_samples)
                    accepted_counts[th] += len(valid_samples)

            if pure_count >= n_val and all(c >= n_val for c in accepted_counts.values()):
                break

        X_pure_stacked = np.vstack(pure_accumulated)
        np.random.seed(seed)
        np.random.shuffle(X_pure_stacked)
        X_pure_final = X_pure_stacked[:n_val]
        val_results['Pure'].append(calculate_auc(X_real_val, X_pure_final, n_val, seed))

        train_end_time = time.time()
        time_train = train_end_time - train_start_time

        seed_test = seed + 1000

        print(f" -> Generating Test Sets (Isolating generation times)...")

        # --- A. PURE TEST GENERATION ---
        start_time_pure = time.time()
        print(f"    - Generating Pure Test Set...", end=" ", flush=True)

        # We can generate n_test directly since there is no rejection filtering
        X_pure_final_test = generate_batch(model, scaler, feature_names, traditional_features, trad_indices, standard_scaler, n_test, transform_params_path, device=device, seed=seed_test)

        # Shuffle before slicing
        np.random.seed(seed_test)
        np.random.shuffle(X_pure_final_test)

        time_pure = time.time() - start_time_pure
        test_times['Pure'].append(time_pure)
        print(f"Done in {time_pure:.2f}s.")

        test_results['Pure'].append(calculate_auc(X_real_test, X_pure_final_test, n_test, seed_test))

        # --- B. PRUNED TEST GENERATION (PER THRESHOLD) ---
        final_pruned_test_samples = {}
        total_generated_for_th_test = {th: 0 for th in thresholds}
        accepted_counts_test = {th: 0 for th in thresholds}

        batch_test_gen_size = n_test * 3

        for th in thresholds:
            start_time_th = time.time()
            print(f"    - Generating Pruned Test Set (Threshold {th})...", end=" ", flush=True)

            accepted_th_list = []
            th_seed = seed_test + int(th * 100) # Unique seed offset per threshold to avoid overlap
            drop_rng_test = np.random.RandomState(th_seed)

            while accepted_counts_test[th] < n_test:
                th_seed += 1
                X_batch = generate_batch(model, scaler, feature_names, traditional_features, trad_indices, standard_scaler, batch_test_gen_size, transform_params_path, device=device, seed=th_seed)
                batch_leaves = dt_prune.apply(X_batch)
                batch_size = len(X_batch)

                total_generated_for_th_test[th] += batch_size
                u_test = drop_rng_test.uniform(0, 1, batch_size)

                drop_probs = np.array([bad_leaves_dict[th].get(leaf, 0.0) for leaf in batch_leaves])
                valid_mask = u_test >= drop_probs
                valid_samples = X_batch[valid_mask]

                accepted_th_list.append(valid_samples)
                accepted_counts_test[th] += len(valid_samples)

            # Stack, shuffle, and slice the exact amount needed
            X_pruned_test_stacked = np.vstack(accepted_th_list)
            np.random.seed(seed_test)
            np.random.shuffle(X_pruned_test_stacked)
            final_pruned_test_samples[th] = X_pruned_test_stacked[:n_test]

            time_th = time.time() - start_time_th
            test_times[f'Pruned_{th}'].append(time_th)
            print(f"Done in {time_th:.2f}s.")

        iter_save_dir = os.path.join(output_dir, f"saved_samples_iter_{i}")
        os.makedirs(iter_save_dir, exist_ok=True)

        metadata = {
            "iteration": i,
            "seed": seed,
            "seed_val": seed + 100,
            "seed_test": seed_test,
            "n_val": n_val,
            "n_test": n_test,
            "time_train_sec": time_train
        }
        with open(os.path.join(iter_save_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
        train_time_rows.append(metadata)

        iter_scalers_dict = {'traditional': standard_scaler}
        if spectral_scaler_fitted is not None:
            iter_scalers_dict['spectral'] = spectral_scaler_fitted
        if spectral_pca_fitted is not None:
            iter_scalers_dict['pca'] = spectral_pca_fitted

        joblib.dump(iter_scalers_dict, os.path.join(iter_save_dir, 'fm_scaler_iter.pkl'))

        if transform_params_path and os.path.exists(transform_params_path):
            shutil.copy2(transform_params_path, os.path.join(iter_save_dir, os.path.basename(transform_params_path)))

        df_pure_val = pd.DataFrame(X_pure_final, columns=feature_names)
        df_pure_val.to_parquet(os.path.join(iter_save_dir, 'pure_val.parquet'), index=False)

        df_pure_test = pd.DataFrame(X_pure_final_test, columns=feature_names)
        df_pure_test.to_parquet(os.path.join(iter_save_dir, 'pure_test.parquet'), index=False)

        for th in thresholds:
            X_pruned_stacked = np.vstack(accepted_data[th])
            np.random.seed(seed)
            np.random.shuffle(X_pruned_stacked)
            X_pruned_final = X_pruned_stacked[:n_val]
            val_results[f'Pruned_{th}'].append(calculate_auc(X_real_val, X_pruned_final, n_val, seed))

            rej_rate = 1.0 - (accepted_counts[th] / total_generated_for_th[th]) if total_generated_for_th[th] > 0 else 0.0
            val_rejection_rates[th].append(rej_rate)

            df_pruned_val = pd.DataFrame(X_pruned_final, columns=feature_names)
            df_pruned_val.to_parquet(os.path.join(iter_save_dir, f'pruned_val_{th}.parquet'), index=False)

            X_pruned_final_test = final_pruned_test_samples[th]
            test_results[f'Pruned_{th}'].append(calculate_auc(X_real_test, X_pruned_final_test, n_test, seed_test))

            rej_rate_test = 1.0 - (accepted_counts_test[th] / total_generated_for_th_test[th]) if total_generated_for_th_test[th] > 0 else 0.0
            test_rejection_rates[th].append(rej_rate_test)

            df_pruned_test = pd.DataFrame(X_pruned_final_test, columns=feature_names)
            df_pruned_test.to_parquet(os.path.join(iter_save_dir, f'pruned_test_{th}.parquet'), index=False)

        print(f" -> Generating 10X Test Sets...")
        n_test_10x = n_test * 10
        seed_test_10x = seed_test + 2000

        # --- C. PURE 10X TEST GENERATION ---
        start_time_pure_10x = time.time()
        print(f"    - Generating Pure 10X Test Set...", end=" ", flush=True)

        X_pure_final_test_10x = generate_batch(model, scaler, feature_names, traditional_features, trad_indices, standard_scaler, n_test_10x, transform_params_path, device=device, seed=seed_test_10x)

        np.random.seed(seed_test_10x)
        np.random.shuffle(X_pure_final_test_10x)

        df_pure_test_10x = pd.DataFrame(X_pure_final_test_10x, columns=feature_names)
        df_pure_test_10x.to_parquet(os.path.join(iter_save_dir, 'pure_test_10x.parquet'), index=False)

        test_results_10x['Pure'].append(calculate_auc(X_real_test, X_pure_final_test_10x, n_test, seed_test_10x, ratio=10))
        print(f"Done in {time.time() - start_time_pure_10x:.2f}s.")

        # --- D. PRUNED 10X TEST GENERATION ---
        batch_test_gen_size_10x = n_test_10x * 3

        for th in thresholds:
            start_time_th_10x = time.time()
            print(f"    - Generating Pruned 10X Test Set (Threshold {th})...", end=" ", flush=True)

            accepted_th_list_10x = []
            accepted_counts_test_10x = 0
            total_generated_for_th_test_10x = 0
            th_seed_10x = seed_test_10x + int(th * 100)
            drop_rng_test_10x = np.random.RandomState(th_seed_10x)

            while accepted_counts_test_10x < n_test_10x:
                th_seed_10x += 1
                X_batch_10x = generate_batch(model, scaler, feature_names, traditional_features, trad_indices, standard_scaler, batch_test_gen_size_10x, transform_params_path, device=device, seed=th_seed_10x)
                batch_leaves_10x = dt_prune.apply(X_batch_10x)
                batch_size_10x = len(X_batch_10x)

                total_generated_for_th_test_10x += batch_size_10x

                u_test_10x = drop_rng_test_10x.uniform(0, 1, batch_size_10x)
                drop_probs_10x = np.array([bad_leaves_dict[th].get(leaf, 0.0) for leaf in batch_leaves_10x])
                valid_mask_10x = u_test_10x >= drop_probs_10x
                valid_samples_10x = X_batch_10x[valid_mask_10x]

                accepted_th_list_10x.append(valid_samples_10x)
                accepted_counts_test_10x += len(valid_samples_10x)

            X_pruned_test_stacked_10x = np.vstack(accepted_th_list_10x)
            np.random.seed(seed_test_10x)
            np.random.shuffle(X_pruned_test_stacked_10x)
            X_pruned_final_test_10x = X_pruned_test_stacked_10x[:n_test_10x]

            rej_rate_test_10x = 1.0 - (accepted_counts_test_10x / total_generated_for_th_test_10x) \
                if total_generated_for_th_test_10x > 0 else 0.0
            test_rejection_rates_10x[th].append(rej_rate_test_10x)

            df_pruned_test_10x = pd.DataFrame(X_pruned_final_test_10x, columns=feature_names)
            df_pruned_test_10x.to_parquet(os.path.join(iter_save_dir, f'pruned_test_10x_{th}.parquet'), index=False)

            test_results_10x[f'Pruned_{th}'].append(calculate_auc(X_real_test, X_pruned_final_test_10x, n_test, seed_test_10x, ratio=10))
            print(f"Done in {time.time() - start_time_th_10x:.2f}s.")

        print(f"Iteration {i+1} Pure Val AUC: {val_results['Pure'][-1]:.4f} | Pure Test AUC: {test_results['Pure'][-1]:.4f}")

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"\nPipeline completed in {(time.time() - start_time)/60:.2f} minutes.")

    os.makedirs(output_dir, exist_ok=True)
    # Per-iteration seeds and timings for the whole run.
    pd.DataFrame(train_time_rows).to_csv(os.path.join(output_dir, "train_times.csv"), index=False)

    pd.DataFrame(val_results).to_csv(os.path.join(output_dir, "phase1_val_results.csv"), index=False)
    pd.DataFrame(val_rejection_rates).to_csv(os.path.join(output_dir, "phase1_rejection_rates.csv"), index=False)
    pd.DataFrame(test_results).to_csv(os.path.join(output_dir, "phase2_test_results.csv"), index=False)
    pd.DataFrame(test_rejection_rates).to_csv(os.path.join(output_dir, "phase2_test_rejection_rates.csv"), index=False)
    pd.DataFrame(test_rejection_rates_10x).to_csv(os.path.join(output_dir, "phase2_test_rejection_rates_10x.csv"), index=False)
    pd.DataFrame(test_times).to_csv(os.path.join(output_dir, "phase2_test_generation_times.csv"), index=False)
    pd.DataFrame(test_results_10x).to_csv(os.path.join(output_dir, "phase2_test_results_10x.csv"), index=False)
    print("Raw results saved to CSVs.")

    print("\n" + "="*50)
    print("FINAL SCIENTIFIC EVALUATION")
    print("="*50)

    df_val = pd.DataFrame(val_results)
    mean_aucs_val = df_val.mean()
    # Sample std is undefined for a single iteration.
    std_aucs_val = df_val.std().fillna(0.0) if len(df_val) > 1 else 0.0
    upper_bound_val = mean_aucs_val + std_aucs_val

    best_strategy = upper_bound_val.drop('Pure').idxmin()
    best_threshold = float(best_strategy.replace('Pruned_', ''))

    print(f"Phase 1 (Validation) Selection:")
    print(f"-> Selected Global Best Threshold: {best_threshold}")
    print(f"   (Minimized Mean + Std Validation AUC: {upper_bound_val[best_strategy]:.4f})")

    df_test = pd.DataFrame(test_results)

    mean_pure_test = df_test['Pure'].mean()
    std_pure_test = df_test['Pure'].std()

    mean_pruned_test = df_test[best_strategy].mean()
    std_pruned_test = df_test[best_strategy].std()

    print("\nPhase 2 (Definitive Test) Results:")
    print(f"Pure Synthetic vs Test Set AUC     : {mean_pure_test:.4f} ± {std_pure_test:.4f}")
    print(f"Optimal Pruned vs Test Set AUC     : {mean_pruned_test:.4f} ± {std_pruned_test:.4f}")

    print("\n==================================================")
    diff = mean_pure_test - mean_pruned_test
    if diff > 0:
        print(f">>> CONCLUSION: The classifier-based rejection strategy (Threshold: {best_threshold})")
        print(f"    STATISTICALLY IMPROVED generative fidelity.")
        print(f"    Adversarial Distinguishability (AUC) dropped by {diff:.4f} on the isolated Test Set. <<<")
    else:
        print(f">>> CONCLUSION: The classifier-based rejection strategy (Threshold: {best_threshold})")
        print(f"    DID NOT improve generative fidelity.")
        print(f"    Adversarial Distinguishability (AUC) worsened by {abs(diff):.4f} on the isolated Test Set. <<<")
    print("==================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, help="Dataset mode")
    parser.add_argument('--N', type=int, default=10, help="Number of Monte Carlo iterations")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the output files")
    args = parser.parse_args()

    run_experiment(args.mode, args.N, args.output_dir)
