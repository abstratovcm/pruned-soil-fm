import os
import sys
import json
import shutil
import subprocess
import argparse
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flowMatching.data import get_traditional_features, split_and_scale_data
from flowMatching.postprocess import clean_samples
from paths import TABSYN_DIR, prepared_dataset, tabsyn_results

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

def prepare_synthetic(df_syn_raw, all_features, traditional_features, traditional_scaler, trad_indices):
    """Clean and scale a synthetic pool for adversarial evaluation."""
    X_syn_raw = df_syn_raw[all_features].values
    X_syn_cleaned = clean_samples(X_syn_raw, all_features, transform_params_path=None)
    X_syn_final = X_syn_cleaned.copy()
    if hasattr(traditional_scaler, 'feature_names_in_'):
        X_trad_df = pd.DataFrame(X_syn_final[:, trad_indices], columns=traditional_features)
        X_syn_final[:, trad_indices] = traditional_scaler.transform(X_trad_df)
    else:
        X_syn_final[:, trad_indices] = traditional_scaler.transform(X_syn_final[:, trad_indices])
    return X_syn_final

def run_tabsyn_experiments(start_iter, end_iter):
    tabsyn_dir = TABSYN_DIR

    cmd_env = os.environ.copy()
    cmd_env["PYTHONPATH"] = tabsyn_dir + os.pathsep + cmd_env.get("PYTHONPATH", "")

    datasets = [
        "chem_nospectral",
        "chem_spectral",
        "chem_phys_nospectral",
        "chem_phys_spectral"
    ]

    for mode in datasets:
        print(f"\n{'='*80}")
        print(f"Starting TabSyn Baseline for: {mode} (Iterations {start_iter} to {end_iter-1})")
        print(f"{'='*80}\n")

        dataset_csv = prepared_dataset(mode)

        if not os.path.exists(dataset_csv):
            print(f"Skipping {mode}: Data not found at {dataset_csv}.")
            continue

        df_full = pd.read_csv(dataset_csv)

        traditional_features = get_traditional_features(mode)

        raw_spectral_features = []
        if 'nospectral' not in mode:
            for col in df_full.columns:
                if col not in traditional_features and col not in ['Point_ID', 'geometry']:
                    try:
                        float(col)
                        raw_spectral_features.append(col)
                    except ValueError:
                        continue

        # Prepare incremental save file
        mode_results_dir = tabsyn_results(mode)
        os.makedirs(mode_results_dir, exist_ok=True)
        results_file = os.path.join(
            mode_results_dir, f"tabsyn_baseline_{start_iter}_to_{end_iter-1}.csv")

        # Write header if file doesn't exist
        if not os.path.exists(results_file):
            with open(results_file, 'w') as f:
                f.write("Iteration,Seed,TabSyn_Test_AUC,TabSyn_Test_AUC_10x,Achieved_Ratio_10x,"
                        "Time_Train_Sec,Time_Sample_Sec,Time_Sample_10x_Sec\n")

        for i in range(start_iter, end_iter):
            print(f"\n--- Iteration {i+1} (Seed {42+i}) ---")

            seed = 42 + i

            X_train_tabsyn, X_val_tabsyn, X_test_eval, all_features, traditional_scaler, _, _ = split_and_scale_data(
                df_full, traditional_features, raw_spectral_features, seed=seed
            )

            # TabSyn requires a categorical and a target column
            X_train_tabsyn['dummy_cat'] = 'A'
            X_train_tabsyn['dummy_target'] = np.random.randn(len(X_train_tabsyn))

            X_val_tabsyn['dummy_cat'] = 'A'
            X_val_tabsyn['dummy_target'] = np.random.randn(len(X_val_tabsyn))

            all_features_with_dummies = all_features + ['dummy_cat', 'dummy_target']

            # Save Train (70%) and Val (15%) to disk
            tabsyn_data_dir = os.path.join(tabsyn_dir, 'data', mode)
            os.makedirs(tabsyn_data_dir, exist_ok=True)

            train_csv_path = os.path.join(tabsyn_data_dir, f'{mode}_train.csv')
            val_csv_path = os.path.join(tabsyn_data_dir, f'{mode}_val.csv')

            X_train_tabsyn.to_csv(train_csv_path, index=False)
            X_val_tabsyn.to_csv(val_csv_path, index=False)

            info_dir = os.path.join(tabsyn_dir, 'data', 'Info')
            os.makedirs(info_dir, exist_ok=True)

            # Fully compliant schema forcing TabSyn to use our exact splits
            info_dict = {
                "name": mode,
                "task_type": "regression",
                "header": 0,
                "column_names": all_features_with_dummies,
                "num_col_idx": list(range(len(all_features))),
                "cat_col_idx": [len(all_features)],
                "target_col_idx": [len(all_features) + 1],
                "file_type": "csv",
                "data_path": f"data/{mode}/{mode}_train.csv",
                "test_path": f"data/{mode}/{mode}_val.csv"
            }
            with open(os.path.join(info_dir, f'{mode}.json'), 'w') as f:
                json.dump(info_dict, f, indent=4)

            # --- FIX TABSYN'S CACHING BUG ---
            cached_val_path = os.path.join(tabsyn_data_dir, 'test.data')
            if os.path.exists(cached_val_path):
                os.remove(cached_val_path)

            # --- TIMED BLOCK 1: TRAINING ---
            train_start_time = time.time()

            subprocess.run(["python", "process_dataset.py", "--dataname", mode], cwd=tabsyn_dir, env=cmd_env, check=True, capture_output=False)

            print("  -> Training VAE...")
            subprocess.run(["python", "main.py", "--dataname", mode, "--method", "vae", "--mode", "train"], cwd=tabsyn_dir, env=cmd_env, check=True, capture_output=False)

            print("  -> Training Diffusion...")
            subprocess.run(["python", "main.py", "--dataname", mode, "--method", "tabsyn", "--mode", "train"], cwd=tabsyn_dir, env=cmd_env, check=True, capture_output=False)

            train_end_time = time.time()
            time_train = train_end_time - train_start_time

            # --- TIMED BLOCK 2: INFERENCE (SAMPLING) ---
            sample_start_time = time.time()

            print("  -> Generating Samples...")
            save_path = os.path.join(tabsyn_data_dir, f'synthetic_{i}.csv')
            subprocess.run(["python", "main.py", "--dataname", mode, "--method", "tabsyn", "--mode", "sample", "--save_path", save_path], cwd=tabsyn_dir, env=cmd_env, check=True, capture_output=False)
            df_syn = pd.read_csv(save_path)

            sample_end_time = time.time()
            time_sample = sample_end_time - sample_start_time
            # -------------------------------------------

            n_test = len(X_test_eval)

            # --- TIMED BLOCK 3: INFERENCE (SAMPLING, 10x) ---
            sample_10x_start_time = time.time()

            n_target_10x = n_test * 10
            per_call_estimate = len(X_train_tabsyn)
            n_calls_needed = int(np.ceil(n_target_10x / per_call_estimate))
            print(f"  -> --num-samples not honored by TabSyn; need {n_target_10x} rows, "
                  f"each call yields ~{per_call_estimate}; running {n_calls_needed} call(s)...")

            syn_dfs_10x = []
            for call_idx in range(n_calls_needed):
                save_path_call = os.path.join(tabsyn_data_dir, f'synthetic_{i}_10x_call{call_idx}.csv')
                print(f"    - Sampling call {call_idx + 1}/{n_calls_needed}...")
                subprocess.run(
                    ["python", "main.py", "--dataname", mode, "--method", "tabsyn", "--mode", "sample",
                     "--save_path", save_path_call],
                    cwd=tabsyn_dir, env=cmd_env, check=True, capture_output=False
                )
                syn_dfs_10x.append(pd.read_csv(save_path_call))

            print(f"  -> Duplicate check: {syn_dfs_10x[0].equals(syn_dfs_10x[1])}")  # should be False

            df_syn_10x_pool = pd.concat(syn_dfs_10x, ignore_index=True)

            # Shuffle and slice to exactly n_target_10x, then save as the canonical 10x file
            df_syn_10x_pool = df_syn_10x_pool.sample(frac=1, random_state=seed).reset_index(drop=True)
            df_syn_10x = df_syn_10x_pool.iloc[:n_target_10x].reset_index(drop=True)

            save_path_10x = os.path.join(tabsyn_data_dir, f'synthetic_{i}_10x.csv')
            df_syn_10x.to_csv(save_path_10x, index=False)

            # Clean up intermediate per-call files now that they're consolidated
            for call_idx in range(n_calls_needed):
                save_path_call = os.path.join(tabsyn_data_dir, f'synthetic_{i}_10x_call{call_idx}.csv')
                if os.path.exists(save_path_call):
                    os.remove(save_path_call)

            time_sample_10x = time.time() - sample_10x_start_time
            # -------------------------------------------

            if len(df_syn_10x) < n_target_10x:
                print(f"  -> WARNING: pooled {n_calls_needed} calls but only got "
                      f"{len(df_syn_10x)} rows (needed {n_target_10x}).")
            else:
                print(f"  -> Confirmed: {len(df_syn_10x)} rows pooled from {n_calls_needed} calls "
                      f"({n_target_10x} needed).")

            trad_indices = [idx for idx, f in enumerate(all_features) if f in traditional_features]

            X_syn_final = prepare_synthetic(df_syn, all_features, traditional_features, traditional_scaler, trad_indices)
            X_syn_final_10x = prepare_synthetic(df_syn_10x, all_features, traditional_features, traditional_scaler, trad_indices)

            X_real_test = X_test_eval[all_features].values
            if hasattr(traditional_scaler, 'feature_names_in_'):
                X_test_trad_df = pd.DataFrame(X_real_test[:, trad_indices], columns=traditional_features)
                X_real_test[:, trad_indices] = traditional_scaler.transform(X_test_trad_df)
            else:
                X_real_test[:, trad_indices] = traditional_scaler.transform(X_real_test[:, trad_indices])

            auc = calculate_auc(X_real_test, X_syn_final, n_test, seed + 1000)
            auc_10x = calculate_auc(X_real_test, X_syn_final_10x, n_test, seed + 1000, ratio=10, verbose=True)
            achieved_ratio_10x = min(n_target_10x, len(X_syn_final_10x)) / n_test

            print(f"  -> Done! AUC (1x): {auc:.4f} | AUC (10x, achieved {achieved_ratio_10x:.2f}x): {auc_10x:.4f} "
                  f"| Train Time: {time_train/60:.1f} mins | Sample Time (1x): {time_sample:.1f}s "
                  f"| Sample Time (10x): {time_sample_10x:.1f}s")

            with open(results_file, 'a') as f:
                f.write(f"{i+1},{seed},{auc},{auc_10x},{achieved_ratio_10x:.4f},"
                         f"{time_train:.2f},{time_sample:.2f},{time_sample_10x:.2f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0, help="Starting iteration index (0-indexed)")
    parser.add_argument('--end', type=int, default=100, help="Ending iteration index (exclusive)")
    args = parser.parse_args()

    run_tabsyn_experiments(args.start, args.end)