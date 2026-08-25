import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Make the repository root importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from flowMatching.data import get_traditional_features, split_and_scale_data
    from flowMatching.postprocess import clean_samples
    from paths import fm_results, downstream_results, tabsyn_data, prepared_dataset
except ModuleNotFoundError:
    print("Error: Could not import flowMatching. Ensure the repository layout is intact.")
    sys.exit(1)

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Fits the model and returns R2, RMSE, and MAE."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    if y_pred.ndim > 1:
        y_pred = y_pred.squeeze()
        
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    return r2, rmse, mae

def run_downstream_utility(mode, N, input_dir, output_dir, target_col='OC', use_rf=False):
    print(f"Starting Downstream Utility Evaluation (Target: {target_col})")
    print(f"Loading FM generated data from: {input_dir}")
    print(f"Saving results to: {output_dir}")
    
    # 1. Dynamically determine the Optimal Threshold
    val_results_path = os.path.join(input_dir, "phase1_val_results.csv")
    if not os.path.exists(val_results_path):
        print(f"Error: Could not find {val_results_path} to determine the optimal threshold.")
        return

    val_results = pd.read_csv(val_results_path)
    mean_val = val_results.mean()
    std_val = val_results.std()
    ucb_val = mean_val + std_val
    
    best_strategy = ucb_val.drop('Pure').idxmin()
    best_threshold = float(best_strategy.replace('Pruned_', ''))
    
    print(f"-> Auto-detected Optimal Strategy: {best_strategy} (Threshold: {best_threshold})")

    # 2. Prepared dataset for this variant
    dataset_csv = prepared_dataset(mode)

    if not os.path.exists(dataset_csv):
        print(f"Error: Original dataset not found at {dataset_csv}.")
        return

    df_full = pd.read_csv(dataset_csv)
    traditional_features = get_traditional_features(mode)
    
    raw_spectral_features = []
    if 'nospectral' not in mode:
        for col in df_full.columns:
            if col not in traditional_features and col not in ['Point_ID', 'cluster', 'geometry']:
                try:
                    float(col)
                    raw_spectral_features.append(col)
                except ValueError:
                    continue

    # Conditions tracked across runs
    conditions = [
        'TRTR', 
        'TabSyn_1xTrain',
        'TSTR_Pure_1xTrain', 
        f'TSTR_Pruned_{best_threshold}_1xTrain',
        'TabSyn_2xTrain',
        'TSTR_Pure_2xTrain', 
        f'TSTR_Pruned_{best_threshold}_2xTrain',
        'Augmented_Real_Pruned_1xTrain',
        'Augmented_Real_Pruned_2xTrain' 
    ]
    
    results = {}

    for i in range(N):
        iter_dir = os.path.join(input_dir, f"saved_samples_iter_{i}")
        
        if not os.path.exists(iter_dir):
            print(f"\n  -> Missing FM directory {iter_dir}, stopping iteration loop.")
            break

        pure_10x_path = os.path.join(iter_dir, 'pure_test_10x.parquet')
        pruned_10x_path = os.path.join(iter_dir, f'pruned_test_10x_{best_threshold}.parquet')
        if not os.path.exists(pure_10x_path) or not os.path.exists(pruned_10x_path):
            print(f"\n  -> Missing 10x sample files in {iter_dir}. Run eval/run_flow_pruning.py first, stopping iteration loop.")
            break

        with open(os.path.join(iter_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
            
        seed = metadata['seed']
        
        # 3. Reconstruct exact Real splits (with PCs)
        X_real_train_df, _, X_real_test_df, all_features, standard_scaler, _, _ = split_and_scale_data(
            df_full, traditional_features, raw_spectral_features, seed=seed
        )
        
        if hasattr(standard_scaler, 'feature_names_in_'):
            X_real_train_df[traditional_features] = standard_scaler.transform(X_real_train_df[traditional_features])
            X_real_test_df[traditional_features] = standard_scaler.transform(X_real_test_df[traditional_features])
        else:
            X_real_train_df[traditional_features] = standard_scaler.transform(X_real_train_df[traditional_features].values)
            X_real_test_df[traditional_features] = standard_scaler.transform(X_real_test_df[traditional_features].values)

        # 4. Isolate Target (y) and Predictors (X)
        if target_col not in all_features:
            print(f"  -> Critical Error: Target '{target_col}' not found in features.")
            return

        target_idx = all_features.index(target_col)
        predictor_indices = [idx for idx in range(len(all_features)) if idx != target_idx]

        X_real_train_arr = X_real_train_df[all_features].values
        X_real_test_arr = X_real_test_df[all_features].values

        X_real_train = X_real_train_arr[:, predictor_indices]
        y_real_train = X_real_train_arr[:, target_idx]
        X_real_test = X_real_test_arr[:, predictor_indices]
        y_real_test = X_real_test_arr[:, target_idx]

        # 5. DYNAMIC MODEL INITIALIZATION
        n_features = X_real_train.shape[1]
        
        models = {'PLSR': PLSRegression(n_components=n_features)}
        if use_rf:
            models['RF'] = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

        if not results:
            results = {m: {cond: {'R2': [], 'RMSE': [], 'MAE': []} for cond in conditions} for m in models.keys()}

        # 6. Load FM Synthetic Datasets
        n_train_size = len(X_real_train)
        n_2x_size = n_train_size * 2

        df_pure_10x = pd.read_parquet(pure_10x_path)
        df_pruned_10x = pd.read_parquet(pruned_10x_path)

        # Safety Check Limits
        len_pure = len(df_pure_10x)
        len_pruned = len(df_pruned_10x)
        
        if len_pure < n_train_size or len_pruned < n_train_size:
            print(f"\n  [!] WARNING (Iter {i}): Not enough synthetic data for 1xTrain!")
            print(f"      Need {n_train_size} rows, but have Pure: {len_pure}, Pruned: {len_pruned}. Capping.")
        
        if len_pure < n_2x_size or len_pruned < n_2x_size:
            print(f"\n  [!] WARNING (Iter {i}): Not enough synthetic data for 2xTrain!")
            print(f"      Need {n_2x_size} rows, but have Pure: {len_pure}, Pruned: {len_pruned}. Capping.")

        take_pure_1x = min(n_train_size, len_pure)
        take_pure_2x = min(n_2x_size, len_pure)
        take_pruned_1x = min(n_train_size, len_pruned)
        take_pruned_2x = min(n_2x_size, len_pruned)

        rng = np.random.default_rng(seed)
        idx_pure = rng.permutation(len_pure)
        idx_pruned = rng.permutation(len_pruned)

        syn_pure_1x_arr = df_pure_10x.iloc[idx_pure[:take_pure_1x]][all_features].values
        syn_pure_2x_arr = df_pure_10x.iloc[idx_pure[:take_pure_2x]][all_features].values
        syn_pruned_1x_arr = df_pruned_10x.iloc[idx_pruned[:take_pruned_1x]][all_features].values
        syn_pruned_2x_arr = df_pruned_10x.iloc[idx_pruned[:take_pruned_2x]][all_features].values

        train_sets = {
            'TRTR': (X_real_train, y_real_train),
            'TSTR_Pure_1xTrain': (syn_pure_1x_arr[:, predictor_indices], syn_pure_1x_arr[:, target_idx]),
            f'TSTR_Pruned_{best_threshold}_1xTrain': (syn_pruned_1x_arr[:, predictor_indices], syn_pruned_1x_arr[:, target_idx]),
            'TSTR_Pure_2xTrain': (syn_pure_2x_arr[:, predictor_indices], syn_pure_2x_arr[:, target_idx]),
            f'TSTR_Pruned_{best_threshold}_2xTrain': (syn_pruned_2x_arr[:, predictor_indices], syn_pruned_2x_arr[:, target_idx])
        }

        # 7. Process TabSyn Baseline (Cleaning & Scaling natively)
        tabsyn_path_10x = os.path.join(tabsyn_data(mode), f'synthetic_{i}_10x.csv')
        if os.path.exists(tabsyn_path_10x):
            df_tabsyn_10x = pd.read_csv(tabsyn_path_10x)
            
            # Strip Dummies
            for col in ['dummy_cat', 'dummy_target']:
                if col in df_tabsyn_10x.columns:
                    df_tabsyn_10x = df_tabsyn_10x.drop(columns=[col])

            # Clean and Scale (Mimicking prepare_synthetic)
            X_tabsyn_raw = df_tabsyn_10x[all_features].values
            X_tabsyn_cleaned = clean_samples(X_tabsyn_raw, all_features, transform_params_path=None)
            
            trad_indices = [idx for idx, f in enumerate(all_features) if f in traditional_features]
            if hasattr(standard_scaler, 'feature_names_in_'):
                X_trad_df = pd.DataFrame(X_tabsyn_cleaned[:, trad_indices], columns=traditional_features)
                X_tabsyn_cleaned[:, trad_indices] = standard_scaler.transform(X_trad_df)
            else:
                X_tabsyn_cleaned[:, trad_indices] = standard_scaler.transform(X_tabsyn_cleaned[:, trad_indices])

            # Subsample to the same sizes as the Flow Matching conditions
            len_tabsyn = len(X_tabsyn_cleaned)

            if len_tabsyn < n_train_size:
                print(f"\n  [!] WARNING (Iter {i}): Not enough synthetic data for 1xTabSyn!")
                print(f"      Need {n_train_size} rows, but have: {len_tabsyn}. Capping.")
        
            if len_tabsyn < n_2x_size:
                print(f"\n  [!] WARNING (Iter {i}): Not enough synthetic data for 2xTabSyn!")
                print(f"      Need {n_2x_size} rows, but have: {len_tabsyn}. Capping.")

            take_tabsyn_1x = min(n_train_size, len_tabsyn)
            take_tabsyn_2x = min(n_2x_size, len_tabsyn)
            
            idx_tabsyn = rng.permutation(len_tabsyn)
            syn_tabsyn_1x_arr = X_tabsyn_cleaned[idx_tabsyn[:take_tabsyn_1x]]
            syn_tabsyn_2x_arr = X_tabsyn_cleaned[idx_tabsyn[:take_tabsyn_2x]]
            
            train_sets['TabSyn_1xTrain'] = (syn_tabsyn_1x_arr[:, predictor_indices], syn_tabsyn_1x_arr[:, target_idx])
            train_sets['TabSyn_2xTrain'] = (syn_tabsyn_2x_arr[:, predictor_indices], syn_tabsyn_2x_arr[:, target_idx])
        else:
            print(f"  [!] WARNING (Iter {i}): TabSyn 10x file missing at {tabsyn_path_10x}. Skipping TabSyn for this fold.")

        # 8. Augmented Data
        aug_X_1x = np.vstack([X_real_train, syn_pruned_1x_arr[:, predictor_indices]])
        aug_y_1x = np.concatenate([y_real_train, syn_pruned_1x_arr[:, target_idx]])
        train_sets['Augmented_Real_Pruned_1xTrain'] = (aug_X_1x, aug_y_1x)

        aug_X_2x = np.vstack([X_real_train, syn_pruned_2x_arr[:, predictor_indices]])
        aug_y_2x = np.concatenate([y_real_train, syn_pruned_2x_arr[:, target_idx]])
        train_sets['Augmented_Real_Pruned_2xTrain'] = (aug_X_2x, aug_y_2x)

        # 9. Evaluate models
        for model_name, model in models.items():
            for cond_name, (X_tr, y_tr) in train_sets.items():
                r2, rmse, mae = evaluate_model(model, X_tr, y_tr, X_real_test, y_real_test)
                results[model_name][cond_name]['R2'].append(r2)
                results[model_name][cond_name]['RMSE'].append(rmse)
                results[model_name][cond_name]['MAE'].append(mae)
                
        if (i+1) % 10 == 0 or N < 10:
            print(f"  -> Completed iteration {i+1}/{N}")

    if not results:
        print("\nNo iterations were evaluated; nothing to summarise.")
        return

    print("\n" + "="*60)
    print("DOWNSTREAM UTILITY RESULTS (Mean ± Std across iterations)")
    print("="*60)

    os.makedirs(output_dir, exist_ok=True)

    # 10. Compute stats and Wilcoxon tests
    for model_name in results.keys():
        print(f"\n--- Model: {model_name} ---")
        trtr_r2 = results[model_name]['TRTR']['R2']
        trtr_rmse = results[model_name]['TRTR']['RMSE']
        
        summary_rows = []
        for cond in conditions:
            r2_mean = np.mean(results[model_name][cond]['R2'])
            r2_std = np.std(results[model_name][cond]['R2'])
            rmse_mean = np.mean(results[model_name][cond]['RMSE'])
            rmse_std = np.std(results[model_name][cond]['RMSE'])
            
            pval_r2 = "N/A"
            pval_rmse = "N/A"
            if cond != 'TRTR':
                try:
                    _, pval_r2 = wilcoxon(trtr_r2, results[model_name][cond]['R2'])
                    _, pval_rmse = wilcoxon(trtr_rmse, results[model_name][cond]['RMSE'])
                    pval_r2 = f"{pval_r2:.4f}"
                    pval_rmse = f"{pval_rmse:.4f}"
                except ValueError: 
                    pval_r2, pval_rmse = "1.0000", "1.0000"

            print(f"{cond:<35} | R2: {r2_mean:6.3f} ± {r2_std:.3f} (p={pval_r2}) | RMSE: {rmse_mean:6.3f} ± {rmse_std:.3f} (p={pval_rmse})")
            
            summary_rows.append({
                'Condition': cond,
                'R2_Mean': r2_mean, 'R2_Std': r2_std, 'R2_p_vs_TRTR': pval_r2,
                'RMSE_Mean': rmse_mean, 'RMSE_Std': rmse_std, 'RMSE_p_vs_TRTR': pval_rmse,
                'MAE_Mean': np.mean(results[model_name][cond]['MAE']),
                'MAE_Std': np.std(results[model_name][cond]['MAE'])
            })
            
        pd.DataFrame(summary_rows).to_csv(os.path.join(output_dir, f"downstream_metrics_{model_name}.csv"), index=False)
    
    print(f"\nSaved summary metrics to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate downstream regression on real vs synthetic data.")
    parser.add_argument('--mode', type=str, required=True, help="Dataset mode (e.g., chem_nospectral, chem_spectral)")
    parser.add_argument('--N', type=int, default=100, help="Number of Monte Carlo iterations (default: 100)")
    parser.add_argument('--target', type=str, default='OC', help="Downstream target to predict (default: OC)")
    parser.add_argument('--use_rf', action='store_true', help="Also evaluate using Random Forest")
    args = parser.parse_args()

    dynamic_input_dir = fm_results(args.mode)
    dynamic_output_dir = downstream_results(args.mode)

    run_downstream_utility(
        mode=args.mode, 
        N=args.N, 
        input_dir=dynamic_input_dir, 
        output_dir=dynamic_output_dir, 
        target_col=args.target, 
        use_rf=args.use_rf
    )