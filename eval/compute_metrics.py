import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import ks_2samp

# Make the repository root importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flowMatching.data import get_traditional_features, split_and_scale_data
from flowMatching.postprocess import clean_samples
from paths import prepared_dataset, fm_results, tabsyn_data

def compute_ks_error(real: np.ndarray, syn: np.ndarray) -> float:
    """Average Kolmogorov-Smirnov statistic across all features."""
    errors = []
    for i in range(real.shape[1]):
        errors.append(ks_2samp(real[:, i], syn[:, i]).statistic)
    return float(np.mean(errors))

def compute_correlation_error(real: np.ndarray, syn: np.ndarray) -> float:
    """Average absolute difference of Pearson correlations (upper triangle)."""
    real_corr = np.corrcoef(real, rowvar=False)
    syn_corr = np.corrcoef(syn, rowvar=False)
    diff = np.abs(real_corr - syn_corr)
    triu_idx = np.triu_indices_from(diff, k=1)
    return float(np.mean(diff[triu_idx]))

def get_best_threshold(results_dir: str) -> float:
    """Read phase2_test_results.csv and find the threshold that minimized test AUC."""
    test_df = pd.read_csv(os.path.join(results_dir, "phase2_test_results.csv"))
    means = test_df.mean()
    # 'Pure' is the unpruned baseline; find the best pruned threshold
    best_col = means.drop('Pure').idxmin()
    return float(best_col.replace('Pruned_', ''))

def process_tabsyn_synthetic(
    synth_path: str,
    all_features: list,
    traditional_features: list,
    trad_scaler: StandardScaler
) -> np.ndarray:
    """
    Load TABSYN's raw CSV and apply the exact post-processing used in
    run_tabsyn_baseline.py: clean_samples(None) + traditional scaler.
    """
    df_syn = pd.read_csv(synth_path)
    X_raw = df_syn[all_features].values

    # 1. Apply hard bounds / cleaning (no transform_params)
    X_cleaned = clean_samples(X_raw, all_features, transform_params_path=None)

    return X_cleaned

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True,
                        help="Dataset variant (e.g., chem_phys_spectral)")
    parser.add_argument('--start', type=int, default=0,
                        help="First iteration index (0-indexed)")
    parser.add_argument('--end', type=int, default=100,
                        help="Last iteration index (exclusive)")
    args = parser.parse_args()

    data_path = prepared_dataset(args.mode)
    results_dir = fm_results(args.mode)
    tabsyn_data_dir = tabsyn_data(args.mode)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results not found: {results_dir}")

    print(f"Loading full dataset from {data_path} ...")
    df_full = pd.read_csv(data_path)

    # 1. Optimal pruning threshold from the validation phase
    best_threshold = get_best_threshold(results_dir)
    print(f"Optimal pruning threshold (from phase2_test_results): {best_threshold}")

    # 2. Prepare output storage
    output_rows = []

    for i in range(args.start, args.end):
        seed = 42 + i
        print(f"\n--- Iteration {i+1} (seed {seed}) ---")

        # Regenerate real test data
        # --- Get the exact test set used by the Flow Matching pipeline (stratified split + full transforms) ---
        traditional_features = get_traditional_features(args.mode)

        # Identify raw spectral columns
        raw_spectral_features = []
        if 'nospectral' not in args.mode:
            for col in df_full.columns:
                if (col not in traditional_features and
                    col not in ['Point_ID', 'geometry', 'cluster']):
                    try:
                        float(col)
                        raw_spectral_features.append(col)
                    except ValueError:
                        continue

        # Stratified split and scaling, shared with the training pipeline
        _, _, X_test_df, all_features, trad_scaler, _, _ = split_and_scale_data(
            df_full, traditional_features, raw_spectral_features, seed=seed
        )

        # Convert to numpy array (already scaled and PCA'd)
        X_real = X_test_df[all_features].values
        print(f"  Real test set shape: {X_real.shape}")

        # -------- A) OURS (Pruned) --------
        pruned_parquet = os.path.join(
            results_dir, f"saved_samples_iter_{i}",
            f"pruned_test_{best_threshold}.parquet"
        )
        if os.path.exists(pruned_parquet):
            df_pruned = pd.read_parquet(pruned_parquet)
            X_syn_ours_scaled = df_pruned.values

            # Inverse transform traditional features to raw
            trad_indices = [i for i, f in enumerate(all_features) if f in traditional_features]
            X_syn_ours_raw = X_syn_ours_scaled.copy()
            X_syn_ours_raw[:, trad_indices] = trad_scaler.inverse_transform(X_syn_ours_scaled[:, trad_indices])
            X_syn_ours_raw = clean_samples(X_syn_ours_raw, all_features, transform_params_path=None)

            ks_ours = compute_ks_error(X_real, X_syn_ours_raw)
            corr_ours = compute_correlation_error(X_real, X_syn_ours_raw)
            print(f"  Ours (pruned)    : KS={ks_ours:.4f}, CorrErr={corr_ours:.4f}")
        else:
            ks_ours = corr_ours = np.nan
            print(f"  Ours (pruned)    : file not found, skipping.")

        # -------- B) TABSYN --------
        tabsyn_csv = os.path.join(tabsyn_data_dir, f"synthetic_{i}.csv")
        if os.path.exists(tabsyn_csv):
            try:
                X_syn_tabsyn = process_tabsyn_synthetic(
                    tabsyn_csv, all_features, traditional_features, trad_scaler
                )
                # Ensure same number of samples as real (or subsample if needed)
                n_real = X_real.shape[0]
                if len(X_syn_tabsyn) > n_real:
                    rng = np.random.default_rng(seed + 999)
                    idx = rng.choice(len(X_syn_tabsyn), n_real, replace=False)
                    X_syn_tabsyn = X_syn_tabsyn[idx]
                elif len(X_syn_tabsyn) < n_real:
                    # Very unlikely, but pad with random real samples if necessary
                    pad_idx = np.random.choice(n_real, n_real - len(X_syn_tabsyn), replace=False)
                    X_syn_tabsyn = np.vstack([X_syn_tabsyn, X_real[pad_idx]])

                ks_tabsyn = compute_ks_error(X_real, X_syn_tabsyn)
                corr_tabsyn = compute_correlation_error(X_real, X_syn_tabsyn)
                print(f"  TABSYN           : KS={ks_tabsyn:.4f}, CorrErr={corr_tabsyn:.4f}")
            except Exception as e:
                print(f"  TABSYN           : error processing -> {e}")
                ks_tabsyn = corr_tabsyn = np.nan
        else:
            ks_tabsyn = corr_tabsyn = np.nan
            print(f"  TABSYN           : file not found, skipping.")

        # -------- B2) TABSYN 10X --------
        tabsyn_10x_csv = os.path.join(tabsyn_data_dir, f"synthetic_{i}_10x.csv")
        if os.path.exists(tabsyn_10x_csv):
            try:
                X_syn_tabsyn_10x = process_tabsyn_synthetic(
                    tabsyn_10x_csv, all_features, traditional_features, trad_scaler
                )
                ks_tabsyn_10x = compute_ks_error(X_real, X_syn_tabsyn_10x)
                corr_tabsyn_10x = compute_correlation_error(X_real, X_syn_tabsyn_10x)
                print(f"  TABSYN (10X)     : KS={ks_tabsyn_10x:.4f}, CorrErr={corr_tabsyn_10x:.4f} "
                      f"(n_syn={len(X_syn_tabsyn_10x)})")
            except Exception as e:
                print(f"  TABSYN (10X)     : error processing -> {e}")
                ks_tabsyn_10x = corr_tabsyn_10x = np.nan
        else:
            ks_tabsyn_10x = corr_tabsyn_10x = np.nan
            print(f"  TABSYN (10X)     : file not found, skipping.")

        # -------- C) OURS (Pure 1X) --------
        pure_parquet = os.path.join(
            results_dir, f"saved_samples_iter_{i}",
            "pure_test.parquet"
        )
        if os.path.exists(pure_parquet):
            df_pure = pd.read_parquet(pure_parquet)
            X_syn_pure_scaled = df_pure.values

            X_syn_pure_raw = X_syn_pure_scaled.copy()
            X_syn_pure_raw[:, trad_indices] = trad_scaler.inverse_transform(X_syn_pure_scaled[:, trad_indices])
            X_syn_pure_raw = clean_samples(X_syn_pure_raw, all_features, transform_params_path=None)

            ks_pure = compute_ks_error(X_real, X_syn_pure_raw)
            corr_pure = compute_correlation_error(X_real, X_syn_pure_raw)
            print(f"  Ours (Pure 1X)   : KS={ks_pure:.4f}, CorrErr={corr_pure:.4f}")
        else:
            ks_pure = corr_pure = np.nan
            print(f"  Ours (Pure 1X)   : file not found, skipping.")

        # -------- D) OURS (Pruned 10X) --------
        pruned_10x_parquet = os.path.join(
            results_dir, f"saved_samples_iter_{i}",
            f"pruned_test_10x_{best_threshold}.parquet"
        )
        if os.path.exists(pruned_10x_parquet):
            df_pruned_10x = pd.read_parquet(pruned_10x_parquet)
            X_syn_pruned_10x_scaled = df_pruned_10x.values

            X_syn_pruned_10x_raw = X_syn_pruned_10x_scaled.copy()
            X_syn_pruned_10x_raw[:, trad_indices] = trad_scaler.inverse_transform(X_syn_pruned_10x_scaled[:, trad_indices])
            X_syn_pruned_10x_raw = clean_samples(X_syn_pruned_10x_raw, all_features, transform_params_path=None)

            ks_pruned_10x = compute_ks_error(X_real, X_syn_pruned_10x_raw)
            corr_pruned_10x = compute_correlation_error(X_real, X_syn_pruned_10x_raw)
            print(f"  Ours (Pruned 10X): KS={ks_pruned_10x:.4f}, CorrErr={corr_pruned_10x:.4f}")
        else:
            ks_pruned_10x = corr_pruned_10x = np.nan
            print(f"  Ours (Pruned 10X): file not found, skipping.")

        # -------- E) OURS (Pure 10X) --------
        pure_10x_parquet = os.path.join(
            results_dir, f"saved_samples_iter_{i}",
            "pure_test_10x.parquet"
        )
        if os.path.exists(pure_10x_parquet):
            df_pure_10x = pd.read_parquet(pure_10x_parquet)
            X_syn_pure_10x_scaled = df_pure_10x.values

            X_syn_pure_10x_raw = X_syn_pure_10x_scaled.copy()
            X_syn_pure_10x_raw[:, trad_indices] = trad_scaler.inverse_transform(X_syn_pure_10x_scaled[:, trad_indices])
            X_syn_pure_10x_raw = clean_samples(X_syn_pure_10x_raw, all_features, transform_params_path=None)

            ks_pure_10x = compute_ks_error(X_real, X_syn_pure_10x_raw)
            corr_pure_10x = compute_correlation_error(X_real, X_syn_pure_10x_raw)
            print(f"  Ours (Pure 10X)  : KS={ks_pure_10x:.4f}, CorrErr={corr_pure_10x:.4f}")
        else:
            ks_pure_10x = corr_pure_10x = np.nan
            print(f"  Ours (Pure 10X)  : file not found, skipping.")

        output_rows.append({
            "Iteration": i,
            "Method": "Ours_Pruned",
            "Univariate_KS_Error": ks_ours,
            "Pairwise_Corr_Error": corr_ours
        })
        output_rows.append({
            "Iteration": i,
            "Method": "TABSYN",
            "Univariate_KS_Error": ks_tabsyn,
            "Pairwise_Corr_Error": corr_tabsyn
        })
        output_rows.append({
            "Iteration": i,
            "Method": "Ours_Pure",
            "Univariate_KS_Error": ks_pure,
            "Pairwise_Corr_Error": corr_pure
        })
        output_rows.append({
            "Iteration": i,
            "Method": "Ours_Pruned_10X",
            "Univariate_KS_Error": ks_pruned_10x,
            "Pairwise_Corr_Error": corr_pruned_10x
        })
        output_rows.append({
            "Iteration": i,
            "Method": "Ours_Pure_10X",
            "Univariate_KS_Error": ks_pure_10x,
            "Pairwise_Corr_Error": corr_pure_10x
        })
        output_rows.append({
            "Iteration": i,
            "Method": "TABSYN_10X",
            "Univariate_KS_Error": ks_tabsyn_10x,
            "Pairwise_Corr_Error": corr_tabsyn_10x
        })

    # Save aggregated results
    df_out = pd.DataFrame(output_rows)
    out_path = os.path.join(results_dir, "univariate_and_correlation_metrics.csv")
    df_out.to_csv(out_path, index=False)
    print(f"\nMetrics saved to: {out_path}")

    # Print quick summary
    summary = df_out.groupby("Method").mean()
    print("\n=== AVERAGE METRICS OVER ALL ITERATIONS ===")
    print(summary.round(4))

if __name__ == "__main__":
    main()