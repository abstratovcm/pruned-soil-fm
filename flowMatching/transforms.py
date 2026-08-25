import numpy as np
import joblib
import os

class PiecewiseExponentialTransform:
    def __init__(self, lower_q=0.05, upper_q=0.95):
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.params = {}

    def fit(self, df, features):
        """
        Calculates thresholds and scaling factors for piecewise exponential transform.
        Logic:
        - Middle (T1 <= x <= T2): Linear (Identity)
        - Lower Tail (x < T1): x = T1 - (1/beta)*(exp(alpha*(y_raw)) - 1)  Wait, y = T1 - ...
          Target: As x moves away from T1, y moves away exponentially.
          Forward: y = T1 - (1/beta) * (exp(beta * (T1 - x)) - 1)
          Slope at T1: dy/dx = - (1/beta) * exp(0) * beta * (-1) = 1. Matches Identity.
        - Upper Tail (x > T2): y = T2 + (1/beta) * (exp(beta * (x - T2)) - 1)
          Slope at T2: dy/dx = 1. Matches Identity.

        We fix beta = 1.0 for simplicity, or we could tune it. Let's start with beta=1.0.
        """
        for feat in features:
            if feat not in df.columns:
                continue

            vals = df[feat].values
            T1 = np.percentile(vals, self.lower_q * 100)
            T2 = np.percentile(vals, self.upper_q * 100)

            # Beta parameter controls steepness of the exponential
            # Larger beta = faster explosion = stronger wall
            beta = 1.0

            self.params[feat] = {
                'T1': T1,
                'T2': T2,
                'beta': beta
            }

    def transform(self, df):
        df_out = df.copy()
        for feat, p in self.params.items():
            if feat not in df_out.columns:
                continue

            if df_out[feat].dtype != np.float64:
                df_out[feat] = df_out[feat].astype(np.float64)

            T1, T2, beta = p['T1'], p['T2'], p['beta']
            vals = df_out[feat].values

            mask_low = vals < T1
            mask_high = vals > T2
            # mask_mid = (vals >= T1) & (vals <= T2) # Identity, do nothing

            # Lower Tail: x < T1
            # y = T1 - (1/beta)*(exp(beta*(T1-x)) - 1)
            # As x -> -inf, y -> -inf exponentially
            diff_low = T1 - vals[mask_low]
            df_out.loc[mask_low, feat] = T1 - (1.0/beta) * (np.expm1(beta * diff_low))

            # Upper Tail: x > T2
            # y = T2 + (1/beta)*(exp(beta*(x-T2)) - 1)
            diff_high = vals[mask_high] - T2
            df_out.loc[mask_high, feat] = T2 + (1.0/beta) * (np.expm1(beta * diff_high))

        return df_out

    def inverse_transform(self, df):
        df_out = df.copy()
        for feat, p in self.params.items():
            if feat not in df_out.columns:
                continue

            if df_out[feat].dtype != np.float64:
                df_out[feat] = df_out[feat].astype(np.float64)

            T1, T2, beta = p['T1'], p['T2'], p['beta']
            vals = df_out[feat].values

            # Determine boundaries in Y-space corresponding to T1, T2
            # Middle maps to Identity, so Y_T1 = T1, Y_T2 = T2

            mask_low = vals < T1
            mask_high = vals > T2

            # Inverse Lower: y = T1 - (1/b)(exp(b(T1-x)) - 1)
            # (T1 - y)*b = exp(b(T1-x)) - 1
            # 1 + b(T1-y) = exp(b(T1-x))
            # ln(...) = b(T1-x)
            # T1 - x = (1/b) ln(1 + b(T1-y))
            # x = T1 - (1/b) ln(1 + b(T1-y))
            # Ensure argument to log is positive: 1 + b(T1-y) > 0 => b(T1-y) > -1.
            # Since y < T1, T1-y > 0, so b(T1-y) is positive. Safe.

            diff_low_y = T1 - vals[mask_low]
            # Log1p(z) calculates ln(1+z)
            df_out.loc[mask_low, feat] = T1 - (1.0/beta) * np.log1p(beta * diff_low_y)

            # Inverse Upper: y = T2 + (1/b)(exp(b(x-T2)) - 1)
            # (y - T2)*b = exp(...) - 1
            # ln(1 + b(y-T2)) = b(x-T2)
            # x = T2 + (1/b) ln(1 + b(y-T2))

            diff_high_y = vals[mask_high] - T2
            df_out.loc[mask_high, feat] = T2 + (1.0/beta) * np.log1p(beta * diff_high_y)

        return df_out

    def save(self, path):
        joblib.dump(self.params, path)
        print(f"Saved PiecewiseExp params to {path}")

    def load(self, path):
        if os.path.exists(path):
            self.params = joblib.load(path)
        else:
            print(f"Warning: Params file not found at {path}")

class LogLinearTransform:
    def __init__(self, percentile=75, offsets=None):
        self.percentile = percentile
        self.offsets = offsets if offsets else {}
        self.params = {} # Stores {feature_name: {'T': threshold, 'm': slope, 'b': intercept, 'offset': offset}}

    def fit(self, df, features):
        """
        Calculates the transition threshold T, slope m, and intercept b for each feature.
        """
        for feat in features:
            if feat not in df.columns:
                continue

            offset = self.offsets.get(feat, 1.0)

            # Calculate Threshold T (percentile)
            vals = df[feat].values
            T = np.percentile(vals, self.percentile)

            # Safety: Ensure T >= -offset (typically T >= 0 for offset=1)
            if offset == 1.0:
                T = max(0.0, T)
            else:
                T = max(-offset + 1e-3, T)

            # Calculate Slope m at T for continuity
            # Derivative of log(x+offset) is 1/(x+offset)
            m = 1.0 / (T + offset)

            # Calculate Intercept b for continuity
            y_at_T = np.log(T + offset) if offset != 1.0 else np.log1p(T)
            b = y_at_T - (m * T)

            self.params[feat] = {
                'T': T,
                'm': m,
                'b': b,
                'offset': offset
            }

    def transform(self, df):
        """
        Applies the piecewise transform:
        x <= T: log(x + offset)
        x > T:  m*x + b
        """
        # Ensure we are working with floats to avoid LossySetitemError
        df_out = df.copy()

        for feat, p in self.params.items():
            if feat not in df_out.columns:
                continue

            # Cast column to float64 for safety/precision
            if df_out[feat].dtype != np.float64:
                df_out[feat] = df_out[feat].astype(np.float64)

            T = p['T']
            m = p['m']
            b = p['b']
            offset = p.get('offset', 1.0)

            vals = df_out[feat].values

            # Create masks
            mask_log = vals <= T
            mask_lin = vals > T

            # Apply Log part
            # Clip to be safe for log (or log1p) allowing some dequantization noise
            if offset == 1.0:
                safe_vals = np.clip(vals, -0.99, None) # Relaxed to allow symmetric dequantization noise near 0
                df_out.loc[mask_log, feat] = np.log1p(safe_vals[mask_log])
            else:
                safe_vals = np.clip(vals, -offset + 1e-3, None)
                df_out.loc[mask_log, feat] = np.log(safe_vals[mask_log] + offset)

            # Apply Linear part
            df_out.loc[mask_lin, feat] = m * vals[mask_lin] + b

        return df_out

    def inverse_transform(self, df):
        """
        Reverses the transform:
        y <= log(T+offset): exp(y) - offset
        y > log(T+offset):  (y - b) / m
        """
        df_out = df.copy()
        for feat, p in self.params.items():
            if feat not in df_out.columns:
                continue

            # Cast column to float64 for safety/precision
            if df_out[feat].dtype != np.float64:
                df_out[feat] = df_out[feat].astype(np.float64)

            T = p['T']
            m = p['m']
            b = p['b']
            offset = p.get('offset', 1.0)

            # Calculate the y-value at the transition point
            y_trans = np.log(T + offset) if offset != 1.0 else np.log1p(T)

            vals = df_out[feat].values

            mask_log = vals <= y_trans
            mask_lin = vals > y_trans

            # Inverse Log
            if offset == 1.0:
                df_out.loc[mask_log, feat] = np.expm1(vals[mask_log])
            else:
                df_out.loc[mask_log, feat] = np.exp(vals[mask_log]) - offset

            # Inverse Linear
            # x = (y - b) / m
            df_out.loc[mask_lin, feat] = (vals[mask_lin] - b) / m

        return df_out

    def save(self, path):
        joblib.dump(self.params, path)
        print(f"Saved LogLinear params to {path}")

    def load(self, path):
        if os.path.exists(path):
            self.params = joblib.load(path)
            # print(f"Loaded LogLinear params for: {list(self.params.keys())}")
        else:
            print(f"Warning: Params file not found at {path}")


class DequantizationTransform:
    def __init__(self, precision_map=None, margin=0.45):
        self.precision_map = precision_map if precision_map is not None else {}
        self.margin = margin
        self.params = {} # Stores features that were dequantized

    def fit(self, df, features):
        for feat in features:
            if feat in df.columns and feat in self.precision_map:
                self.params[feat] = self.precision_map[feat]

    def transform(self, df):
        df_out = df.copy()
        for feat, precision in self.params.items():
            if feat in df_out.columns:
                if df_out[feat].dtype != np.float64:
                    df_out[feat] = df_out[feat].astype(np.float64)

                # Calculate the bounds based on precision and margin
                scale = self.margin * (10.0 ** -precision)

                # Generate uniform noise (Pure symmetric, no physical clipping here)
                noise = np.random.uniform(low=-scale, high=scale, size=len(df_out))
                df_out[feat] = df_out[feat] + noise
        return df_out

    def save(self, path):
        import joblib
        import os
        joblib.dump(self.params, path)
        print(f"Saved Dequantization params to {path}")

    def load(self, path):
        import joblib
        import os
        if os.path.exists(path):
            self.params = joblib.load(path)
        else:
            print(f"Warning: Params file not found at {path}")
