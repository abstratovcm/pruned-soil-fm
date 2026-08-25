import numpy as np
import pandas as pd
import os
import joblib
from flowMatching.transforms import LogLinearTransform, PiecewiseExponentialTransform

# Physical Constraints and Precision derived from LUCAS 2015 Data Analysis & README
PRECISION_MAP = {
    'Clay': 0, 'Sand': 0, 'Silt': 0,        # Texture: Integer (0 decimals)
    'pH(CaCl2)': 1, 'pH(H2O)': 2,           # pH: 1 and 2 decimals respectively
    'EC': 2,                                # Electrical Conductivity: 2 decimals
    'OC': 1,                                # Organic Carbon: 1 decimal
    'CaCO3': 0,                             # Carbonates: Integer (0 decimals)
    'P': 1, 'N': 1, 'K': 1,                 # Nutrients: 1 decimal
    'Elevation': 0                          # Elevation: Integer (0 decimals)
}

# Hard Clipping Ranges (Min, Max) - Documented bounds only
CLIP_RANGES = {
    'Clay': (0.0, None),
    'Sand': (0.0, None),
    'Silt': (0.0, None),
    'pH(CaCl2)': (0.0, None),
    'pH(H2O)': (0.0, None),
    'EC': (0.0, None),
    'OC': (0.0, None),
    'CaCO3': (0.0, None),
    'P': (0.0, None),
    'N': (0.0, None),
    'K': (0.0, None),
    'Elevation': (None, None)
}

SHIFT_LOG_FEATURES = {'Elevation': 200} # Kept for reference, handled by LogLinearTransform now

def inverse_log_transform(df, params_path=None):
    """
    Reverses the log transformations applied in data preprocessing.
    """
    # If no path is provided, skip transforms and warn
    if params_path is None:
        print("Warning: No transform parameters path provided. Skipping inverse transforms.")
        return df

    # If the path does not exist, skip and warn
    if not os.path.exists(params_path):
        print(f"Warning: Transform parameters file '{params_path}' does not exist. Skipping inverse transforms.")
        return df

    # At this point, params_path is valid and exists
    try:
        params = joblib.load(params_path)
        # Determine transformer type(s)
        first_key = list(params.keys())[0]
        first_val = params[first_key]

        if isinstance(first_val, dict) and 'T' in first_val:
            # Single LogLinear transform
            transformer = LogLinearTransform()
            transformer.load(params_path)
            df = transformer.inverse_transform(df)
        elif isinstance(first_val, dict) and 'beta' in first_val:
            # Single PiecewiseExponential transform
            transformer = PiecewiseExponentialTransform()
            transformer.load(params_path)
            df = transformer.inverse_transform(df)
        else:
            # Check for a bundle of transforms (loglinear + piecewise_exp)
            if 'loglinear' in params and 'piecewise_exp' in params:
                # Reverse order of training application
                # Training: Dequant -> PiecewiseExp -> LogLinear
                # Inverse: LogLinear -> PiecewiseExp
                ll_trans = params['loglinear']
                df = ll_trans.inverse_transform(df)

                pe_trans = params['piecewise_exp']
                df = pe_trans.inverse_transform(df)
            else:
                print(f"  [clean_samples] Warning: Unknown param format in {params_path}")
    except Exception as e:
        print(f"  [clean_samples] Error loading params: {e}")

    return df

def clean_samples(data, feature_names, transform_params_path=None):
    """
    Applies strict physical constraints, LOD snapping, and precision rounding to soil data samples.
    """
    # Convert to DataFrame for easier column manipulation
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.DataFrame(data, columns=feature_names)

    # 1. INVERSE TRANSFORM (Log/LogLinear -> Linear)
    df = inverse_log_transform(df, params_path=transform_params_path)

    # Hard Clipping (Ranges)
    for feat, (min_val, max_val) in CLIP_RANGES.items():
        if feat in df.columns:
            df[feat] = df[feat].clip(lower=min_val, upper=max_val)

    # Precision Rounding (Natural Quantization)
    texture_feats = ['Clay', 'Sand', 'Silt']
    for feat, decimals in PRECISION_MAP.items():
        if feat in df.columns and feat not in texture_feats:
            df[feat] = df[feat].round(decimals)

    # Texture Constraint (Normalize -> Round Integer)
    if all(feat in df.columns for feat in texture_feats):
        # Ensure non-negative
        df[texture_feats] = df[texture_feats].clip(lower=0.0)

        # Calculate sum
        sums = df[texture_feats].sum(axis=1)
        mask = sums > 0

        if mask.any():
            df.loc[mask, texture_feats] = df.loc[mask, texture_feats].div(sums[mask], axis=0) * 100.0

            df[texture_feats] = df[texture_feats].round(0)

    # Return as numpy array if input was numpy array
    if not isinstance(data, pd.DataFrame):
        return df.values

    return df
