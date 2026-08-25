import pandas as pd
import geopandas as gpd
import os
import glob
import warnings

from paths import DATA_DIR, PREPARED_DATA

warnings.filterwarnings("ignore")

# Constants
TOPSOIL_DIR = os.path.join(DATA_DIR, 'LUCAS2015_topsoildata_20200323')
DATA_PATH = os.path.join(TOPSOIL_DIR, 'LUCAS_Topsoil_2015_20200323.csv')
SHP_PATH = os.path.join(TOPSOIL_DIR, 'LUCAS_Topsoil_2015_20200323-shapefile',
                        'LUCAS_Topsoil_2015_20200323.shp')
SPECTRAL_DIR = os.path.join(DATA_DIR, 'LUCAS2015_spectra', 'LUCAS2015_Soil_Spectra_EU28')
OUTPUT_DIR = PREPARED_DATA

FEATURES_FULL = ['pH(CaCl2)', 'pH(H2O)', 'EC', 'OC', 'CaCO3', 'P', 'N', 'K', 'Elevation']
FEATURES_SUBSET = ['Clay', 'Sand', 'Silt'] + FEATURES_FULL

MODES = ['chem_nospectral', 'chem_spectral', 'chem_phys_spectral', 'chem_phys_nospectral']

def load_spectral_data():
    if not os.path.exists(SPECTRAL_DIR):
        return pd.DataFrame()

    all_files = glob.glob(os.path.join(SPECTRAL_DIR, "*.csv"))
    spectral_dfs = []

    for filename in all_files:
        df = pd.read_csv(filename, low_memory=False)
        spectral_cols = [col for col in df.columns if _is_float(col)]
        if not spectral_cols:
            continue
        
        if 'PointID' in df.columns:
            df = df.rename(columns={'PointID': 'Point_ID'})
        elif 'Point_ID' not in df.columns:
            continue

        cols_to_keep = ['Point_ID'] + spectral_cols
        spectral_dfs.append(df[cols_to_keep])

    if not spectral_dfs:
        return pd.DataFrame()

    full_spectral_df = pd.concat(spectral_dfs, ignore_index=True)
    averaged_spectral_df = full_spectral_df.groupby('Point_ID').mean().reset_index()
    return averaged_spectral_df

def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def main():
    print("Loading base data...")
    df = pd.read_csv(DATA_PATH)
    
    # Point_ID order comes from the shapefile
    # We do NOT read geometry to save memory and time
    gdf_geo = gpd.read_file(SHP_PATH, ignore_geometry=True)[['Point_ID']]
    
    df['Point_ID'] = df['Point_ID'].astype(int)
    gdf_geo['Point_ID'] = gdf_geo['Point_ID'].astype(int)
    
    base_merged = gdf_geo.merge(df, on='Point_ID', how='inner')
    
    spectral_df = load_spectral_data()
    spectral_features = [c for c in spectral_df.columns if c != 'Point_ID'] if not spectral_df.empty else []

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for mode in MODES:
        print(f"\nProcessing {mode}...")
        
        if mode in ['chem_phys_nospectral', 'chem_phys_spectral']:
            base_features = FEATURES_SUBSET
        else:
            base_features = FEATURES_FULL
            
        use_spectral = 'nospectral' not in mode
        
        if use_spectral and not spectral_df.empty:
            merged_df = base_merged.merge(spectral_df, on='Point_ID', how='inner')
            current_spectral_features = spectral_features
        else:
            merged_df = base_merged.copy()
            current_spectral_features = []
            
        all_features = base_features + current_spectral_features
        valid_features = [f for f in all_features if f in merged_df.columns]
        
        cols_to_keep = ['Point_ID'] + valid_features
        df_selected = merged_df[cols_to_keep].copy()
        
        initial_len = len(df_selected)
        df_selected = df_selected.dropna()
        print(f"  Dropped {initial_len - len(df_selected)} rows with missing values.")
        print(f"  Remaining data points: {len(df_selected)}")
        
        df_selected['cluster'] = 0
        
        mode_dir = os.path.join(OUTPUT_DIR, mode)
        os.makedirs(mode_dir, exist_ok=True)
        
        output_path = os.path.join(mode_dir, f"{mode}.csv")
        df_selected.to_csv(output_path, index=False)
        print(f"  Saved to {output_path}")

if __name__ == "__main__":
    main()
