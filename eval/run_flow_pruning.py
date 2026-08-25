import os
import sys
import subprocess
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import fm_results

def run_all_experiments(N=100):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_script = os.path.join(base_dir, 'validation_pipeline.py')

    datasets = [
        "chem_nospectral",
        "chem_spectral",
        "chem_phys_nospectral",
        "chem_phys_spectral"
    ]

    for mode in datasets:
        print(f"\n{'='*80}")
        print(f"Starting Simplified Experiment for Dataset Variant: {mode}")
        print(f"{'='*80}\n")

        output_dir = fm_results(mode)
        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            "python", pipeline_script,
            "--mode", mode,
            "--N", str(N),
            "--output_dir", output_dir
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"\nExperiment for {mode} completed successfully.\n")
        except subprocess.CalledProcessError as e:
            print(f"\nError: Experiment for {mode} failed with return code {e.returncode}.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=100, help="Number of Monte Carlo iterations (default: 100)")
    args = parser.parse_args()
    
    run_all_experiments(N=args.N)
