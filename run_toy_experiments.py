import os
import subprocess
import json
import numpy as np
import argparse
import time

def find_latest_directory(path):
    try:
        directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        if not directories:
            return None, None
        return max(directories, 
                        key=lambda d: os.path.getmtime(os.path.join(path, d)))
    except FileNotFoundError:
        return None, "Directory path not found"
    except Exception as e:
        return None, f"Error: {str(e)}"

def find_latest_pkl_file(path):
    try:
        pkl_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.pkl')]
        if not pkl_files:
            return None, "No .pkl files found"
        return max(pkl_files, 
                         key=lambda f: os.path.getmtime(os.path.join(path, f)))
    except FileNotFoundError:
        return None, "Directory path not found"
    except Exception as e:
        return None, f"Error: {str(e)}"

def run_experiment(dataset, seeds, base_outdir, gen_outdir, n_samples, num_steps, batch_size, master_port, num_mixture=8, radius=8.0, sigma=1.0):
    # Ensure output directories exist
    os.makedirs(base_outdir, exist_ok=True)
    os.makedirs(gen_outdir, exist_ok=True)

    # Store metrics for all runs
    all_metrics = []

    # Training and generation commands
    for seed in seeds:
        # Training command
        train_cmd = [
            "torchrun", "--standalone", f"--nproc_per_node=4", f"--master_port={master_port}",
            "train.py",
            "--outdir", base_outdir,
            "--data", f"custom:{dataset}",
            "--dataset-type", dataset,
            "--n-samples", str(n_samples),
            "--cond", "0",
            "--arch", "ddpmpp",
            "--precond", "edm",
            "--duration", "200",
            "--batch", "1024",
            "--sigma", "0.5",
            "--seed", str(seed)
        ]
        if dataset == "ngaussian":
            train_cmd.extend(["--num-mixture", str(num_mixture), "--radius", str(radius), "--sigma", str(sigma)])

        print(f"Running training for {dataset} with seed {seed}...")
        subprocess.run(train_cmd, check=True)

        # Find the latest snapshot
        run_dir = find_latest_directory(base_outdir)
        snapshot = os.path.join(base_outdir, run_dir, find_latest_pkl_file(os.path.join(base_outdir, run_dir)))

        # Generation command
        gen_cmd = [
            "torchrun", "--standalone", f"--nproc_per_node=4", f"--master_port={master_port}",
            "generate.py",
            "--outdir", gen_outdir,
            "--network", snapshot,
            "--seeds", "0-7999",
            "--batch", str(batch_size),
            "--steps", str(num_steps),
            "--dataset-type", dataset,
            "--n-samples", str(n_samples)
        ]
        if dataset == "ngaussian":
            gen_cmd.extend(["--num-mixture", str(num_mixture), "--radius", str(radius), "--sigma", str(sigma)])

        print(f"Running generation for {dataset} with seed {seed}...")
        subprocess.run(gen_cmd, check=True)

        # Load metrics
        metrics_file = os.path.join(gen_outdir, "metrics.json")
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        metrics['seed'] = seed
        all_metrics.append(metrics)

    # Compute mean and variance of metrics
    wasserstein_values = [m['wasserstein'] for m in all_metrics]
    mmd_values = [m['mmd'] for m in all_metrics]
    nll_values = [m['nll'] for m in all_metrics]

    metrics_summary = {
        'dataset': dataset,
        'seeds': seeds,
        'wasserstein_mean': float(np.mean(wasserstein_values)),
        'wasserstein_variance': float(np.var(wasserstein_values)),
        'mmd_mean': float(np.mean(mmd_values)),
        'mmd_variance': float(np.var(mmd_values)),
        'nll_mean': float(np.mean(nll_values)),
        'nll_variance': float(np.var(nll_values))
    }

    # Save summary to text file
    summary_file = os.path.join(gen_outdir, f"{dataset}_metrics_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Wasserstein Mean: {metrics_summary['wasserstein_mean']:.6f}\n")
        f.write(f"Wasserstein Variance: {metrics_summary['wasserstein_variance']:.6f}\n")
        f.write(f"MMD Mean: {metrics_summary['mmd_mean']:.6f}\n")
        f.write(f"MMD Variance: {metrics_summary['mmd_variance']:.6f}\n")
        f.write(f"NLL Mean: {metrics_summary['nll_mean']:.6f}\n")
        f.write(f"NLL Variance: {metrics_summary['nll_variance']:.6f}\n")

    print(f"Metrics summary saved to {summary_file}")
    print(metrics_summary)

    return metrics_summary

def main():
    parser = argparse.ArgumentParser(description="Run training and generation experiments for custom 2D point cloud datasets.")
    parser.add_argument('--dataset', type=str, choices=['swissroll', 'checkerboard', 'ngaussian', 'all'], default='all', help='Dataset to run experiments on (or "all" for all datasets).')
    parser.add_argument('--master-port', type=int, default=12345, help='Master port for distributed training.')
    parser.add_argument('--base-outdir', type=str, default='training-runs', help='Base output directory for training runs.')
    args = parser.parse_args()

    # Experiment parameters
    seeds = [33, 85, 69]
    base_outdir = args.base_outdir
    n_samples = 8000
    num_steps = 500
    batch_size = 1000
    num_mixture = 8
    radius = 8.0
    sigma = 1.0

    datasets = ['swissroll', 'checkerboard', 'ngaussian'] if args.dataset == 'all' else [args.dataset]

    # Run experiments for each dataset
    for dataset in datasets:
        gen_outdir = f"generated-samples/{dataset}"
        print(f"\nRunning experiments for {dataset}...")
        run_experiment(
            dataset=dataset,
            seeds=seeds,
            base_outdir=base_outdir,
            gen_outdir=gen_outdir,
            n_samples=n_samples,
            num_steps=num_steps,
            batch_size=batch_size,
            master_port=args.master_port,
            num_mixture=num_mixture,
            radius=radius,
            sigma=sigma
        )

if __name__ == "__main__":
    main()