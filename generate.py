"""
torchrun --standalone --nproc_per_node=1 generate.py \
    --outdir=generated-samples \
    --network=training-runs/00023-swissroll-uncond-ddpmpp-edm-gpus1-batch2048-fp32/network-snapshot-002000.pkl \
    --seeds=0-999 \
    --batch=1000 \
    --steps=40 \
    --dataset-type=swissroll
"""
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random 2D point cloud samples and compute evaluation metrics (Wasserstein, MMD, NLL)
using the techniques described in the paper "Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import json
import dnnlib
from torch_utils import distributed as dist
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import ot

#----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2).

def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat.view(1, 1, 1, 1).repeat(x_hat.shape[0], 1, 1, 1), class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next.view(1, 1, 1, 1).repeat(x_next.shape[0], 1, 1, 1), class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

#----------------------------------------------------------------------------
# Generalized ablation sampler.

def ablation_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=None, sigma_max=None, rho=7,
    solver='heun', discretization='edm', schedule='linear', scaling='none',
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    # Define noise level schedule.
    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
        t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

        # Euler step.
        h = t_next - t_hat
        denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels).to(torch.float64)
        d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
        x_prime = x_hat + alpha * h * d_cur
        t_prime = t_hat + alpha * h

        # Apply 2nd order correction.
        if solver == 'euler' or i == num_steps - 1:
            x_next = x_hat + h * d_cur
        else:
            assert solver == 'heun'
            denoised = net(x_prime / s(t_prime), sigma(t_prime), class_labels).to(torch.float64)
            d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)

    return x_next

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------
# MMD computation with Gaussian kernel

def compute_mmd(samples1, samples2, sigma=1.0):
    """Compute Maximum Mean Discrepancy with Gaussian kernel."""
    n1, n2 = len(samples1), len(samples2)
    X = samples1
    Y = samples2
    
    # Compute kernel matrices
    XX = torch.cdist(X, X, p=2) ** 2
    YY = torch.cdist(Y, Y, p=2) ** 2
    XY = torch.cdist(X, Y, p=2) ** 2
    
    # Gaussian kernel
    K_XX = torch.exp(-XX / (2 * sigma ** 2))
    K_YY = torch.exp(-YY / (2 * sigma ** 2))
    K_XY = torch.exp(-XY / (2 * sigma ** 2))
    
    # MMD calculation
    mmd = (K_XX.sum() / (n1 * n1) + K_YY.sum() / (n2 * n2) - 2 * K_XY.sum() / (n1 * n2)).item()
    return mmd

#----------------------------------------------------------------------------
# Load true dataset samples

def load_true_samples(dataset_type, n_samples=8000, num_mixture=8, radius=8.0, sigma=1.0):
    """Load true samples from Custom2DDataset."""
    from training.dataset import Custom2DDataset
    dataset = Custom2DDataset(
        path=f'custom:{dataset_type}',
        dataset_type=dataset_type,
        n_samples=n_samples,
        num_mixture=num_mixture,
        radius=radius,
        sigma=sigma
    )
    indices = np.random.choice(len(dataset), size=n_samples, replace=False)
    samples = np.array([dataset[i][0].squeeze(0).squeeze(-1) for i in indices])  # Shape [n_samples, 2]
    return torch.from_numpy(samples).float()

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--outdir',                  help='Where to save the output samples', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-999', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=1000, show_default=True)
@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=40, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)
@click.option('--solver',                  help='Ablate ODE solver', metavar='euler|heun',                          type=click.Choice(['euler', 'heun']))
@click.option('--disc', 'discretization',  help='Ablate time step discretization {t_i}', metavar='vp|ve|iddpm|edm', type=click.Choice(['vp', 've', 'iddpm', 'edm']))
@click.option('--schedule',                help='Ablate noise schedule sigma(t)', metavar='vp|ve|linear',           type=click.Choice(['vp', 've', 'linear']))
@click.option('--scaling',                 help='Ablate signal scaling s(t)', metavar='vp|none',                    type=click.Choice(['vp', 'none']))
@click.option('--dataset-type',            help='Custom dataset type (swissroll|checkerboard|ngaussian)', type=click.Choice(['swissroll', 'checkerboard', 'ngaussian']), default='swissroll', show_default=True)
@click.option('--n-samples',               help='Number of true samples for metrics', metavar='INT', type=click.IntRange(min=1), default=8000, show_default=True)
@click.option('--num-mixture',             help='Number of mixtures for NGaussianMixtures', metavar='INT', type=click.IntRange(min=1), default=8, show_default=True)
@click.option('--radius',                  help='Radius for NGaussianMixtures', metavar='FLOAT', type=click.FloatRange(min=0), default=8.0, show_default=True)
@click.option('--sigma',                   help='Sigma for NGaussianMixtures', metavar='FLOAT', type=click.FloatRange(min=0), default=1.0, show_default=True)

def main(network_pkl, outdir, subdirs, seeds, class_idx, max_batch_size, dataset_type, n_samples, num_mixture, radius, sigma, **sampler_kwargs):
    """Generate random 2D point cloud samples and compute evaluation metrics (Wasserstein, MMD, NLL)
    using the techniques described in the paper "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Generate 1000 points for SwissRoll dataset and compute metrics
    torchrun --standalone --nproc_per_node=8 generate.py --outdir=generated-samples \\
        --network=training-runs/00001-swissroll-uncond-ddpmpp-edm-gpus8-batch512-fp32/network-snapshot-002000.pkl \\
        --seeds=0-999 --batch=1000 --steps=40 --dataset-type=swissroll --n-samples=1000
    """
    dist.init()
    device = torch.device('cuda')

    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)

    # Load true dataset samples for metrics.
    dist.print0(f'Loading true {dataset_type} samples for metrics...')
    true_samples = load_true_samples(dataset_type, n_samples=n_samples, num_mixture=num_mixture, radius=radius, sigma=sigma).to(device)

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Initialize metrics storage.
    metrics = {'wasserstein': [], 'mmd': [], 'nll': []}

    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} samples to "{outdir}"...')
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, 1], device=device)  # [batch, 1, 2, 1]
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

        # Generate samples.
        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
        have_ablation_kwargs = any(x in sampler_kwargs for x in ['solver', 'discretization', 'schedule', 'scaling'])
        sampler_fn = ablation_sampler if have_ablation_kwargs else edm_sampler
        samples = sampler_fn(net, latents, class_labels, randn_like=rnd.randn_like, **sampler_kwargs)

        # Convert samples to numpy for saving and metrics.
        samples_np = samples.to(torch.float32).cpu().numpy()  # [batch, 1, 2, 1]
        points = samples_np.squeeze(1).squeeze(-1)  # [batch, 2]
        points = torch.from_numpy(points).to(device)  # [batch, 2]

        # Compute metrics (only on rank 0 to avoid duplication).
        if dist.get_rank() == 0:
            # Wasserstein distance (Wasserstein-2 via optimal transport).
            true_points = true_samples[:batch_size]  # Match batch size
            cost_matrix = torch.cdist(points, true_points, p=2) ** 2  # [batch, batch]
            a = torch.ones(batch_size, device=device) / batch_size  # Uniform distribution
            b = torch.ones(batch_size, device=device) / batch_size
            wasserstein = ot.emd2(a, b, cost_matrix).item()
            metrics['wasserstein'].append(wasserstein)

            # MMD with Gaussian kernel.
            mmd = compute_mmd(points, true_points, sigma=1.0)
            metrics['mmd'].append(mmd)

            # NLL via KDE.
            kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(true_points.cpu().numpy())
            nll = -kde.score_samples(points.cpu().numpy()).mean()
            metrics['nll'].append(nll)

        # Save samples as NumPy arrays.
        if dist.get_rank() == 0:
            output_dir = os.path.join(outdir, f'{batch_seeds[0]-batch_seeds[0]%1000:06d}') if subdirs else outdir
            os.makedirs(output_dir, exist_ok=True)
            np.save(os.path.join(output_dir, f'samples_{batch_seeds[0]}-{batch_seeds[-1]}.npy'), samples_np)

            # Visualize as scatter plot.
            plt.figure(figsize=(8, 8))
            plt.scatter(points.cpu().numpy()[:, 0], points.cpu().numpy()[:, 1], s=1, label='Generated')
            plt.scatter(true_points.cpu().numpy()[:, 0], true_points.cpu().numpy()[:, 1], s=1, alpha=0.5, label='True')
            plt.title(f'{dataset_type} Samples (Seeds {batch_seeds[0]}-{batch_seeds[-1]})')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            plt.savefig(os.path.join(output_dir, f'samples_{batch_seeds[0]}-{batch_seeds[-1]}.png'))
            plt.close()

    # Aggregate and save metrics.
    if dist.get_rank() == 0:
        metrics_avg = {
            'wasserstein': np.mean(metrics['wasserstein']),
            'mmd': np.mean(metrics['mmd']),
            'nll': np.mean(metrics['nll'])
        }
        with open(os.path.join(outdir, 'metrics.json'), 'w') as f:
            json.dump(metrics_avg, f, indent=2)
        dist.print0(f'Metrics: {metrics_avg}')

    # Done.
    torch.distributed.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()