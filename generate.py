# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

<<<<<<< Updated upstream
"""Generate random MNIST images and compute FID score using the techniques described in the paper
=======
"""Generate random MNIST images and compute FID score and Sliced Wasserstein Distance using the techniques described in the paper
>>>>>>> Stashed changes
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

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
import PIL.Image
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3
import torchvision.transforms as transforms
import torchvision.datasets as datasets
<<<<<<< Updated upstream
=======
import ot
>>>>>>> Stashed changes

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
<<<<<<< Updated upstream
=======
# Compute Sliced Wasserstein Distance

def compute_sliced_wasserstein_distance(x: np.ndarray, y: np.ndarray, n_projections: int = 1000) -> float:
    """
    Compute the sliced Wasserstein distance between two sets of samples.
    
    Args:
        x (np.ndarray): First set of samples, shape (n_samples, n_features)
        y (np.ndarray): Second set of samples, shape (n_samples, n_features)
        n_projections (int): Number of random projections to use
    
    Returns:
        float: Sliced Wasserstein distance
    """
    n = x.shape[0]
    m = y.shape[0]
    d = x.shape[1]
    a = np.ones(n) / n
    b = np.ones(m) / m
    
    # Subsample to make computation faster
    n_samples = min(1000, min(n, m))
    x_subset = x[np.random.choice(n, n_samples, replace=False)]
    y_subset = y[np.random.choice(m, n_samples, replace=False)]
    
    # Generate random projections
    projections = np.random.randn(n_projections, d)
    projections /= np.sqrt(np.sum(projections**2, axis=1, keepdims=True))
    
    sw_distance = 0.0
    for proj in projections:
        x_proj = np.dot(x_subset, proj)
        y_proj = np.dot(y_subset, proj)
        sw_distance += ot.emd2(a[:n_samples], b[:n_samples], ot.dist(x_proj.reshape(-1, 1), y_proj.reshape(-1, 1)))
    
    return sw_distance / n_projections

#----------------------------------------------------------------------------
>>>>>>> Stashed changes
# FID computation using pytorch-fid library

def compute_fid_pytorch_fid(original_images, generated_images, batch_size=50, device='cuda' if torch.cuda.is_available() else 'cpu', dims=2048):
    """
    Compute FID score using pytorch-fid library for two sets of images.

    Args:
        original_images (torch.Tensor or np.ndarray): Shape (N, 784), original dataset images, values in [0, 1].
        generated_images (torch.Tensor or np.ndarray): Shape (N, 784), generated dataset images, values in [0, 1].
        batch_size (int): Batch size for processing images through Inception V3.
        device (str): Device to run computations ('cuda' or 'cpu').
        dims (int): Dimensionality of Inception V3 features (default: 2048).

    Returns:
        float: FID score.
    """
    # Load Inception V3 model
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device).eval()

    # Define preprocessing: Reshape, convert to 3-channel, resize to 299x299, normalize
    preprocess = transforms.Compose([
        transforms.Lambda(lambda x: x.view(-1, 1, 28, 28)),  # Reshape (N, 784) to (N, 1, 28, 28)
        transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1)),  # Repeat grayscale to 3 channels
        transforms.Resize((299, 299), antialias=True),      # Resize to 299x299 for Inception V3
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
    ])

    def get_activations(images, model, batch_size, device):
        """Extract Inception V3 features for a dataset."""
        n_batches = images.shape[0] // batch_size + (1 if images.shape[0] % batch_size else 0)
        activations = []
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, images.shape[0])
            batch = images[start:end].to(device)
            batch = preprocess(batch)
            with torch.no_grad():
                act = model(batch)[0]
            # If model output is not scalar, apply global spatial average pooling
            if act.size(2) != 1 or act.size(3) != 1:
                act = torch.nn.functional.adaptive_avg_pool2d(act, output_size=(1, 1))
            activations.append(act.squeeze().cpu().numpy())
        return np.concatenate(activations, axis=0)

    # Convert numpy arrays to torch tensors if necessary
    if isinstance(original_images, np.ndarray):
        original_images = torch.from_numpy(original_images).float()
    if isinstance(generated_images, np.ndarray):
        generated_images = torch.from_numpy(generated_images).float()

    # Ensure inputs are float tensors
    original_images = original_images.float()
    generated_images = generated_images.float()

    # Get activations for both datasets
    act1 = get_activations(original_images, model, batch_size, device)
    act2 = get_activations(generated_images, model, batch_size, device)

    # Compute FID using pytorch-fid's calculate_frechet_distance
    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)
    fid_value = fid_score.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    return fid_value

#----------------------------------------------------------------------------
# Load true MNIST samples

def load_true_mnist_samples(n_samples=1000):
    """Load true MNIST samples."""
    transform = transforms.Compose([
<<<<<<< Updated upstream
        transforms.ToTensor(),  # Converts to [0, 1]
        transforms.Lambda(lambda x: x.view(-1))  # Flatten to [784]
=======
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.view(-1))
>>>>>>> Stashed changes
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    indices = np.random.choice(len(dataset), size=n_samples, replace=False)
    samples = torch.stack([dataset[i][0] for i in indices])  # Shape [n_samples, 784]
    return samples

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

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--outdir',                  help='Where to save the output samples', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-999', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)
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
@click.option('--n-samples',               help='Number of true samples for metrics', metavar='INT', type=click.IntRange(min=1), default=1000, show_default=True)

def main(network_pkl, outdir, subdirs, seeds, class_idx, max_batch_size, n_samples, **sampler_kwargs):
<<<<<<< Updated upstream
    """Generate random MNIST images and compute FID score using the techniques described in the paper
=======
    """Generate random MNIST images and compute FID score and Sliced Wasserstein Distance using the techniques described in the paper
>>>>>>> Stashed changes
    "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
<<<<<<< Updated upstream
    # Generate 1000 MNIST images and compute FID
=======
    # Generate 1000 MNIST images and compute FID and Sliced Wasserstein Distance
>>>>>>> Stashed changes
    torchrun --standalone --nproc_per_node=1 generate.py --outdir=generated-samples \\
        --network=training-runs/00023-mnist-uncond-ddpmpp-edm-gpus1-batch2048-fp32/network-snapshot-002000.pkl \\
        --seeds=0-999 --batch=64 --steps=40 --n-samples=1000
    """
    dist.init()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

<<<<<<< Updated upstream
    # Load true MNIST samples for FID.
    dist.print0(f'Loading true MNIST samples for FID...')
=======
    # Load true MNIST samples for FID and Sliced Wasserstein.
    dist.print0(f'Loading true MNIST samples for metrics...')
>>>>>>> Stashed changes
    true_samples = load_true_mnist_samples(n_samples=n_samples).to(device)  # [n_samples, 784]

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Initialize metrics storage.
<<<<<<< Updated upstream
    metrics = {'fid': []}
=======
    metrics = {'fid': [], 'sliced_wasserstein': []}
>>>>>>> Stashed changes

    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    generated_samples = []
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)  # [batch, 1, 28, 28]
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

<<<<<<< Updated upstream
        # Convert samples to numpy for saving and FID.
=======
        # Convert samples to numpy for saving and metrics.
>>>>>>> Stashed changes
        samples_np = samples.to(torch.float32).cpu().numpy()  # [batch, 1, 28, 28]
        samples_flat = samples_np.reshape(batch_size, -1)  # [batch, 784]
        generated_samples.append(samples_flat)

        # Save images.
        if dist.get_rank() == 0:
            output_dir = os.path.join(outdir, f'{batch_seeds[0]-batch_seeds[0]%1000:06d}') if subdirs else outdir
            os.makedirs(output_dir, exist_ok=True)
            images_np = (samples_np * 127.5 + 128).clip(0, 255).astype(np.uint8).transpose(0, 2, 3, 1)  # [batch, 28, 28, 1]
            for seed, image_np in zip(batch_seeds, images_np):
                image_path = os.path.join(output_dir, f'{seed:06d}.png')
                PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)

<<<<<<< Updated upstream
    # Compute FID (only on rank 0).
    if dist.get_rank() == 0:
        generated_samples = np.concatenate(generated_samples, axis=0)  # [n_total, 784]
        fid = compute_fid_pytorch_fid(true_samples.cpu().numpy(), generated_samples, batch_size=max_batch_size, device=device)
        metrics['fid'].append(fid)

        # Save metrics.
        metrics_avg = {'fid': float(np.mean(metrics['fid']))}
=======
    # Compute metrics (only on rank 0).
    if dist.get_rank() == 0:
        generated_samples = np.concatenate(generated_samples, axis=0)  # [n_total, 784]
        true_samples_np = true_samples.cpu().numpy()

        # Compute FID
        fid = compute_fid_pytorch_fid(true_samples_np, generated_samples, batch_size=max_batch_size, device=device)
        metrics['fid'].append(fid)

        # Compute Sliced Wasserstein Distance
        sw_distance = compute_sliced_wasserstein_distance(true_samples_np, generated_samples, n_projections=1000)
        metrics['sliced_wasserstein'].append(sw_distance)

        # Save metrics.
        metrics_avg = {
            'fid': float(np.mean(metrics['fid'])),
            'sliced_wasserstein': float(np.mean(metrics['sliced_wasserstein']))
        }
>>>>>>> Stashed changes
        with open(os.path.join(outdir, 'metrics.json'), 'w') as f:
            json.dump(metrics_avg, f, indent=2)
        dist.print0(f'Metrics: {metrics_avg}')

    # Done.
    torch.distributed.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()