"""Compute FID (Fréchet Inception Distance) between two image sets.

This implementation works on Python 3.11 by avoiding torchvision.feature_extraction
and instead uses a forward hook on Inception's `avgpool` layer to capture
2048-d features.

Usage:
  python fid_score.py --path1 path/to/real --path2 path/to/fake --batch-size 32 [--gpu]

Inputs for `--path`: folder of images (.png/.jpg/.jpeg) or a .npz with 'mu' and 'sigma'.

Dependencies: torch, torchvision, numpy, scipy, pillow, tqdm (optional)
"""

import os
import pathlib
import argparse
import numpy as np
import torch
from scipy import linalg
from PIL import Image

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x):
        return x

try:
    from torchvision.models import inception_v3, Inception_V3_Weights
    from torchvision import transforms
except Exception:
    raise RuntimeError('This script requires torchvision with model weights support. Please install/upgrade torchvision.')


def list_images(path):
    p = pathlib.Path(path)
    exts = ['*.png', '*.jpg', '*.jpeg']
    files = []
    for e in exts:
        files.extend(sorted(p.glob(e)))
    return files


def image_to_tensor(img_path):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img)


def get_inception_model(device='cpu'):
    # load pretrained inception v3 and switch to eval
    # Use the weights enum (compatible across torchvision versions)
    try:
        weights = Inception_V3_Weights.IMAGENET1K_V1
    except Exception:
        # fallback if enum name differs
        weights = None
    if weights is not None:
        model = inception_v3(weights=weights)
    else:
        model = inception_v3(pretrained=True)
    model.eval()
    model.to(device)
    return model


def get_activations(files, model, batch_size=50, device='cpu'):
    """Get 2048-d activations from inception avgpool via forward hook."""
    model.eval()
    n_files = len(files)
    act = np.empty((n_files, 2048), dtype=np.float32)

    # hook to capture avgpool output
    activations = []
    def _hook(module, input, output):
        # output shape (N, 2048, 1, 1)
        activations.append(output.detach().cpu())

    handle = model.avgpool.register_forward_hook(_hook)

    try:
        for i in tqdm(range(0, n_files, batch_size)):
            batch_files = files[i:i+batch_size]
            tensors = [image_to_tensor(str(f)) for f in batch_files]
            batch = torch.stack(tensors, dim=0).to(device)
            activations.clear()
            with torch.no_grad():
                _ = model(batch)  # forward; hook populates `activations`
            if len(activations) == 0:
                raise RuntimeError('Failed to capture activations from Inception model')
            out = activations[0]  # tensor shape (N,2048,1,1)
            out = out.reshape(out.size(0), -1).cpu().numpy()
            act[i:i+out.shape[0]] = out
    finally:
        handle.remove()

    return act


def calculate_activation_statistics(files, model, batch_size=50, device='cpu'):
    act = get_activations(files, model, batch_size, device)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def _compute_statistics_of_path(path, model, batch_size, device):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        files = list_images(path)
        if len(files) == 0:
            raise RuntimeError(f'No images found in {path}')
        m, s = calculate_activation_statistics(files, model, batch_size, device)
    return m, s


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_fid_given_paths(paths, batch_size, device='cpu'):
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    model = get_inception_model(device)
    m1, s1 = _compute_statistics_of_path(paths[0], model, batch_size, device)
    m2, s2 = _compute_statistics_of_path(paths[1], model, batch_size, device)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value


def parse_args():
    parser = argparse.ArgumentParser(description='Compute FID between two folders of images or .npz stats')
    parser.add_argument('--path1', required=True, help='Path to real images or .npz')
    parser.add_argument('--path2', required=True, help='Path to generated images or .npz')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--gpu', action='store_true', help='use GPU')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    fid_value = calculate_fid_given_paths([args.path1, args.path2], args.batch_size, device)
    print(f'FID: {fid_value:.6f}')