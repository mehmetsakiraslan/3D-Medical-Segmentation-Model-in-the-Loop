#!/usr/bin/env python3
import os
import argparse

import numpy as np
import nibabel as nib
from tifffile import imwrite

# Only import torch if CUDA mode is requested
try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None


def extract_2d_patches(
    input_nii: str,
    output_dir: str,
    patch_size: int = 256,
    overlap_pct: float = 0.2,
    axis: int = 2,
    use_cuda: bool = False
):
    """
    Load a 3D NIfTI volume and tile it into overlapping 2D patches
    along a user-specified axis (0, 1, or 2). By default axis=2 (axial).
    If use_cuda=True, slicing and padding happen with PyTorch on GPU.

    Each patch is saved as a separate .tif (uint16), so 3D Slicer can open them.
    """
    # --- 1) Load volume via nibabel ---
    img = nib.load(input_nii)
    data_np = img.get_fdata()          # usually float32 or int
    original_dtype = data_np.dtype     # remember original dtype
    X, Y, Z = data_np.shape
    print(f"Loaded volume of shape (X, Y, Z) = ({X}, {Y}, {Z})")

    os.makedirs(output_dir, exist_ok=True)

    # Compute overlap in pixels and stride
    overlap_px = int(patch_size * overlap_pct)
    stride = patch_size - overlap_px
    if stride < 1:
        raise ValueError("Overlap percentage too high; resulting stride < 1")

    # --- 2) Move data to GPU if requested ---
    if use_cuda:
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but PyTorch or GPU is unavailable.")
        # Convert to torch.Tensor on GPU (preserve dtype)
        data_torch = torch.from_numpy(data_np).to(
            dtype=torch.from_numpy(data_np).dtype
        ).cuda()
    else:
        data_torch = None

    # --- 3) Set up slicing based on chosen axis ---
    if axis == 0:
        n_slices, D0, D1 = X, Y, Z
        slice_fn = lambda s: data_torch[s, :, :] if use_cuda else data_np[s, :, :]
    elif axis == 1:
        n_slices, D0, D1 = Y, X, Z
        slice_fn = lambda s: data_torch[:, s, :] if use_cuda else data_np[:, s, :]
    elif axis == 2:
        n_slices, D0, D1 = Z, X, Y
        slice_fn = lambda s: data_torch[:, :, s] if use_cuda else data_np[:, :, s]
    else:
        raise ValueError("Axis must be 0, 1, or 2.")

    count = 0

    # --- 4) Loop over each 2D slice and tile into patches ---
    for s in range(n_slices):
        slice_2d = slice_fn(s)

        for i in range(0, D0, stride):
            for j in range(0, D1, stride):
                # Extract (i:i+patch_size, j:j+patch_size)
                if use_cuda:
                    patch = slice_2d[i : i + patch_size, j : j + patch_size]
                    h, w = patch.shape
                    pad_h = patch_size - h
                    pad_w = patch_size - w
                    if pad_h > 0 or pad_w > 0:
                        # pad = (left, right, top, bottom)
                        pad = (0, pad_w, 0, pad_h)
                        patch = F.pad(patch, pad, mode='constant', value=0)
                    patch_np = patch.cpu().numpy()
                else:
                    patch = slice_2d[i : i + patch_size, j : j + patch_size]
                    h, w = patch.shape
                    pad_h = patch_size - h
                    pad_w = patch_size - w
                    if pad_h > 0 or pad_w > 0:
                        patch = np.pad(
                            patch,
                            ((0, pad_h), (0, pad_w)),
                            mode='constant',
                            constant_values=0
                        )
                    patch_np = patch

                # --- 5) Convert patch to uint16 before saving for Slicer compatibility ---
                # If the original was float, we scale to the full uint16 range.
                if np.issubdtype(original_dtype, np.floating):
                    # Normalize to [0..65535]
                    minv, maxv = patch_np.min(), patch_np.max()
                    if maxv > minv:
                        patch_uint16 = ((patch_np - minv) / (maxv - minv) * 65535.0).astype(np.uint16)
                    else:
                        patch_uint16 = np.zeros_like(patch_np, dtype=np.uint16)
                else:
                    # Already integer: just cast to uint16
                    patch_uint16 = patch_np.astype(np.uint16)

                # --- 6) Save as TIFF ---
                out_name = f"patch_z{s:03d}_x-{i:04d}_y{j:04d}.tif"
                out_path = os.path.join(output_dir, out_name)
                imwrite(out_path, patch_uint16)
                count += 1

    print(f"\n✓ Extracted and saved {count} overlapping patches to '{output_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tile a 3D .nii.gz volume into overlapping 2D patches (saved as .tif)."
    )
    parser.add_argument(
        "input_nii",
        help="Path to the input 3D .nii.gz file."
    )
    parser.add_argument(
        "output_dir",
        help="Directory where the patch .tif files will be written."
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=256,
        help="Size of the square patches (default: 256)."
    )
    parser.add_argument(
        "--overlap_pct",
        type=float,
        default=0.2,
        help="Fractional overlap between patches (default: 0.2)."
    )
    parser.add_argument(
        "--axis",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help=(
            "Axis along which to slice: "
            "0 = sagittal (X), "
            "1 = coronal (Y), "
            "2 = axial (Z). "
            "Default is 2 (axial)."
        )
    )
    parser.add_argument(
        "--use_cuda",
        action="store_true",
        help="If set, do slicing + padding on GPU via PyTorch/CUDA."
    )

    args = parser.parse_args()
    extract_2d_patches(
        args.input_nii,
        args.output_dir,
        patch_size=args.patch_size,
        overlap_pct=args.overlap_pct,
        axis=args.axis,
        use_cuda=args.use_cuda
    )

