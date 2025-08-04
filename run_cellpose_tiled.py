#!/usr/bin/env python3

import os
import tifffile
import numpy as np
import subprocess
from tqdm import tqdm
import shutil

# ==== CONFIGURATION ====
INPUT_TIF = "/scratch/msa6093/AAA392_gut_organ_image.tif"
PATCH_DIR = "/scratch/msa6093/cellpose_patches"
OUTPUT_TIF = "/scratch/msa6093/AAA392_gut_organ_segmented_merged.tif"

# Patch parameters (Z, Y, X)
PATCH_SIZE = (64, 512, 512)
OVERLAP = (16, 128, 128)  # overlap in each axis

# Cellpose CLI arguments
CELLPose_ARGS = "--do_3D --use_gpu --min_size 20 --flow_threshold 0.0 --save_tif --no_npy"

# ==== HELPERS ====
def compute_ranges(dim, patch, overlap):
    if patch >= dim:
        return [(0, dim)]
    stride = patch - overlap
    ranges = []
    start = 0
    while start + patch < dim:
        ranges.append((start, start + patch))
        start += stride
    ranges.append((dim - patch, dim))
    return ranges

def generate_patches_and_centers(vol_shape, patch_size, overlap):
    Z, Y, X = vol_shape
    zs = compute_ranges(Z, patch_size[0], overlap[0])
    ys = compute_ranges(Y, patch_size[1], overlap[1])
    xs = compute_ranges(X, patch_size[2], overlap[2])
    center = np.array([Z/2, Y/2, X/2])
    patch_infos = []
    for iz, (z0, z1) in enumerate(zs):
        for iy, (y0, y1) in enumerate(ys):
            for ix, (x0, x1) in enumerate(xs):
                # center of this patch
                cz = (z0 + z1) / 2.0
                cy = (y0 + y1) / 2.0
                cx = (x0 + x1) / 2.0
                dist = np.linalg.norm(np.array([cz, cy, cx]) - center)
                name = f"patch_{iz}_{iy}_{ix}"
                slicer = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
                patch_infos.append({
                    "name": name,
                    "slicer": slicer,
                    "center_dist": dist,
                })
    # sort so that patches with smaller distance (closer to volume center) come later
    patch_infos.sort(key=lambda x: x["center_dist"], reverse=True)
    return patch_infos

# ==== MAIN ====
def main():
    if not os.path.isfile(INPUT_TIF):
        raise FileNotFoundError(f"Input TIFF not found: {INPUT_TIF}")

    # prepare directories
    if os.path.exists(PATCH_DIR):
        shutil.rmtree(PATCH_DIR)
    os.makedirs(PATCH_DIR, exist_ok=True)

    print("[+] Loading input volume")
    volume = tifffile.imread(INPUT_TIF)  # (Z,Y,X)
    vol_shape = volume.shape
    print(f"[+] Volume shape: {vol_shape}")

    # generate patch metadata sorted for merge
    patch_infos = generate_patches_and_centers(vol_shape, PATCH_SIZE, OVERLAP)
    print(f"[+] Total patches: {len(patch_infos)}")

    # container for merged result (label-preserving overwrite)
    merged = np.zeros(vol_shape, dtype=np.uint16)

    # process patches sequentially
    for info in tqdm(patch_infos, desc="Patches -> Cellpose -> Merge"):
        name = info["name"]
        zsl, ysl, xsl = info["slicer"]
        patch = volume[zsl, ysl, xsl]
        patch_path = os.path.join(PATCH_DIR, f"{name}.tif")
        mask_path = os.path.join(PATCH_DIR, f"{name}_cp_masks.tif")

        # write patch
        tifffile.imwrite(patch_path, patch.astype(np.uint16))

        # run Cellpose CLI
        cmd = f"cellpose --image_path {patch_path} {CELLPose_ARGS} --savedir {PATCH_DIR}"
        try:
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print(f"[!] Cellpose failed on {name}: {e}")
            raise

        if not os.path.isfile(mask_path):
            raise RuntimeError(f"Missing mask output for {name}")

        mask = tifffile.imread(mask_path).astype(np.uint16)

        # merge: since patch_infos sorted with farthest-first, closer-to-center patches come later and overwrite
        merged[zsl, ysl, xsl] = mask

    # save final merged volume
    tifffile.imwrite(OUTPUT_TIF, merged)
    print(f"[+] Saved merged segmentation to {OUTPUT_TIF}")

    # cleanup
    shutil.rmtree(PATCH_DIR)
    print("[+] Cleaned up intermediate patches.")

if __name__ == "__main__":
    main()
