#!/usr/bin/env python3
"""
Revised 3D Cellpose-SAM inference on large volumes with block-wise processing and overlap-based stitching.

Changes made:
1. Updated patch size to (512, 512, 512) and overlap to (64, 64, 64) configurable at top.
2. Stitching now only resolves conflicts in overlapping regions (where patches overlap).
3. Replaced global majority voting with a local 70% IoU-based merge: when two patches assign the same voxel to different labels,
   we compute overlap between corresponding label instances; if >70% overlap, unify labels across the volume.
4. Switched from skimage.measure.label to scipy.ndimage.label to avoid scikit-image dependency.

"""
import os
import tifffile
import numpy as np
import subprocess
from tqdm import tqdm
import shutil
from scipy.ndimage import label  # use scipy instead of scikit-image

# ==== CONFIGURATION ====
INPUT_TIF = "/scratch/msa6093/AAA392_gut_organ_image.tif"
PATCH_DIR = "/scratch/msa6093/cellpose_patches_iou"
OUTPUT_TIF = "/scratch/msa6093/AAA392_gut_organ_segmented_iou.tif"

# Patch parameters (Z, Y, X) and overlap
PATCH_SIZE = (512, 512, 512)
OVERLAP = (64, 64, 64)  # overlap only used for stitching

# Cellpose CLI arguments
CELLPose_ARGS = "--do_3D --use_gpu --min_size 20 --flow_threshold 0.0 --save_tif --no_npy"

# Helpers
def compute_ranges(dim, patch, overlap):
    """Compute start-stop indices so that patches cover the volume with given overlap."""
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


def generate_patch_slices(vol_shape, patch_size, overlap):
    Z, Y, X = vol_shape
    zs = compute_ranges(Z, patch_size[0], overlap[0])
    ys = compute_ranges(Y, patch_size[1], overlap[1])
    xs = compute_ranges(X, patch_size[2], overlap[2])
    patch_infos = []
    for iz, (z0, z1) in enumerate(zs):
        for iy, (y0, y1) in enumerate(ys):
            for ix, (x0, x1) in enumerate(xs):
                name = f"patch_{iz}_{iy}_{ix}"
                slicer = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
                patch_infos.append({"name": name, "slicer": slicer})
    return patch_infos

# Main
def main():
    if not os.path.isfile(INPUT_TIF):
        raise FileNotFoundError(f"Input TIFF not found: {INPUT_TIF}")

    # Prepare directories
    if os.path.exists(PATCH_DIR):
        shutil.rmtree(PATCH_DIR)
    os.makedirs(PATCH_DIR, exist_ok=True)

    print("[+] Loading input volume")
    volume = tifffile.imread(INPUT_TIF)
    vol_shape = volume.shape
    print(f"[+] Volume shape: {vol_shape}")

    # Compute patch regions
    patch_infos = generate_patch_slices(vol_shape, PATCH_SIZE, OVERLAP)
    print(f"[+] Total patches: {len(patch_infos)}")

    # Store masks for each patch
    masks = {}

    for info in tqdm(patch_infos, desc="Processing patches"):
        name = info['name']
        zsl, ysl, xsl = info['slicer']
        patch = volume[zsl, ysl, xsl]
        patch_path = os.path.join(PATCH_DIR, f"{name}.tif")
        mask_path = os.path.join(PATCH_DIR, f"{name}_cp_masks.tif")

        # Save patch and run Cellpose
        tifffile.imwrite(patch_path, patch.astype(np.uint16))
        cmd = f"cellpose --image_path {patch_path} {CELLPose_ARGS} --savedir {PATCH_DIR}"
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        if not os.path.isfile(mask_path):
            raise RuntimeError(f"Mask missing for {name}")

        # Load mask and store
        masks[name] = tifffile.imread(mask_path).astype(np.uint16)

    # Initialize merged volume and a map of global label IDs
    merged = np.zeros(vol_shape, dtype=np.uint16)
    next_label = 1
    label_map = {}  # (patch, local_label) -> global_label

    # Iterate patches in order, merging overlaps by IoU > 0.7
    for info in tqdm(patch_infos, desc="Stitching patches"):
        name = info['name']
        zsl, ysl, xsl = info['slicer']
        mask = masks[name]

        # Relabel to ensure contiguous labels in patch
        mask, _ = label(mask > 0)

        # For each label in this patch
        for local_lbl in np.unique(mask):
            if local_lbl == 0:
                continue
            # Create binary volume of this instance in global coords
            instance = (mask == local_lbl)
            # Extract region in merged
            region = merged[zsl, ysl, xsl] > 0
            overlap = instance & region
            if overlap.sum() > 0:
                # Compute IoU with existing labels in overlap
                existing_labels = np.unique(merged[zsl, ysl, xsl][overlap])
                for gl in existing_labels:
                    existing_inst = (merged[zsl, ysl, xsl] == gl)
                    iou = overlap.sum() / (existing_inst & instance).sum()
                    if iou >= 0.7:
                        # Assign this instance to existing label
                        label_map[(name, local_lbl)] = gl
                        break
                else:
                    # No high-IoU match: assign new global label
                    label_map[(name, local_lbl)] = next_label
                    next_label += 1
            else:
                # No overlap: new instance
                label_map[(name, local_lbl)] = next_label
                next_label += 1

        # Write global labels into merged
        for (patch_name, lbl), gl in label_map.items():
            if patch_name != name:
                continue
            instance = (mask == lbl)
            merged[zsl, ysl, xsl][instance] = gl

    # Save result
    tifffile.imwrite(OUTPUT_TIF, merged)
    print(f"[+] Saved final segmentation to {OUTPUT_TIF}")

    # Cleanup
    shutil.rmtree(PATCH_DIR)
    print("[+] Cleaned up intermediate patches.")

if __name__ == "__main__":
    main()
