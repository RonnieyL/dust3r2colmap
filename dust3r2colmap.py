import argparse
import torch
import numpy as np
from pathlib import Path
import os
import shutil
import cv2

# --- Essential DUSt3R/MASt3R Imports ---
try:
    from mast3r.model import AsymmetricMASt3R
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    from dust3r.utils.device import to_numpy
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    from dust3r.utils.geometry import inv, geotrf
except ImportError as e:
    print(f"Error importing DUSt3R/MASt3R components: {e}")
    print("Please ensure DUSt3R/MASt3R is installed and accessible.")
    exit(1)

# --- COLMAP Utilities ---
from utils.colmap_utils import (
    init_filestructure, load_images as load_images_utils,
    save_extrinsic, save_intrinsics, save_points3D,
    save_images_and_masks, compute_co_vis_masks,
    project_points, normalize_depth
)

def main(args):
    device = torch.device(args.device)

    # 1. Load Model
    print("Loading MASt3R model...")
    model = AsymmetricMASt3R.from_pretrained(args.ckpt_path).to(device)
    print("Model loaded.")

    # 2. Load Images
    print(f"Loading images from: {args.image_dir}")
    image_data, (org_width, org_height) = load_images_utils(args.image_dir, size=args.image_size, verbose=True)
    image_files = [img['instance'] for img in image_data]
    processed_shapes = [(img['img'].shape[2], img['img'].shape[3]) for img in image_data]

    # 3. Create Pairs and Run Inference
    print("Creating pairs and running inference...")
    pairs = make_pairs(image_data, scene_graph='complete', symmetrize=True) #For images [A,B,C], creates pairs: (A,B), (A,C), (B,C)
    output = inference(pairs, model, device, batch_size=1, verbose=True)
    print("Inference complete.")

    # 4. Global Alignment
    print("Running global alignment...")
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    scene.compute_global_alignment(init="mst", niter=args.align_iter, schedule=args.schedule, lr=args.lr)
    print("Alignment complete.")

    # 5. Extract Geometry from Scene Object
    print("Extracting geometry...")
    with torch.no_grad():
        poses_w2c = inv(scene.get_im_poses()).cpu().numpy()
        poses_c2w = inv(poses_w2c)
        focals = scene.get_focals().cpu().numpy()
        try:
            pps = scene.get_principal_points().cpu().numpy()
        except AttributeError:
            print("Principal points not directly available from scene, assuming image center.")
            pps = np.array([[w/2.0, h/2.0] for h, w in processed_shapes])

        pts3d_list = scene.get_pts3d()
        conf_list = scene.get_conf()

    print("Geometry extracted.")

    # 6. Initialize COLMAP File Structure
    print("Initializing COLMAP file structure...")
    save_path, sparse_0_path, sparse_1_path = init_filestructure(
        Path(args.output_dir), n_views=args.n_views if hasattr(args, 'n_views') else None
    )

    # 7. Save Camera Extrinsics
    print("Saving camera extrinsics...")
    save_extrinsic(sparse_0_path, poses_w2c, image_files, '.jpg')

    # 8. Save Camera Intrinsics
    print("Saving camera intrinsics...")
    save_intrinsics(sparse_0_path, focals, (org_width, org_height), 
                   (processed_shapes[0][0], processed_shapes[0][1], 3))

    # 9. Compute Co-visibility Masks
    print("Computing co-visibility masks...")
    sorted_conf_indices = np.argsort([conf.mean().item() for conf in conf_list])[::-1]
    overlapping_masks = compute_co_vis_masks(
        sorted_conf_indices,
        to_numpy(conf_list),
        to_numpy(pts3d_list),
        np.array([[f, f, pps[i][0], pps[i][1]] for i, f in enumerate(focals)]),
        poses_w2c,
        (len(image_files), processed_shapes[0][1], processed_shapes[0][0], 3),
        depth_threshold=args.depth_threshold
    )

    # 10. Save Points3D
    print("Saving 3D points...")
    save_points3D(
        sparse_0_path,
        to_numpy(image_data),
        to_numpy(pts3d_list),
        to_numpy(conf_list),
        overlapping_masks,
        use_masks=True,
        save_all_pts=True,
        save_txt_path=sparse_0_path,
        depth_threshold=args.depth_threshold,
        max_pts_num=args.max_points
    )

    # 11. Save Images and Masks
    print("Saving images and masks...")
    save_images_and_masks(
        sparse_0_path,
        args.n_views if hasattr(args, 'n_views') else None,
        to_numpy(image_data),
        overlapping_masks,
        image_files,
        '.jpg'
    )

    print("COLMAP conversion complete!")
    print(f"Output saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DUSt3R/MASt3R output to COLMAP format.")
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing input images.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save COLMAP results.')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to MASt3R model checkpoint.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu).')
    parser.add_argument('--image_size', type=int, default=512, help='Image size used by DUSt3R/MASt3R.')
    parser.add_argument('--depth_threshold', type=float, default=0.1, help='Depth threshold for co-visibility.')
    parser.add_argument('--max_points', type=int, default=100000, help='Maximum number of points in points3D.txt.')
    parser.add_argument('--n_views', type=int, default=None, help='Number of views for reconstruction.')
    
    # Alignment args
    parser.add_argument('--align_iter', type=int, default=300, help='Number of iterations for global alignment.')
    parser.add_argument('--schedule', type=str, default='cosine', help='Learning rate schedule for alignment.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for alignment.')

    args = parser.parse_args()
    main(args)