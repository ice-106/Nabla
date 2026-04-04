import glob
import os
import torch
import pandas as pd
import argparse
import numpy as np
import pickle

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate MPVPE between two folders containing two set of mesh sequences (OBJ) with separate body/hand scoring"
    )
    parser.add_argument(
        "--gt", required=True, help="Folder containing sub-folders of ground-truth OBJ frames"
    )
    parser.add_argument(
        "--pred", required=True, help="Folder containing sub-folders of predicted OBJ frames"
    )
    parser.add_argument(
        "--ext", default="obj", help="File extension to look for (default: obj)"
    )
    parser.add_argument(
        "--outdir", default=".", help="Folder of the output csv reporting the Mean Per Vertices Position Error (MPVPE)"
    )
    parser.add_argument(
        "--model", default="OSX", help="Extraction model name for folder management"
    )
    parser.add_argument(
        "--smplx-segmentation", type=str, default=None,
        help="Path to SMPLX vertex segmentation file (pickle with left_hand_vertex_indices, right_hand_vertex_indices)"
    )
    parser.add_argument(
        "--body-data-dir", type=str, default=None,
        help="Directory containing body_data.pkl files with hand indices"
    )
    parser.add_argument(
        "--visualize-sample", action="store_true", help="Create visualization of vertex regions for first mesh"
    )
    
    args = parser.parse_args()
    return args

def load_smplx_segmentation(segmentation_path=None, body_data_path=None):
    """
    Load SMPLX vertex segmentation.
    
    Returns dict with:
        - left_hand_indices: list of vertex indices for left hand
        - right_hand_indices: list of vertex indices for right hand  
        - body_indices: list of vertex indices for body (everything except hands)
    """
    
    if body_data_path and os.path.exists(body_data_path):
        # Load from body_data.pkl (your format)
        with open(body_data_path, 'rb') as f:
            body_data = pickle.load(f)
        
        left_hand_indices = list(body_data.get('left_hand_vertex_indices', []))
        right_hand_indices = list(body_data.get('right_hand_vertex_indices', []))
        total_verts = body_data.get('vertices', np.zeros((10475, 3))).shape[0]
        
        # Body = all vertices except hands
        hand_indices_set = set(left_hand_indices) | set(right_hand_indices)
        body_indices = [i for i in range(total_verts) if i not in hand_indices_set]
        
        print(f"Loaded segmentation from body_data.pkl:")
        print(f"  Left hand: {len(left_hand_indices)} vertices")
        print(f"  Right hand: {len(right_hand_indices)} vertices")
        print(f"  Body: {len(body_indices)} vertices")
        
        return {
            'left_hand_indices': left_hand_indices,
            'right_hand_indices': right_hand_indices,
            'body_indices': body_indices
        }
    
    elif segmentation_path and os.path.exists(segmentation_path):
        # Load from dedicated segmentation file
        with open(segmentation_path, 'rb') as f:
            seg_data = pickle.load(f)
        return seg_data
    
    else:
        # Use standard SMPLX segmentation
        # These are the official SMPLX vertex indices
        print("Using standard SMPLX segmentation (you should verify these match your data)")
        return get_standard_smplx_segmentation()

def get_standard_smplx_segmentation():
    """
    Returns standard SMPLX vertex segmentation.
    
    SMPLX has 10475 vertices:
    - Body/face: various scattered indices
    - Left hand: 778 vertices (scattered)
    - Right hand: 778 vertices (scattered)
    
    NOTE: You should replace this with your actual indices from body_data.pkl
    """
    
    return {
        'left_hand_indices': list(range(4595, 5394)),  
        'right_hand_indices': list(range(7331, 8128)),
        'body_indices': list(set(range(10475)) - set(range(4595, 5394)) - set(range(7331, 8128)))
    }

def load_obj_manual(filepath):
    vertices = []
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v":
                data = [float(x) for x in parts[1:4]]  # Only xyz, ignore color
                vertices.append(data)
    return {"vertices": vertices}

def rigid_transform_3D_torch_batch(P, Q):
    """
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    in a batched manner.
    :param P: A BxNx3 matrix of points
    :param Q: A BxNx3 matrix of points
    :return: A tuple containing scale, rotation matrix, and translation vector.
    """
    _, n, dim = P.shape
    assert P.shape == Q.shape, "Matrix dimensions must match"

    # Compute centroids
    centroid_P = torch.mean(P, dim=1, keepdims=True)  # Bx1x3
    centroid_Q = torch.mean(Q, dim=1, keepdims=True)  # Bx1x3

    # Center the points
    p = P - centroid_P  # BxNx3
    q = Q - centroid_Q  # BxNx3

    # Compute the covariance matrix
    H = torch.matmul(p.transpose(1, 2), q) / n  # Bx3x3

    # SVD
    U, S, Vt = torch.linalg.svd(H)  # Bx3x3

    # Validate right-handed coordinate system
    d = torch.det(torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2)))  # B
    flip = d < 0.0
    if flip.any().item():
        Vt[flip, -1] *= -1.0
        S[flip, -1] *= -1.0

    # Optimal rotation
    R = torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2))

    varP = torch.var(P, dim=1, correction=0).sum(dim=-1)
    c = 1 / varP * torch.sum(S, dim=-1)
    c = c.unsqueeze(-1).unsqueeze(-1)

    t = -torch.matmul(c * R, centroid_P.transpose(1, 2)) + centroid_Q.transpose(1, 2)

    return c, R, t


def rigid_align_torch_batch(P, Q):
    c, R, t = rigid_transform_3D_torch_batch(P, Q)
    P2 = torch.matmul(c * R, P.transpose(1, 2)).transpose(1, 2) + t.transpose(1, 2)
    return P2


def load_mesh_sequence_from_folder(folderpath, ext="obj", expect_v=None):
    """
    Load all mesh files with extension `ext` from `folderpath`, sorted by filename.
    Only the vertex positions are extracted.
    """
    pattern = os.path.join(folderpath, f"*.{ext}")
    files = sorted(glob.glob(pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"No '*.{ext}' files found in {folderpath}")

    frames = []
    for fp in files:
        obj = load_obj_manual(fp)
        verts = obj.get("vertices", None)
        if verts is None:
            raise ValueError(f"No vertices found in {fp}")
        verts = torch.tensor(verts, dtype=torch.float32)
        if verts.dim() != 2 or verts.shape[1] != 3:
            raise ValueError(f"Vertices in {fp} have unexpected shape {verts.shape}")
        if expect_v is not None and verts.shape[0] != expect_v:
            raise AssertionError(
                f"Vertex count mismatch in {fp}: got {verts.shape[0]}, expected {expect_v}"
            )
        frames.append(verts)

    # verify consistent vertex count
    vcount = frames[0].shape[0]
    for i, f in enumerate(frames):
        if f.shape[0] != vcount:
            raise AssertionError(
                f"Inconsistent vertex count at index {i}: {f.shape[0]} vs {vcount}"
            )

    stacked = torch.stack(frames, dim=0)  # F x V x 3
    return stacked


def load_mesh_sequence_from_folder_with_suffix(folderpath, suffix="_out.obj", expect_v=None):
    """
    Load all mesh files with suffix from `folderpath`, sorted by filename.
    """
    pattern = os.path.join(folderpath, f"*{suffix}")
    files = sorted(glob.glob(pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"No '*{suffix}' files found in {folderpath}")

    frames = []
    for fp in files:
        obj = load_obj_manual(fp)
        verts = obj.get("vertices", None)
        if verts is None:
            raise ValueError(f"No vertices found in {fp}")
        verts = torch.tensor(verts, dtype=torch.float32)
        if verts.dim() != 2 or verts.shape[1] != 3:
            raise ValueError(f"Vertices in {fp} have unexpected shape {verts.shape}")
        if expect_v is not None and verts.shape[0] != expect_v:
            raise AssertionError(
                f"Vertex count mismatch in {fp}: got {verts.shape[0]}, expected {expect_v}"
            )
        frames.append(verts)

    vcount = frames[0].shape[0]
    for i, f in enumerate(frames):
        if f.shape[0] != vcount:
            raise AssertionError(
                f"Inconsistent vertex count at index {i}: {f.shape[0]} vs {vcount}"
            )

    stacked = torch.stack(frames, dim=0)  # F x V x 3
    return stacked


def compute_mpvpe_with_rigid_align(mesh_gt, mesh_out, vertex_indices=None,align_indices=None, return_per_frame=False):
    """
    Compute MPVPE between mesh_out and mesh_gt, optionally for specific vertex indices.
    
    Args:
        mesh_gt: Ground truth meshes (F, V, 3)
        mesh_out: Predicted meshes (F, V, 3)
        vertex_indices: Optional list/array of vertex indices to evaluate (for region-specific scoring)
        align_indices: Optional list/array of vertex indices to use for rigid alignment
        return_per_frame: If True, return per-frame scores
    """
    if mesh_gt.shape != mesh_out.shape:
        raise AssertionError(
            f"Shape mismatch: mesh_gt {mesh_gt.shape} vs mesh_out {mesh_out.shape}"
        )

    mesh_gt = mesh_gt.float()
    mesh_out = mesh_out.float()

    # If vertex_indices specified, extract those vertices
    if align_indices is not None:
        mesh_gt_for_align = mesh_gt[:, align_indices, :]
        mesh_out_for_align = mesh_out[:, align_indices, :]
    else:
        mesh_gt_for_align = mesh_gt
        mesh_out_for_align = mesh_out

    # Align output to ground truth (batched)
    mesh_out_align = rigid_align_torch_batch(mesh_out_for_align, mesh_gt_for_align)

    if vertex_indices is not None:
        mesh_gt_eval = mesh_gt[:, vertex_indices, :]
        mesh_out_eval = mesh_out_align[:, vertex_indices, :]
    else:
        mesh_gt_eval = mesh_gt
        mesh_out_eval = mesh_out_align

    # per-vertex euclidean distances: F x V
    dists = torch.sqrt(torch.sum((mesh_out_eval - mesh_gt_eval) ** 2, dim=-1) + 1e-12)

    # per-frame MPVPE (mean over vertices)
    per_frame = torch.mean(dists, dim=1)

    # overall MPVPE (mean over frames)
    overall = torch.mean(per_frame)

    if return_per_frame:
        return overall, per_frame
    return overall


def load_sub_folders_from_folder(folder_path):
    return sorted(os.listdir(folder_path))


def visualize_vertex_regions(mesh, body_idx, lhand_idx, rhand_idx, output_path):
    """
    Create an OBJ file with vertex colors to visualize regions.
    Body = red, left hand = green, right hand = blue, other = gray
    """
    from pathlib import Path
    
    # Take first frame
    verts = mesh[0].numpy()
    n_verts = verts.shape[0]
    
    # Create color array (default gray)
    colors = np.ones((n_verts, 3)) * 0.5
    
    # Color body vertices red
    colors[body_idx] = [1.0, 0.0, 0.0]
    
    # Color left hand green
    colors[lhand_idx] = [0.0, 1.0, 0.0]
    
    # Color right hand blue
    colors[rhand_idx] = [0.0, 0.0, 1.0]
    
    # Write OBJ with colors
    with open(output_path, 'w') as f:
        f.write("# Vertex region visualization\n")
        f.write("# Red = Body, Green = Left Hand, Blue = Right Hand\n")
        for i, (v, c) in enumerate(zip(verts, colors)):
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}\n")
    
    print(f"Saved vertex region visualization to: {output_path}")
    print(f"  Body vertices: {len(body_idx)} (red)")
    print(f"  Left hand vertices: {len(lhand_idx)} (green)")
    print(f"  Right hand vertices: {len(rhand_idx)} (blue)")


if __name__ == "__main__":
    args = parse_args()
    folders = load_sub_folders_from_folder(args.gt)
    
    # Define vertex regions
    segmentation = None
    if args.body_data_dir:
        body_data_files = sorted(glob.glob(os.path.join(args.body_data_dir, "*_body_data.pkl")))
        if body_data_files:
            segmentation = load_smplx_segmentation(body_data_path=body_data_files[0])
    elif args.smplx_segmentation:
        segmentation = load_smplx_segmentation(segmentation_path=args.smplx_segmentation)
    
    if segmentation is None:
        print("\nERROR: Could not load vertex segmentation!")
        print("Please provide either:")
        print("  --body-data-dir: Directory with body_data.pkl files")
        print("  --smplx-segmentation: Path to segmentation pickle file")
        print("\nFalling back to overall MPVPE only (no regional scores)")
        
        body_indices = None
        left_hand_indices = None
        right_hand_indices = None
        both_hands_indices = None
    else:
        body_indices = segmentation['body_indices']
        left_hand_indices = segmentation['left_hand_indices']
        right_hand_indices = segmentation['right_hand_indices']
        both_hands_indices = left_hand_indices + right_hand_indices
    
    # Dictionary for keeping track of MPVPEs
    scores = {
        "folder": [],
        "mpvpe_overall": [],
        "mpvpe_body": [],
        "mpvpe_left_hand": [],
        "mpvpe_right_hand": [],
        "mpvpe_both_hands": [],
        "per_frame_overall": [],
        "per_frame_body": [],
        "per_frame_left_hand": [],
        "per_frame_right_hand": [],
        "per_frame_both_hands": []
    }
    
    os.makedirs(args.outdir, exist_ok=True)
    
    for folder in folders:
        try:
            if args.model == "OSX":
                mesh_gt = load_mesh_sequence_from_folder(os.path.join(args.gt, folder), ext=args.ext)
                print(f"Loaded GT mesh sequence from {folder} with shape {mesh_gt.shape}")
                mesh_out = load_mesh_sequence_from_folder(
                    os.path.join(args.pred, folder), ext=args.ext, expect_v=mesh_gt.shape[1]
                )
                print(f"Loaded Predicted mesh sequence from {folder} with shape {mesh_out.shape}. Model: {args.model}")
            elif args.model == "SMPLerx":
                mesh_gt = load_mesh_sequence_from_folder(os.path.join(args.gt, folder), ext=args.ext)
                print(f"Loaded GT mesh sequence from {folder} with shape {mesh_gt.shape}")
                mesh_out = load_mesh_sequence_from_folder(
                    os.path.join(args.pred, folder, "mesh"), ext=args.ext, expect_v=mesh_gt.shape[1]
                )
                print(f"Loaded Predicted mesh sequence from {folder} with shape {mesh_out.shape}. Model: {args.model}")
            elif args.model == "OSXW":
                mesh_gt = load_mesh_sequence_from_folder(os.path.join(args.gt, folder), ext=args.ext)
                print(f"Loaded GT mesh sequence from {folder} with shape {mesh_gt.shape}")
                mesh_out = load_mesh_sequence_from_folder_with_suffix(
                    os.path.join(args.pred, folder), suffix="_out.obj", expect_v=mesh_gt.shape[1]
                )
                print(f"Loaded Predicted mesh sequence from {folder} with shape {mesh_out.shape}. Model: {args.model}")
                
        except FileNotFoundError as e:
            print(f"Info: Skipping folder '{folder}' - {e}")
            continue
        except Exception as e:
            print(f"Error: Error at '{folder}' due to error - {e}")
            raise e

        # Visualize regions for first folder if requested
        if args.visualize_sample and len(scores["folder"]) == 0:
            viz_path = os.path.join(args.outdir, "vertex_regions_visualization.obj")
            visualize_vertex_regions(mesh_gt, body_indices, left_hand_indices, right_hand_indices, viz_path)

        # Compute overall MPVPE
        overall, per_frame_overall = compute_mpvpe_with_rigid_align(
            mesh_gt, mesh_out, vertex_indices=None, return_per_frame=True
        )
        
        # Compute body MPVPE
        body_mpvpe, per_frame_body = compute_mpvpe_with_rigid_align(
            mesh_gt, mesh_out, vertex_indices=body_indices, return_per_frame=True
        )
        
        # # Compute left hand MPVPE
        left_hand_mpvpe, per_frame_lhand = compute_mpvpe_with_rigid_align(
            mesh_gt, mesh_out, vertex_indices=left_hand_indices, return_per_frame=True
        )
        # left_hand_mpvpe, per_frame_lhand = compute_mpvpe_with_rigid_align(
        #     mesh_gt, mesh_out,align_indices=left_hand_indices, return_per_frame=True
        # )
        
        # Compute right hand MPVPE
        right_hand_mpvpe, per_frame_rhand = compute_mpvpe_with_rigid_align(
            mesh_gt, mesh_out, vertex_indices=right_hand_indices, return_per_frame=True
        )
        # right_hand_mpvpe, per_frame_rhand = compute_mpvpe_with_rigid_align(
        #     mesh_gt, mesh_out,align_indices= right_hand_indices, return_per_frame=True
        # )
        
        # Compute both hands MPVPE
        both_hands_mpvpe, per_frame_bhands = compute_mpvpe_with_rigid_align(
            mesh_gt, mesh_out, vertex_indices=both_hands_indices, return_per_frame=True
        )
        
        print(f"\nResults for folder {folder}:")
        print(f"  Overall MPVPE:     {overall.item():.6f}")
        print(f"  Body MPVPE:        {body_mpvpe.item():.6f}")
        print(f"  Left hand MPVPE:   {left_hand_mpvpe.item():.6f}")
        print(f"  Right hand MPVPE:  {right_hand_mpvpe.item():.6f}")
        print(f"  Both hands MPVPE:  {both_hands_mpvpe.item():.6f}")
        
        scores["folder"].append(folder)
        scores["mpvpe_overall"].append(overall.item())
        scores["mpvpe_body"].append(body_mpvpe.item())
        scores["mpvpe_left_hand"].append(left_hand_mpvpe.item())
        scores["mpvpe_right_hand"].append(right_hand_mpvpe.item())
        scores["mpvpe_both_hands"].append(both_hands_mpvpe.item())
        scores["per_frame_overall"].append(per_frame_overall.tolist())
        scores["per_frame_body"].append(per_frame_body.tolist())
        scores["per_frame_left_hand"].append(per_frame_lhand.tolist())
        scores["per_frame_right_hand"].append(per_frame_rhand.tolist())
        scores["per_frame_both_hands"].append(per_frame_bhands.tolist())
    
    # Save results
    df = pd.DataFrame({
        "folder": scores["folder"],
        "mpvpe_overall": scores["mpvpe_overall"],
        "mpvpe_body": scores["mpvpe_body"],
        "mpvpe_left_hand": scores["mpvpe_left_hand"],
        "mpvpe_right_hand": scores["mpvpe_right_hand"],
        "mpvpe_both_hands": scores["mpvpe_both_hands"],
    })
    
    # Save summary statistics
    csv_path = os.path.join(args.outdir, "score_per_folder_regions.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved per-folder MPVPE scores to {csv_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Mean Overall MPVPE:     {df['mpvpe_overall'].mean():.6f}")
    print(f"Mean Body MPVPE:        {df['mpvpe_body'].mean():.6f}")
    print(f"Mean Left Hand MPVPE:   {df['mpvpe_left_hand'].mean():.6f}")
    print(f"Mean Right Hand MPVPE:  {df['mpvpe_right_hand'].mean():.6f}")
    print(f"Mean Both Hands MPVPE:  {df['mpvpe_both_hands'].mean():.6f}")
    print("="*60)
    
    # Save detailed per-frame data
    detailed_path = os.path.join(args.outdir, "score_per_frame_detailed.csv")
    df_detailed = pd.DataFrame(scores)
    df_detailed.to_csv(detailed_path, index=False)
    print(f"\nSaved detailed per-frame scores to {detailed_path}")