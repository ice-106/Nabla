import glob
import os
import torch
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate MPVPE between two folders containing two set of mesh sequences (OBJ)"
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
    
    args = parser.parse_args()
    return args

def load_obj_manual(filepath):
    vertices = []
    texture_coords = []
    normals = []
    faces = []

    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if not parts:
                continue

            prefix = parts[0]
            data = [float(x) for x in parts[1:]]

            if prefix == "v":
                vertices.append(data)
            elif prefix == "vt":
                texture_coords.append(data)
            elif prefix == "vn":
                normals.append(data)
            elif prefix == "f":
                # Faces can have different formats (e.g., v, v/vt, v/vt/vn)
                # This example assumes v/vt/vn format
                face_data = []
                for p in parts[1:]:
                    indices = [
                        int(i) - 1 for i in p.split("/")
                    ]  # OBJ indices are 1-based
                    face_data.append(indices)
                faces.append(face_data)

    return {
        "vertices": vertices,
        "texture_coords": texture_coords,
        "normals": normals,
        "faces": faces,
    }


def rigid_transform_3D_torch_batch(P, Q):
    """
    Ref: https://hunterheidenreich.com/posts/kabsch_algorithm/
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD, in a batched manner.
    :param P: A BxNx3 matrix of points
    :param Q: A BxNx3 matrix of points
    :return: A tuple containing the optimal rotation matrix, the optimal
             translation vector, and the RMSD.
    """
    _, n, dim = P.shape
    assert P.shape == Q.shape, "Matrix dimensions must match"

    # Compute centroids
    centroid_P = torch.mean(P, dim=1, keepdims=True)  # Bx1x3
    centroid_Q = torch.mean(Q, dim=1, keepdims=True)  #

    # Optimal translation
    t = centroid_Q - centroid_P  # Bx1x3
    t = t.squeeze(1)  # Bx3

    # Center the points
    p = P - centroid_P  # BxNx3
    q = Q - centroid_Q  # BxNx3

    # Compute the covariance matrix
    H = torch.matmul(p.transpose(1, 2), q) / n  # Bx3x3

    # SVD
    U, S, Vt = torch.linalg.svd(H)  # Bx3x3
    # print(S.shape)

    # Validate right-handed coordinate system
    d = torch.det(torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2)))  # B
    flip = d < 0.0
    if flip.any().item():
        Vt[flip, -1] *= -1.0
        S[flip, -1] *= -1.0

    # Optimal rotation
    R = torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2))

    # # RMSD
    # rmsd = torch.sqrt(torch.sum(torch.square(torch.matmul(p, R.transpose(1, 2)) - q), dim=(1, 2)) / P.shape[1])

    varP = torch.var(P, dim=1, correction=0).sum(dim=-1)  # align with np.var()
    c = 1 / varP * torch.sum(S, dim=-1)
    c = c.unsqueeze(-1).unsqueeze(-1)

    # print(c.shape, R.shape)
    t = -torch.matmul(c * R, centroid_P.transpose(1, 2)) + centroid_Q.transpose(1, 2)

    return c, R, t


def rigid_align_torch_batch(P, Q):
    c, R, t = rigid_transform_3D_torch_batch(P, Q)
    # print(c.shape, R.shape, t.shape)
    P2 = torch.matmul(c * R, P.transpose(1, 2)).transpose(1, 2) + t.transpose(1, 2)
    return P2


def load_mesh_sequence_from_folder(folderpath, ext="obj", expect_v=None):
    """
    Load all mesh files with extension `ext` from `folderpath`, sorted by filename.
    Only the vertex positions are extracted (lines starting with 'v').

    Returns:
      Tensor of shape (F, V, 3) where F = number of files (frames),
      V = number of vertices per mesh.

    If `expect_v` is provided, asserts that each mesh contains that many vertices.
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

def compute_mpvpe_with_rigid_align(mesh_gt, mesh_out, return_per_frame=False):
    """
    Compute Mean Per-Vertex Position Error (MPVPE) between `mesh_out` and `mesh_gt`.

    Both inputs must be tensors of shape (F, V, 3). The function performs a rigid
    Procrustes-style alignment of `mesh_out` to `mesh_gt` (per-frame, batched) using
    `rigid_align_torch_batch` and returns the MPVPE.

    If `return_per_frame` is True, also returns a tensor of per-frame MPVPE values.
    """
    if mesh_gt.shape != mesh_out.shape:
        raise AssertionError(
            f"Shape mismatch: mesh_gt {mesh_gt.shape} vs mesh_out {mesh_out.shape}"
        )

    # ensure float tensors
    mesh_gt = mesh_gt.float()
    mesh_out = mesh_out.float()

    # Align output to ground truth (batched)
    mesh_out_align = rigid_align_torch_batch(mesh_out, mesh_gt)

    # per-vertex euclidean distances: F x V
    dists = torch.sqrt(torch.sum((mesh_out_align - mesh_gt) ** 2, dim=-1) + 1e-12)

    # per-frame MPVPE (mean over vertices)
    per_frame = torch.mean(dists, dim=1)

    # overall MPVPE (mean over frames)
    overall = torch.mean(per_frame)

    if return_per_frame:
        return overall, per_frame
    return overall

def load_sub_folders_from_folder(folder_path):
    return sorted(os.listdir(folder_path))

if __name__ == "__main__":

    args = parse_args()
    folders = load_sub_folders_from_folder(args.gt)
    # dict for keeping track MPVPEs
    scores = {"folder": [],
              "score": [],
              "per_frame": []}
    for folder in folders:
        mesh_gt = load_mesh_sequence_from_folder(os.path.join(args.gt, folder), ext=args.ext)
        mesh_out = load_mesh_sequence_from_folder(
            os.path.join(args.pred, folder), ext=args.ext, expect_v=mesh_gt.shape[1]
        )

        overall, per_frame = compute_mpvpe_with_rigid_align(
            mesh_gt, mesh_out, return_per_frame=True
        )
        scores["folder"].append(folder)
        scores["score"].append(overall.item())
        scores["per_frame"].append(per_frame.tolist())
    
    df = pd.DataFrame(scores)
    df.to_csv(os.path.join(args.outdir, "score_per_folder.csv"), index=False)
    print(f"Mean Per Vertices Position Error: {df['score'].mean()}")
