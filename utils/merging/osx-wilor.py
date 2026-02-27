import os
import numpy as np
import pickle
import trimesh
import argparse
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
import glob

def parseargs():
    parser = argparse.ArgumentParser(
        description="Merge WiLoR hand meshes with SMPLX body mesh accurately"
    )
    parser.add_argument(
        '--body-data',
        type=str,
        required=True,
        help='Path to the SMPLX body data pickle file'
    )
    parser.add_argument(
        '--hand-data',
        type=str,
        required=True,
        help='Path to the WiLoR hand data pickle file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='merged_mesh.obj',
        help='Path to save the merged output mesh'
    )
    parser.add_argument(
        '--gt',
        type=str,
        default='gt',
        help='Path to gt'
    )
    return parser.parse_args()

def compute_hand_size(vertices, hand_indices):
    """Compute hand bounding box diagonal size"""
    hand_verts = vertices[hand_indices]
    bbox_size = hand_verts.max(axis=0) - hand_verts.min(axis=0)
    return np.linalg.norm(bbox_size)

def align_hand_to_body_accurate(hand_verts, hand_joints, hand_cam_t,
                           body_wrist_3d, gt_hand_size, hand_type='left'):
    """
    Accurately align WiLoR hand to OSX body using extracted parameters
    """
    
    # Step 1: Apply WiLoR camera translation to get hand in camera space
    hand_cam_space = hand_verts + hand_cam_t
    joints_cam_space = hand_joints + hand_cam_t
    
    # Step 2: Calculate WiLoR hand size
    wilor_hand_size_vec = hand_cam_space.max(axis=0) - hand_cam_space.min(axis=0)
    wilor_hand_size = np.linalg.norm(wilor_hand_size_vec)
    
    # Step 3: Scale to match GT hand size (KEY FIX!)
    scale_factor = gt_hand_size / wilor_hand_size
    
    # Apply scale
    hand_scaled = hand_cam_space * scale_factor
    joints_scaled = joints_cam_space * scale_factor
    
    # Step 4: Position alignment - move hand so wrist aligns with body wrist
    hand_wrist = joints_scaled[0]  # First joint is wrist in WiLoR
    translation = body_wrist_3d - hand_wrist
    
    hand_aligned = hand_scaled + translation
    joints_aligned = joints_scaled + translation
    
    print(f"  WiLoR hand size: {wilor_hand_size:.4f}")
    print(f"  GT hand size: {gt_hand_size:.4f}")
    print(f"  Scale factor: {scale_factor:.4f}")
    print(f"  Translation: {translation}")
    
    return hand_aligned, joints_aligned

def generate_hand_faces(hand_verts, method,hand_type):
    """
    Generate faces for hand vertices
    """
    mano_path = './model/MANO_RIGHT.pkl' if hand_type == 'right' else './model/MANO_LEFT.pkl'
    with open(mano_path, 'rb') as f:
        mano_data = pickle.load(f, encoding='latin1')
        mano_faces = mano_data['f']  # MANO faces
        return mano_faces

def remap_mano_faces_to_smplx(mano_faces, hand_indices):
    """
    Remap MANO face indices to SMPLX vertex indices.
    
    MANO faces reference vertices 0-777.
    hand_indices[i] gives the SMPLX index for MANO vertex i.
    
    So MANO face [a, b, c] becomes [hand_indices[a], hand_indices[b], hand_indices[c]]
    """
    hand_indices_array = np.array(hand_indices)
    
    # Remap all face indices at once
    remapped_faces = hand_indices_array[mano_faces]
    
    return remapped_faces

def load_obj_vertices(filepath):
    """Load only vertices from OBJ file"""
    vertices = []
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(vertices, dtype=np.float32)

def procrustes_align(source, target):
    """
    Compute optimal rotation, scale, and translation to align source to target.
    Returns transformed source vertices and transformation parameters.
    
    Uses Procrustes analysis (same as your evaluation script).
    """
    n = source.shape[0]
    
    # Compute centroids
    centroid_source = np.mean(source, axis=0, keepdims=True)
    centroid_target = np.mean(target, axis=0, keepdims=True)
    
    # Center the points
    source_centered = source - centroid_source
    target_centered = target - centroid_target
    
    # Compute covariance matrix
    H = source_centered.T @ target_centered / n
    
    # SVD
    U, S, Vt = np.linalg.svd(H)
    
    # Handle reflection
    d = np.linalg.det(Vt.T @ U.T)
    if d < 0:
        Vt[-1] *= -1
        S[-1] *= -1
    
    # Optimal rotation
    R = Vt.T @ U.T
    
    # Optimal scale
    var_source = np.var(source, axis=0, ddof=0).sum()
    scale = np.sum(S) / var_source
    
    # Optimal translation
    t = centroid_target.T - scale * R @ centroid_source.T
    
    # Apply transformation
    aligned = (scale * R @ source.T).T + t.T
    
    return aligned, scale, R, t.flatten()

def merge_hands_with_body_accurate(body_data_path, hand_data_path, output_path, gt_path):
    """
    Accurate merging using proper SMPLX joint and vertex information
    """
    
    # Load data
    with open(body_data_path, 'rb') as f:
        body_data = pickle.load(f)
    
    with open(hand_data_path, 'rb') as f:
        hand_data = pickle.load(f)
    
    print("=" * 60)
    print("MERGING HANDS WITH BODY")
    print("=" * 60)

    # Find GT mesh file from directory
    if os.path.isdir(gt_path):
        gt_mesh_files = sorted(glob.glob(os.path.join(gt_path, '*.obj')))
        if not gt_mesh_files:
            raise FileNotFoundError(f"No .obj files found in GT directory: {gt_path}")
        gt_mesh_file = gt_mesh_files[0]
    else:
        gt_mesh_file = gt_path
    
    gt_verts = load_obj_vertices(gt_mesh_file)

    # Extract body data
    body_verts = body_data['vertices']
    body_faces = body_data['faces']
    body_cam_trans = body_data['cam_trans']

    has_left_hand = bool(hand_data.get('left_hands') and len(hand_data['left_hands']) > 0)
    has_right_hand = bool(hand_data.get('right_hands') and len(hand_data['right_hands']) > 0)
    print(f"WiLoR hands available - Left: {has_left_hand}, Right: {has_right_hand}")
    
    print(f"\nBody mesh loaded:")
    print(f"  Vertices: {body_verts.shape}")
    print(f"  Camera translation: {body_cam_trans}")
    print(f"  Z range: [{body_verts[:, 2].min():.3f}, {body_verts[:, 2].max():.3f}]")
    
    # Get accurate wrist positions and hand regions
    left_wrist_3d = body_data['left_wrist_3d']
    right_wrist_3d = body_data['right_wrist_3d']
    left_hand_indices = body_data['left_hand_vertex_indices']
    right_hand_indices = body_data['right_hand_vertex_indices']
    left_hand_bbox = body_data['left_hand_bbox_3d']
    right_hand_bbox = body_data['right_hand_bbox_3d']
    
    all_indices = set(range(len(body_verts)))

    print(f"\nWrist positions:")
    print(f"  Left: {left_wrist_3d}")
    print(f"  Right: {right_wrist_3d}")
    print(f"  Left hand size: {body_data['left_hand_size_3d']:.3f}")
    print(f"  Right hand size: {body_data['right_hand_size_3d']:.3f}")
    
    gt_left_hand_size = compute_hand_size(gt_verts, left_hand_indices)
    gt_right_hand_size = compute_hand_size(gt_verts, right_hand_indices)

    all_indices_list = sorted(all_indices)
    body_for_align = body_verts[all_indices_list]
    gt_for_align = gt_verts[all_indices_list]
    # Create mask for vertices to keep/replace
    # Remove hands from body mesh only if we have replacement from WiLoR
    body_mask = np.ones(len(body_verts), dtype=bool)
    hand_indices_to_remove = []
    
    _, scale, R_mat, t_vec = procrustes_align(body_for_align, gt_for_align)
    
    print(f"  Scale: {scale:.4f}")
    print(f"  Rotation (approx angle): {np.degrees(np.arccos(np.clip((np.trace(R_mat) - 1) / 2, -1, 1))):.1f}°")
    print(f"  Translation: {t_vec}")
    
    if has_left_hand:
        hand_indices_to_remove.extend(left_hand_indices)
    if has_right_hand:
        hand_indices_to_remove.extend(right_hand_indices)
    
    body_without_hands_verts = body_verts[body_mask]

    body_only_indices = sorted(all_indices - set(hand_indices_to_remove))
    
    face_mask = np.ones(len(body_faces), dtype=bool)
    for i, face in enumerate(body_faces):
        if any(v in hand_indices_to_remove for v in face):
            face_mask[i] = False

    body_faces_no_hands = body_faces[face_mask]

    final_verts = body_verts.copy()
    final_faces = [body_faces_no_hands]
    combined_verts = np.stack(final_verts)

    # Process left hand
    if has_left_hand:
        print(f"\nProcessing left hand:")
        left_hand = hand_data['left_hands'][0]
        left_joints = hand_data['left_joints'][0]
        left_cam_t = hand_data['left_cam_t'][0] if hand_data['left_cam_t'] else np.zeros(3)
        
        print(f"  Original vertices: {left_hand.shape}")
        print(f"  Original range: [{left_hand.min():.3f}, {left_hand.max():.3f}]")
        print(f"  Camera translation: {left_cam_t}")
        
        aligned_left, aligned_left_joints = align_hand_to_body_accurate(
            left_hand, left_joints, left_cam_t,
            left_wrist_3d, gt_left_hand_size, 'left'
        )
        
        print(f"  Aligned range: [{aligned_left.min():.3f}, {aligned_left.max():.3f}]")
        print(f"  Aligned wrist: {aligned_left_joints[0]}")
        print(f"  Distance to target: {np.linalg.norm(aligned_left_joints[0] - left_wrist_3d):.6f}")
        
        for mano_idx, smplx_idx in enumerate(left_hand_indices):
            final_verts[smplx_idx] = aligned_left[mano_idx]

        print(len(final_verts))
        
        left_hand_faces = generate_hand_faces(aligned_left, method='',hand_type='left')

        remapped_left_faces = remap_mano_faces_to_smplx(left_hand_faces, left_hand_indices)
        final_faces.append(remapped_left_faces)
        print(f"  Generated {len(left_hand_faces)} faces for left hand")
        
    # Process right hand
    if has_right_hand:
        print(f"\nProcessing right hand:")
        right_hand = hand_data['right_hands'][0]
        right_joints = hand_data['right_joints'][0]
        right_cam_t = hand_data['right_cam_t'][0] if hand_data['right_cam_t'] else np.zeros(3)
        
        print(f"  Original vertices: {right_hand.shape}")
        print(f"  Original range: [{right_hand.min():.3f}, {right_hand.max():.3f}]")
        print(f"  Camera translation: {right_cam_t}")
        
        aligned_right, aligned_right_joints = align_hand_to_body_accurate(
            right_hand, right_joints, right_cam_t,
            right_wrist_3d, gt_right_hand_size, 'right'
        )
        
        print(f"  Aligned range: [{aligned_right.min():.3f}, {aligned_right.max():.3f}]")
        print(f"  Aligned wrist: {aligned_right_joints[0]}")
        print(f"  Distance to target: {np.linalg.norm(aligned_right_joints[0] - right_wrist_3d):.6f}")
        for mano_idx, smplx_idx in enumerate(right_hand_indices):
            # print(smplx_idx,aligned_right[mano_idx])
            final_verts[smplx_idx] = aligned_right[mano_idx]
        print(len(final_verts))

        right_hand_faces = generate_hand_faces(aligned_right, method='',hand_type='right')
        
        remapped_right_faces = remap_mano_faces_to_smplx(right_hand_faces, right_hand_indices)
        final_faces.append(remapped_right_faces)
        print(f"  Generated {len(right_hand_faces)} faces for right hand")
        

    combined_verts = final_verts
    
    if final_faces:
        final_faces = np.vstack(final_faces)
    else:
        final_faces = np.array([])
    
    mesh = trimesh.Trimesh(vertices=combined_verts, faces=final_faces)
    mesh.update_faces(mesh.nondegenerate_faces(height=1e-8))
    mesh.update_faces(mesh.unique_faces())
    # mesh.remove_unreferenced_vertices()
    # Export
    mesh.export(output_path.replace('.obj', '_out.obj'))
    print(f"\nTotal merged vertices: {len(mesh.vertices)}")
    print(f"Intermediate merged mesh saved to {output_path.replace('.obj', '_vis_mesh.obj')}")
    
    # Also save complete body for reference
    output_body = output_path.replace('.obj', '_original_body.obj')
    body_mesh_full = trimesh.Trimesh(vertices=body_verts, faces=body_faces)
    body_mesh_full.export(output_body)
    print(f"Original body saved to {output_body}")
    
    return mesh

if __name__ == '__main__':

    args = parseargs()

    gt_mesh_path = args.gt

    if os.path.isdir(args.body_data) and os.path.isdir(args.hand_data):
        os.makedirs(args.output, exist_ok=True)
        # Find all body files
        body_pattern = os.path.join(args.body_data, "", "*_body_data.pkl")
        body_files = sorted(glob.glob(body_pattern))
        
        # Find all hand files
        hand_pattern = os.path.join(args.hand_data, "*_hand_data.pkl")
        hand_files = sorted(glob.glob(hand_pattern))
        
        if not body_files:
            print(f"No body files found in {body_pattern}")
            exit(1)
        if not hand_files:
            print(f"No hand files found in {hand_pattern}")
            exit(1)
        
        print(f"Found {len(body_files)} body files and {len(hand_files)} hand files")
        if len(body_files) != len(hand_files):
            print("Warning: Mismatched number of files. Attempting to match by filename...")       
            # Create a mapping of hand files to body files
            matched_pairs = []
            for body_file in body_files:
                body_basename = os.path.basename(body_file).replace('_body_data.pkl', '')
                matching_hand = None
                for hand_file in hand_files:
                    hand_basename = os.path.basename(hand_file).replace('_hand_data.pkl', '')
                    # Try direct match first
                    if body_basename == hand_basename:
                        matching_hand = hand_file
                        break
                    # Try match with _0 suffix for hand files
                    if body_basename + '_0' == hand_basename:
                        matching_hand = hand_file
                        break
                    # Try match removing _0 from body files
                    if body_basename.replace('_0', '') == hand_basename:
                        matching_hand = hand_file
                        break
                if matching_hand:
                    matched_pairs.append((body_file, matching_hand))
                else:
                    print(f"Warning: No matching hand file found for {body_file}, saving original body...")
                    # Save original body as is
                    with open(body_file, 'rb') as f:
                        body_data = pickle.load(f)
                    body_mesh_full = trimesh.Trimesh(vertices=body_data['vertices'], faces=body_data['faces'])
                    output_file = os.path.join(args.output, f"merged_{os.path.basename(body_file).replace('.pkl', '_out.obj')}")
                    body_mesh_full.export(output_file)
                    print(f"Original body saved to {output_file}")
                    

            # Update the file lists to use matched pairs
            body_files = [pair[0] for pair in matched_pairs]
            hand_files = [pair[1] for pair in matched_pairs]
            print(f"Successfully matched {len(matched_pairs)} pairs")

        
        # Process pairs
        for i, (body_file, hand_file) in enumerate(zip(body_files, hand_files)):
            print(f"\nProcessing pair {i+1}/{min(len(body_files), len(hand_files))}")
            print(f"Body: {body_file}")
            print(f"Hand: {hand_file}")
            
            # Create output filename
            base_name = f"merged_{os.path.basename(hand_file)}.obj"
            if args.output.endswith('.obj'):
                output_file = args.output.replace('.obj', f'_{os.path.basename(hand_file)}.obj')
            else:
                output_file = os.path.join(args.output, base_name)
            
            try:
                print(f"Output will be saved to: {output_file.split('_hand_data') }")
                merge_hands_with_body_accurate(body_file, hand_file, output_file.split('_hand_data')[0] + '.obj', gt_mesh_path)
            except Exception as e:
                print(f"Error processing pair {i}: {e}")
                continue
    else:
        merge_hands_with_body_accurate(
            args.body_data,
            args.hand_data,
            args.output,
            gt_mesh_path
        )