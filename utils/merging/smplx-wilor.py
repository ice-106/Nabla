import numpy as np
import torch
import pickle
import trimesh
import smplx
import argparse
import os.path as osp
import os
import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--body', required=True, help='Path to SMPL-X NPZ file')
    parser.add_argument('--hand', required=True, help='Path to WiLoR pickle file')
    parser.add_argument('--smplx-model', required=True, help='Path to SMPL-X model folder')
    parser.add_argument('--output', default='output', help='Output mesh path')
    # parser.add_argument('--gender', default='neutral', choices=['neutral', 'male', 'female'])
    return parser.parse_args()

def get_mano_faces(hand_type):
    """Get MANO faces"""
    mano_path = './model/MANO_RIGHT.pkl' if hand_type == 'right' else './model/MANO_LEFT.pkl'
    with open(mano_path, 'rb') as f:
        mano_data = pickle.load(f, encoding='latin1')
        return mano_data['f']
        
def load_smplx_from_npz(npz_path, smplx_model_path):
    """
    Load SMPL-X parameters from NPZ and generate mesh
    """
    # Load NPZ data
    data = np.load(npz_path)
    
    # Create SMPL-X model
    model = smplx.create(
        smplx_model_path,
        model_type='smplx',
        gender='neutral',
        use_face_contour=True,
        use_pca=False
    )
    
    # Prepare parameters
    params = {
        'global_orient': torch.tensor(data['global_orient'], dtype=torch.float32),
        'body_pose': torch.tensor(data['body_pose'].reshape(1, -1), dtype=torch.float32),
        'left_hand_pose': torch.tensor(data['left_hand_pose'].reshape(1, -1), dtype=torch.float32),
        'right_hand_pose': torch.tensor(data['right_hand_pose'].reshape(1, -1), dtype=torch.float32),
        'jaw_pose': torch.tensor(data['jaw_pose'], dtype=torch.float32),
        'betas': torch.tensor(data.get('betas', np.zeros((1, 10))), dtype=torch.float32),
        'expression': torch.tensor(data.get('expression', np.zeros((1, 10))), dtype=torch.float32),
        'transl': torch.tensor(data.get('transl', np.zeros((1, 3))), dtype=torch.float32)
    }
    
    # Generate mesh
    output = model(**params)
    vertices = output.vertices.detach().cpu().numpy()[0]
    joints = output.joints.detach().cpu().numpy()[0]
    
    return vertices, model.faces, joints

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

def merge_wilor_with_npz(npz_path, wilor_pkl_path, smplx_model_path, output_path):
    """
    Merge WiLoR hands with SMPL-X from NPZ using the working approach
    """
    print("="*60)
    print("MERGING WiLoR HANDS WITH SMPL-X NPZ")
    print("="*60)
    
    # Load SMPL-X
    smplx_verts, smplx_faces, smplx_joints = load_smplx_from_npz(npz_path, smplx_model_path)
    print(f"SMPL-X loaded: {smplx_verts.shape} vertices")
    
    # Load WiLoR
    with open(wilor_pkl_path, 'rb') as f:
        wilor_data = pickle.load(f)
    
    has_left = len(wilor_data['left_hands']) > 0
    has_right = len(wilor_data['right_hands']) > 0
    
    # SMPL-X hand indices
    with open(osp.join('./model/smplx_mano_flame_correspondences/MANO_SMPLX_vertex_ids.pkl'), 'rb') as f:
        hand_vertex_idx = pickle.load(f, encoding='latin1')

    left_hand_indices  = hand_vertex_idx['left_hand']
    right_hand_indices = hand_vertex_idx['right_hand']
    # left_hand_indices  = smpl_x.hand_vertex_idx['left_hand']
    # right_hand_indices = smpl_x.hand_vertex_idx['right_hand']

    
    # Get wrist positions (joint indices in SMPL-X)
    left_wrist_3d = smplx_joints[20]  # Left wrist
    right_wrist_3d = smplx_joints[21]  # Right wrist
    
    # Step 1: Remove hand vertices from body
    all_indices = set(range(len(smplx_verts)))
    body_mask = np.ones(len(smplx_verts), dtype=bool)
    hand_indices_to_remove = set()
    
    if has_left:
        hand_indices_to_remove.update(left_hand_indices)
    if has_right:
        hand_indices_to_remove.update(right_hand_indices)
    
    body_without_hands_verts = smplx_verts[body_mask]
    body_only_indices = sorted(all_indices - hand_indices_to_remove)
    
    face_mask = np.ones(len(smplx_faces), dtype=bool)
    for i, face in enumerate(smplx_faces):
        if any(v in hand_indices_to_remove for v in face):
            face_mask[i] = False
    
    body_faces_no_hands = smplx_faces[face_mask]
    body_only_verts = body_without_hands_verts
    
    final_verts = smplx_verts.copy()
    final_faces = [body_faces_no_hands]
    combined_verts = np.stack(final_verts)
    
    if has_left:
        print("\nProcessing LEFT hand...")
        left_verts = wilor_data['left_hands'][0]
        left_joints = wilor_data['left_joints'][0]
        left_cam_t = wilor_data['left_cam_t'][0] if wilor_data['left_cam_t'] else np.zeros(3)
        
        # Simple alignment: just translate to wrist position
        left_wrist_wilor = left_joints[0] + left_cam_t
        translation = left_wrist_3d - left_wrist_wilor
        aligned_left = left_verts + left_cam_t + translation

        for mano_idx, smplx_idx in enumerate(left_hand_indices):
            final_verts[smplx_idx] = aligned_left[mano_idx]

        print(len(final_verts))
        
        # Add MANO faces for left hand
        left_faces = get_mano_faces('left')
        remapped_left_faces = remap_mano_faces_to_smplx(left_faces, left_hand_indices)
        final_faces.append(remapped_left_faces)
        print(f"  Added {len(aligned_left)} vertices, {len(left_faces)} faces")
    
    # Step 5: Add WiLoR right hand
    if has_right:
        print("\nProcessing RIGHT hand...")
        right_verts = wilor_data['right_hands'][0]
        right_joints = wilor_data['right_joints'][0]
        right_cam_t = wilor_data['right_cam_t'][0] if wilor_data['right_cam_t'] else np.zeros(3)
        
        # Simple alignment
        right_wrist_wilor = right_joints[0] + right_cam_t
        translation = right_wrist_3d - right_wrist_wilor
        aligned_right = right_verts + right_cam_t + translation
        
        for mano_idx, smplx_idx in enumerate(right_hand_indices):
            # print(smplx_idx,aligned_right[mano_idx])
            final_verts[smplx_idx] = aligned_right[mano_idx]
        print(len(final_verts))
        
        # Add MANO faces for right hand
        right_faces = get_mano_faces('right')
        remapped_right_faces = remap_mano_faces_to_smplx(right_faces, right_hand_indices)
        final_faces.append(remapped_right_faces)
        print(f"  Added {len(aligned_right)} vertices, {len(right_faces)} faces")
    
    # Step 6: Create final mesh
    combined_verts = final_verts
    combined_faces = np.vstack(final_faces) 
    
    mesh = trimesh.Trimesh(vertices=combined_verts, faces=combined_faces)
    
    # Clean up
    mesh.update_faces(mesh.nondegenerate_faces(height=1e-8))
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    # Export
    
    # Export
    mesh.export(output_path)
    
    print("\n" + "="*60)
    print(f"Saved to: {output_path}")
    print(f"Total vertices: {len(mesh.vertices)}")
    print(f"Total faces: {len(mesh.faces)}")
    print("="*60)
    
    return mesh


if __name__ == "__main__":
    
    args = parse_args()
    
    os.makedirs(args.output, exist_ok=True)

    if os.path.isdir(args.body) and os.path.isdir(args.hand):
        # Find all body files
        body_pattern = os.path.join(args.body, "smplx", "*.npz")
        body_files = sorted(glob.glob(body_pattern))
        
        # Find all hand files
        hand_pattern = os.path.join(args.hand, "*_hand_data.pkl")
        hand_files = sorted(glob.glob(hand_pattern))
        
        if not body_files:
            print(f"No body files found in {body_pattern}")
            exit(1)
        if not hand_files:
            print(f"No hand files found in {hand_pattern}")
            exit(1)
        
        print(f"Found {len(body_files)} body files and {len(hand_files)} hand files")
        if len(body_files) != len(hand_files):
            print("Warning: Number of body files and hand files do not match. Processing up to the minimum count.")
            matched_pairs = []
            for body_file in body_files:
                base_name = os.path.basename(body_file).replace('.npz', '')
                matching_hand = None
                for hand_file in hand_files:
                    hand_basename = os.path.basename(hand_file).replace('_hand_data.pkl', '')
                    if base_name in os.path.basename(hand_file):
                        matching_hand = hand_file
                        break
                    if base_name + '_0' == hand_basename:
                        matching_hand = hand_file
                        break
                if matching_hand:
                    matched_pairs.append((body_file, matching_hand))
                else:
                    print(f"No matching hand file found for body file {body_file}")
                    smplx_verts, smplx_faces, smplx_joints = load_smplx_from_npz(body_file, args.smplx_model)
                    mesh = trimesh.Trimesh(vertices=smplx_verts, faces=smplx_faces)
                    outname = f"merged_{base_name}.obj"
                    body_only_output = os.path.join(args.output, outname)
                    mesh.export(body_only_output)
                    print(f"Saved body-only mesh to: {body_only_output}")

                    
            body_files = [pair[0] for pair in matched_pairs]
            hand_files = [pair[1] for pair in matched_pairs]
        
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
                merge_wilor_with_npz(body_file, hand_file, args.smplx_model, output_file.split('_hand')[0] + '.obj')
            except Exception as e:
                print(f"Error processing pair {i}: {e}")
                continue
    else:
        # Original single file processing
        merge_wilor_with_npz(
            args.body,
            args.hand,
            args.smplx_model,
            args.output
        )
