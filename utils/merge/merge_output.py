import os
import numpy as np
import pickle
import trimesh
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Merge WiLoR hands with OSX body mesh")
    parser.add_argument('--body-data', type=str, required=True, help='Path to body data pickle file')
    parser.add_argument('--hand-data', type=str, required=True, help='Path to hand data pickle file')
    parser.add_argument('--output', type=str, default='merged_mesh.obj', help='Output path for merged mesh')
    return parser.parse_args()

def align_hand_to_body(hand_verts, hand_joints, hand_cam_t,
                                body_wrist_3d, body_hand_bbox,
                                body_cam_trans, hand_type='left'):
    """
    Accurately align WiLoR hand to OSX body using extracted parameters
    """
    hand_cam_space = hand_verts + hand_cam_t
    joints_cam_space = hand_joints + hand_cam_t
    
    # Use body hand bbox size as reference for scaling
    body_hand_size = body_hand_bbox['max'] - body_hand_bbox['min']
    body_hand_scale = np.linalg.norm(body_hand_size)
    
    # # WiLoR hand size
    wilor_hand_size = hand_cam_space.max(axis=0) - hand_cam_space.min(axis=0)
    wilor_hand_scale = np.linalg.norm(wilor_hand_size)
    
    scale_factor = body_hand_scale / wilor_hand_scale
    
    # Apply scale
    # hand_scaled = hand_cam_space * scale_factor
    # joints_scaled = joints_cam_space * scale_factor
    # scale factor is not so accurate right now
    hand_scaled = hand_cam_space 
    joints_scaled = joints_cam_space
    
    # Move hand so its wrist aligns with body wrist
    hand_wrist = joints_scaled[0]  # First joint is wrist in WiLoR
    translation = body_wrist_3d - hand_wrist
    
    hand_aligned = hand_scaled + translation
    joints_aligned = joints_scaled + translation
    
    print(f"  Scale factor: {scale_factor:.3f}")
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

def merge_hands_with_body(body_data_path, hand_data_path, output_path):
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
    

    print(f"\nWrist positions:")
    print(f"  Left: {left_wrist_3d}")
    print(f"  Right: {right_wrist_3d}")
    print(f"  Left hand size: {body_data['left_hand_size_3d']:.3f}")
    print(f"  Right hand size: {body_data['right_hand_size_3d']:.3f}")
    

    # Create mask for vertices to keep/replace
    # Remove hands from body mesh
    body_mask = np.ones(len(body_verts), dtype=bool)
    # Only remove hand vertices if we have replacement from WiLoR
    if has_left_hand:
        body_mask[left_hand_indices] = False
    if has_right_hand:
        body_mask[right_hand_indices] = False
    
    body_without_hands_verts = body_verts[body_mask]

    all_indices = set(range(len(body_verts)))
    body_only_indices = sorted(all_indices - set(left_hand_indices) - set(right_hand_indices))
    hand_indices_combined = sorted(set(left_hand_indices) | set(right_hand_indices))
    
    face_mask = np.ones(len(body_faces), dtype=bool)
    for i, face in enumerate(body_faces):
        if any(v in hand_indices_combined for v in face):
            face_mask[i] = False

    body_faces_no_hands = body_faces[face_mask]
    
    meshes_to_combine = []

    # Add body without hands
    body_mesh = trimesh.Trimesh(vertices=body_without_hands_verts, faces=None)
    meshes_to_combine.append(body_mesh)

    old_to_new = {}
    for new_idx, old_idx in enumerate(body_only_indices):
        old_to_new[old_idx] = new_idx

    remapped_faces = []
    for face in body_faces_no_hands:
        if all(v in old_to_new for v in face):
            new_face = [old_to_new[v] for v in face]
            remapped_faces.append(new_face)
    
    body_only_verts = body_verts[body_only_indices]
    body_only_faces = np.array(remapped_faces) if remapped_faces else np.array([])
    
    final_verts = [body_only_verts]
    final_faces = [body_only_faces] if len(body_only_faces) > 0 else []
    vertex_offset = len(body_only_verts)
    combined_verts = np.vstack(final_verts)
    combined_faces = []

    if len(body_only_faces) > 0:
        combined_faces.append(body_only_faces)
        print(f"  Body faces without hands: {len(body_only_faces)}")

    # Process left hand
    if has_left_hand:
        print(f"\nProcessing left hand:")
        left_hand = hand_data['left_hands'][0]
        left_joints = hand_data['left_joints'][0]
        left_cam_t = hand_data['left_cam_t'][0] if hand_data['left_cam_t'] else np.zeros(3)
        
        print(f"  Original vertices: {left_hand.shape}")
        print(f"  Original range: [{left_hand.min():.3f}, {left_hand.max():.3f}]")
        print(f"  Camera translation: {left_cam_t}")
        
        aligned_left, aligned_left_joints = align_hand_to_body(
            left_hand, left_joints, left_cam_t,
            left_wrist_3d, left_hand_bbox,
            body_cam_trans, 'left'
        )
        
        print(f"  Aligned range: [{aligned_left.min():.3f}, {aligned_left.max():.3f}]")
        print(f"  Aligned wrist: {aligned_left_joints[0]}")
        print(f"  Distance to target: {np.linalg.norm(aligned_left_joints[0] - left_wrist_3d):.6f}")
        
        final_verts.append(aligned_left)
        vertex_offset += len(aligned_left)
        
        left_mesh = trimesh.Trimesh(vertices=aligned_left, faces=None)
        meshes_to_combine.append(left_mesh)
        
        left_hand_faces = generate_hand_faces(aligned_left, method='',hand_type='left')
        # Offset face indices to account for body vertices
        left_faces_offset = left_hand_faces + len(body_only_verts)
        combined_faces.append(left_faces_offset)
        print(f"  Generated {len(left_hand_faces)} faces for left hand")
        
    # Process right hand
    if has_right_hand:
        print(f"\nProcessing right hand:")
        right_start_idx = len(body_only_verts)
        if has_left_hand:
            right_start_idx += len(aligned_left)
        right_hand = hand_data['right_hands'][0]
        right_joints = hand_data['right_joints'][0]
        right_cam_t = hand_data['right_cam_t'][0] if hand_data['right_cam_t'] else np.zeros(3)
        
        print(f"  Original vertices: {right_hand.shape}")
        print(f"  Original range: [{right_hand.min():.3f}, {right_hand.max():.3f}]")
        print(f"  Camera translation: {right_cam_t}")
        
        aligned_right, aligned_right_joints = align_hand_to_body(
            right_hand, right_joints, right_cam_t,
            right_wrist_3d, right_hand_bbox,
            body_cam_trans, 'right'
        )
        
        print(f"  Aligned range: [{aligned_right.min():.3f}, {aligned_right.max():.3f}]")
        print(f"  Aligned wrist: {aligned_right_joints[0]}")
        print(f"  Distance to target: {np.linalg.norm(aligned_right_joints[0] - right_wrist_3d):.6f}")
        
        final_verts.append(aligned_right)
        
        right_mesh = trimesh.Trimesh(vertices=aligned_right, faces=None)
        meshes_to_combine.append(right_mesh)

        #Generate faces from vertices
        right_hand_faces = generate_hand_faces(aligned_right, method='',hand_type='right')

        # Offset face indices
        right_faces_offset = right_hand_faces + right_start_idx
        combined_faces.append(right_faces_offset)
        print(f"  Generated {len(right_hand_faces)} faces for right hand")
        

    combined_verts = np.vstack(final_verts)
    
    if combined_faces:
        final_faces = np.vstack(combined_faces)
    else:
        final_faces = np.array([])
    
    mesh = trimesh.Trimesh(vertices=combined_verts, faces=final_faces)
    mesh.update_faces(mesh.nondegenerate_faces(height=1e-8))
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()

    # Export mesh output for visualization
    mesh.export(output_path.replace('.obj', '_vis_mesh.obj'))
    print(f"\nIntermediate merged mesh saved to {output_path.replace('.obj', '_vis_mesh.obj')}")

    # Combine all meshes
    combined_mesh = trimesh.util.concatenate(meshes_to_combine)

    # Export final combined vertices
    combined_mesh.export(output_path)
    print(f"\n" + "=" * 60)
    print(f"Merged mesh saved to {output_path}")
    print(f"Total vertices: {len(combined_mesh.vertices)}")
    
    # Also save the original body for reference
    output_body = output_path.replace('.obj', '_original_body.obj')
    body_mesh_full = trimesh.Trimesh(vertices=body_verts, faces=body_faces)
    body_mesh_full.export(output_body)
    print(f"Original body saved to {output_body}")
    
    return combined_mesh



if __name__ == '__main__':

    args = parse_args()
    merge_hands_with_body(
        args.body_data,
        args.hand_data,
        args.output
    )    
  
    # visualize_final_result(
    #     args.body_data, 
    #     args.hand_data,
    #     args.output
    # )