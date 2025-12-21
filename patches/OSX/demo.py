from common.utils.human_models import smpl_x
from common.utils.vis import render_mesh, save_obj, vis_keypoints
from common.utils.preprocessing import load_img, process_bbox, generate_patch_image
from common.base import Demoer
import cv2
from config import cfg
import pickle
import os
import sys
import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids', default='0')
    parser.add_argument('--img_path', type=str, default='input.png')
    parser.add_argument('--img_folder', type=str, default=None,
                        help='Process all images in folder')
    parser.add_argument('--output_folder', type=str, default='output')
    parser.add_argument('--encoder_setting', type=str,
                        default='osx_l', choices=['osx_b', 'osx_l'])
    parser.add_argument('--decoder_setting', type=str, default='normal',
                        choices=['normal', 'wo_face_decoder', 'wo_decoder'])
    parser.add_argument('--pretrained_model_path', type=str,
                        default='../pretrained_models/osx_l.pth.tar')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args


def process_image(img_path, output_folder, demoer, detector, transform):
    """Process a single image"""
    # Extract filename without extension from img_path
    frame = osp.splitext(osp.basename(img_path))[0]

    # prepare input image
    original_img = load_img(img_path)
    original_img_height, original_img_width = original_img.shape[:2]
    os.makedirs(output_folder, exist_ok=True)

    # detect human bbox with yolov5s
    with torch.no_grad():
        results = detector(original_img)
    person_results = results.xyxy[0][results.xyxy[0][:, 5] == 0]
    class_ids, confidences, boxes = [], [], []
    for detection in person_results:
        x1, y1, x2, y2, confidence, class_id = detection.tolist()
        class_ids.append(class_id)
        confidences.append(confidence)
        boxes.append([x1, y1, x2 - x1, y2 - y1])
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    vis_mesh = original_img.copy()
    vis_kpts = original_img.copy()
    for num, indice in enumerate(indices):
        bbox = boxes[indice]  # x,y,h,w
        bbox = process_bbox(bbox, original_img_width, original_img_height)
        img, img2bb_trans, bb2img_trans = generate_patch_image(
            original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape)
        img = transform(img.astype(np.float32))/255
        img = img.cuda()[None, :, :, :]
        inputs = {'img': img}
        targets = {}
        meta_info = {}

        # mesh recovery
        with torch.no_grad():
            out = demoer.model(inputs, targets, meta_info, 'test')

        mesh = out['smplx_mesh_cam'].detach().cpu().numpy()
        mesh = mesh[0]

        # Debug line to see what's available
    print("Available outputs:", out.keys())
    smplx_joints_3d = out.get('smplx_joint_cam', None)
    mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]
    cam_trans = out['cam_trans'].detach().cpu().numpy()[0]  # Important!

    # Get the joints from SMPL-X joint regressor
    # SMPL-X has a joint regressor that maps from vertices to joints
    joints_3d = np.dot(smpl_x.J_regressor, mesh)
    left_wrist_3d = joints_3d[20]
    right_wrist_3d = joints_3d[21]
    left_hand_verts = mesh[smpl_x.hand_vertex_idx['left_hand']]
    right_hand_verts = mesh[smpl_x.hand_vertex_idx['right_hand']]

    # Calculate hand bounding boxes from 3D vertices
    left_hand_bbox_3d = {
        'min': left_hand_verts.min(axis=0),
        'max': left_hand_verts.max(axis=0),
        'center': left_hand_verts.mean(axis=0)
    }

    right_hand_bbox_3d = {
        'min': right_hand_verts.min(axis=0),
        'max': right_hand_verts.max(axis=0),
        'center': right_hand_verts.mean(axis=0)
    }

    # save mesh
    save_obj(mesh, smpl_x.face, os.path.join(
        output_folder, f'{frame}_{num}.obj'))

    # Save comprehensive body data for merging
    def save_body_data_for_merge(mesh, joints_3d, joint_proj, cam_trans,
                                 left_wrist_3d, right_wrist_3d,
                                 left_hand_bbox_3d, right_hand_bbox_3d,
                                 num, output_folder):
        """Save all body mesh data needed for accurate merging"""

        body_data = {
            # Mesh data
            'vertices': mesh,
            'faces': smpl_x.face,

            # Joint data
            'joints_3d': joints_3d,  # All 53 original SMPLX joints
            'joints_2d': joint_proj,

            # Camera parameters
            'cam_trans': cam_trans,

            # Wrist specific data
            'left_wrist_3d': left_wrist_3d,
            'right_wrist_3d': right_wrist_3d,
            'left_wrist_2d': joint_proj[20][:2],  # Just x,y
            'right_wrist_2d': joint_proj[21][:2],

            # Hand regions
            'left_hand_vertex_indices': smpl_x.hand_vertex_idx['left_hand'],
            'right_hand_vertex_indices': smpl_x.hand_vertex_idx['right_hand'],
            'left_hand_bbox_3d': left_hand_bbox_3d,
            'right_hand_bbox_3d': right_hand_bbox_3d,

            # Additional useful indices
            'joint_names': smpl_x.orig_joints_name,
            'joint_regressor_indices': smpl_x.J_regressor_idx,

            # For debugging/verification
            'vertex_count': mesh.shape[0],
            'orig_img_shape': (original_img_height, original_img_width),
            'bbox': bbox,
            'focal': [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2],
                      cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]],
            'princpt': [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0],
                        cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]
        }

        # Calculate some additional useful metrics
        # Hand size in 3D (for scaling reference)
        left_hand_size = np.linalg.norm(
            left_hand_bbox_3d['max'] - left_hand_bbox_3d['min'])
        right_hand_size = np.linalg.norm(
            right_hand_bbox_3d['max'] - right_hand_bbox_3d['min'])
        body_data['left_hand_size_3d'] = left_hand_size
        body_data['right_hand_size_3d'] = right_hand_size

        # Get elbow positions for arm alignment (optional, useful for IK)
        left_elbow_3d = joints_3d[18]   # L_Elbow in orig joint set
        right_elbow_3d = joints_3d[19]  # R_Elbow in orig joint set
        body_data['left_elbow_3d'] = left_elbow_3d
        body_data['right_elbow_3d'] = right_elbow_3d

        pkl_path = os.path.join(output_folder, f'person_{num}_body_data.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(body_data, f)

        print(f"Saved comprehensive body data to {pkl_path}")
        print(f"  Body vertices: {mesh.shape}")
        print(f"  3D joints extracted: {joints_3d.shape}")
        print(f"  Left wrist at: {left_wrist_3d}")
        print(f"  Right wrist at: {right_wrist_3d}")
        print(
            f"  Left hand vertices: {len(smpl_x.hand_vertex_idx['left_hand'])}")
        print(
            f"  Right hand vertices: {len(smpl_x.hand_vertex_idx['right_hand'])}")

        return pkl_path

    # render mesh
    focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2],
             cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
    princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] +
               bbox[0], cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]
    vis_mesh = render_mesh(vis_mesh, mesh, smpl_x.face, {
        'focal': focal, 'princpt': princpt})

    # get_2d_pts
    joint_proj = out['smplx_joint_proj'].detach().cpu().numpy()[0]
    joint_proj[:, 0] = joint_proj[:, 0] / \
        cfg.output_hm_shape[2] * cfg.input_img_shape[1]
    joint_proj[:, 1] = joint_proj[:, 1] / \
        cfg.output_hm_shape[1] * cfg.input_img_shape[0]
    joint_proj = np.concatenate(
        (joint_proj, np.ones_like(joint_proj[:, :1])), 1)
    joint_proj = np.dot(
        bb2img_trans, joint_proj.transpose(1, 0)).transpose(1, 0)
    vis_kpts = vis_keypoints(vis_kpts, joint_proj)

    # Save the data
    body_data_path = save_body_data_for_merge(
        mesh, joints_3d, joint_proj, cam_trans,
        left_wrist_3d, right_wrist_3d,
        left_hand_bbox_3d, right_hand_bbox_3d,
        num, args.output_folder
    )

    # Save body data for merging
    body_data_path = save_body_data_for_merge(
        mesh, joints_3d, joint_proj, cam_trans,
        left_wrist_3d, right_wrist_3d,
        left_hand_bbox_3d, right_hand_bbox_3d,
        num, args.output_folder
    )
    # save rendered image
    cv2.imwrite(os.path.join(output_folder,
                f'{frame}_render.jpg'), vis_mesh[:, :, ::-1])
    cv2.imwrite(os.path.join(output_folder,
                f'{frame}_kpts.jpg'), vis_kpts[:, :, ::-1])


args = parse_args()
cfg.set_args(args.gpu_ids)
cudnn.benchmark = True

# load model (once)
cfg.set_additional_args(encoder_setting=args.encoder_setting,
                        decoder_setting=args.decoder_setting, pretrained_model_path=args.pretrained_model_path)
demoer = Demoer()
demoer._make_model()
model_path = args.pretrained_model_path
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))
demoer.model.eval()

# Load detector once
print('Loading YOLOv5 detector...')
detector = torch.hub.load('ultralytics/yolov5',
                          'yolov5s', pretrained=True, verbose=False)
transform = transforms.ToTensor()

# Process images
if args.img_folder:
    # Process all images in folder
    import glob
    img_paths = sorted(glob.glob(osp.join(args.img_folder, '*.jpg')) +
                       glob.glob(osp.join(args.img_folder, '*.png')))
    print(f'Processing {len(img_paths)} images from {args.img_folder}')
    for img_path in img_paths:
        print(f'Processing: {img_path}')
        process_image(img_path, args.output_folder,
                      demoer, detector, transform)
else:
    # Process single image
    process_image(args.img_path, args.output_folder,
                  demoer, detector, transform)
