import os
import cv2
import random
import numpy as np
import math
import torch
import json
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from core.model_config.DiT_DDPM import Options
from core.utils import get_rays, grid_distortion, orbit_camera_jitter

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class HGS_1M(Dataset):

    def _warn(self):
        raise NotImplementedError('this dataset is just an example and cannot be used directly, you should modify it to your own setting!')

    def __init__(self, opt: Options, training=True):
        
        self.opt = opt
        self.training = training

        self.items = np.load('./data/train_DiT.npy').tolist()

        eval_size = 1000
        total_size = len(self.items)
        
        if self.training:
            train_mask = np.ones(total_size, dtype=bool)
            stride = total_size // eval_size
            train_mask[::stride] = False
            self.items = [item for i, item in enumerate(self.items) if train_mask[i]]
        else:
            self.items = self.items[::total_size//eval_size]
            self.items = self.items[:eval_size]
        
        # default camera intrinsics
        intr = np.array([
            [
                1100.0,
                0.0,
                512.0
            ],
            [
                0.0,
                1100.0,
                512.0
            ],
            [
                0.0,
                0.0,
                1.0
            ]
        ])
        self.proj_matrix = self.getProjectionMatrix(self.opt.znear, self.opt.zfar, self.opt.FoVx, self.opt.FoVy, 
                                                   K = intr, img_h = 1024, img_w = 1024).transpose(0, 1) 


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):

        uid = self.items[idx]
        results = {}
        images = []
        masks = []
        cam_poses = []
        vid_cnt = 0
        if self.training:
            vids = [30, 37, 45, 53, 65, 85] + np.random.permutation(range(0, 89)).tolist()
        else:
            vids = [30, 37, 45, 53, 65, 85, 0, 8, 82, 60]
        if self.opt.rgb_shuffle:
            rgb_channels = list(range(3))
            random.shuffle(rgb_channels)

        camera_path = os.path.join(uid, f'camera_full_calibration.json')

        try:
            smpl_param_path = np.load(os.path.join(uid, 'smplx.npz'), allow_pickle=True)
            betas = torch.tensor(smpl_param_path['betas'], dtype=torch.float32)
            body_pose = torch.tensor(smpl_param_path['body_pose'], dtype=torch.float32)
            global_orient = torch.tensor(smpl_param_path['global_orient'], dtype=torch.float32)
            transl = torch.tensor(smpl_param_path['transl'], dtype=torch.float32)
            expression = torch.tensor(smpl_param_path['expression'], dtype=torch.float32)
            left_hand_pose = torch.tensor(smpl_param_path['left_hand_pose'], dtype=torch.float32)
            right_hand_pose = torch.tensor(smpl_param_path['right_hand_pose'], dtype=torch.float32)
            jaw_pose = torch.tensor(smpl_param_path['jaw_pose'],dtype=torch.float32)
            leye_pose = torch.tensor(smpl_param_path['leye_pose'],dtype=torch.float32)
            reye_pose = torch.tensor(smpl_param_path['reye_pose'],dtype=torch.float32)
        except:
            betas = torch.zeros((1,10),dtype=torch.float32)
            body_pose = torch.zeros((1,63),dtype=torch.float32)
            global_orient = torch.zeros((1,3),dtype=torch.float32)
            transl = torch.zeros((1,3),dtype=torch.float32)
            expression = torch.zeros((1,10),dtype=torch.float32)
            left_hand_pose = torch.zeros((1,45),dtype=torch.float32)
            right_hand_pose = torch.zeros((1,45),dtype=torch.float32)
            jaw_pose = torch.zeros((1,3),dtype=torch.float32)
            leye_pose = torch.zeros((1,3),dtype=torch.float32)
            reye_pose = torch.zeros((1,3),dtype=torch.float32)

        smpl_params = torch.cat([transl, global_orient, betas, body_pose, expression, left_hand_pose, right_hand_pose, jaw_pose, leye_pose, reye_pose], dim=-1).squeeze(0)

        for vid in vids:
            image_path = os.path.join(uid, 'rgb_map',f'{vid:04d}.jpg')
            mask_path = os.path.join(uid, 'mask_map',f'{vid:04d}.png')
            try:
                with open(image_path, 'rb') as f:
                    image = np.frombuffer(f.read(), np.uint8)
                image = torch.from_numpy(cv2.imdecode(image, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255) # [1024, 1024, 3] in [0, 1]

                with open(mask_path, 'rb') as f:
                    mask = np.frombuffer(f.read(), np.uint8)
                mask = torch.from_numpy(cv2.imdecode(mask, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255) # [1024, 1024] in [0, 1]
                if len(mask.shape) == 3:
                    mask = mask[..., 0]
                camera_params = json.load(open(camera_path, 'r'))
                camera_pose = camera_params[f'{str(vid).zfill(4)}']
                camera_matrix = torch.eye(4)
                camera_matrix[:3, :3] = torch.tensor(camera_pose['R'])
                #camera_matrix[:3, 2] *= -1
                camera_matrix[:3, 3] = torch.tensor(camera_pose['T'])
            except:
                image = torch.zeros(1024,1024,3)
                mask = torch.zeros(1024,1024)
                camera_matrix = torch.eye(4)

            if self.opt.rgb_shuffle:
                image[:, :, :3] = image[:, :, rgb_channels]

            w2c = camera_matrix
            w2c = w2c.to(dtype=torch.float32).reshape(4, 4)
            
            image = image.permute(2, 0, 1) # [4, 512, 512]
            image = image[[2,1,0]].contiguous() # bgr to rgb

            images.append(image)
            masks.append(mask.squeeze(0))
            cam_poses.append(w2c)

            vid_cnt += 1
            if vid_cnt == self.opt.num_views:
                break

        if vid_cnt < self.opt.num_views:
            print(f'[WARN] dataset {uid}: not enough valid views, only {vid_cnt} views found!')
            n = self.opt.num_views - vid_cnt
            images = images + [images[-1]] * n
            masks = masks + [masks[-1]] * n
            cam_poses = cam_poses + [cam_poses[-1]] * n
          
        images = torch.stack(images, dim=0) # [V, C, H, W]
        masks = torch.stack(masks, dim=0) # [V, H, W]
        cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]

        try:
            UV_inital_path = os.path.join(uid,'UV','smplxuv_albedo.png')
            with open(UV_inital_path, 'rb') as f:
                UV_inital = np.frombuffer(f.read(), np.uint8)
            UV_inital = torch.from_numpy(cv2.imdecode(UV_inital, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255)
        except:
            UV_inital = torch.zeros(1024,1024,3)
        UV_inital = UV_inital.permute(2,0,1).unsqueeze(0).contiguous()
        UV_inital = F.interpolate(UV_inital, size=(self.opt.input_size_h, self.opt.input_size_w), mode='bilinear', align_corners=False).squeeze(0)
        results['UV_inital'] = UV_inital

        images_input = F.interpolate(images[:self.opt.num_input_views].clone(), size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
        cam_poses_input = cam_poses[:self.opt.num_input_views].clone()

        try:
            random_sapid = random.randint(0, 3)
            sapines_input = self.condition_img(images[random_sapid],masks[random_sapid])
        except:
            sapines_input = torch.zeros(3,1024,1024)

        # apply random grid distortion to simulate 3D inconsistency
        if random.random() < self.opt.prob_grid_distortion:
            images_input[1:] = grid_distortion(images_input[1:])
            cv2.imwrite('grid_distortion_ori.png', images_input[1].permute(1, 2, 0).numpy() * 255)
            cv2.imwrite('grid_distortion.png', images_input[1].permute(1, 2, 0).numpy() * 255)
        # apply camera jittering (only to input!)
        if random.random() < self.opt.prob_cam_jitter:
            cam_poses_input[1:] = orbit_camera_jitter(cam_poses_input[1:], is_w2c=True)

        images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        sapines_input = TF.normalize(sapines_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        # resize render ground-truth images, range still in [0, 1]
        results['images_output'] = F.interpolate(images, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        results['masks_output'] = F.interpolate(masks.unsqueeze(1), size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, 1, output_size, output_size]
        
        # build rays for input views
        rays_embeddings = []
        for i in range(self.opt.num_input_views):
            rays_o, rays_d = get_rays(torch.linalg.inv(cam_poses_input[i]), self.opt.input_size, self.opt.input_size, self.opt.FoVy, opengl=False) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)
     
        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]
        final_input = torch.cat([images_input, rays_embeddings], dim=1) # [V=4, 10, H, W]
        results['input'] = final_input
        results['input_images'] = images_input
        results['sapines_input'] = sapines_input
        results['clip_images'] = images
        results['smpl_params'] = smpl_params

        cam_view = cam_poses.transpose(1, 2) # [V, 4, 4] w2c
        cam_view_proj = cam_view @ self.proj_matrix # [V, 4, 4] c2p
        cam_pos = torch.linalg.inv(cam_poses)[:, :3, 3] # [V, 3]
        
        results['cam_view'] = cam_view
        results['cam_view_proj'] = cam_view_proj
        results['cam_pos'] = cam_pos

        results['item'] = uid
        return results
    
    def getProjectionMatrix(self, znear, zfar, fovX, fovY, K = None, img_h = None, img_w = None):
        if K is None:
            tanHalfFovY = math.tan((fovY / 2))
            tanHalfFovX = math.tan((fovX / 2))
            top = tanHalfFovY * znear
            bottom = -top
            right = tanHalfFovX * znear
            left = -right
        else:
            near_fx = znear / K[0, 0]
            near_fy = znear / K[1, 1]

            left = - (img_w - K[0, 2]) * near_fx
            right = K[0, 2] * near_fx
            bottom = (K[1, 2] - img_h) * near_fy
            top = K[1, 2] * near_fy

        P = torch.zeros(4, 4)

        z_sign = 1.0

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        return P
    
    def condition_img(self, img_tensor, mask_tensor):

        img_tensor = img_tensor.permute(1, 2, 0)
        mask = (mask_tensor > 0).cpu().numpy().astype(np.uint8)
        coords = cv2.findNonZero(mask)
        x, y, w, h = cv2.boundingRect(coords)
        foreground = img_tensor[y:y+h, x:x+w]  # [h, w, 3]

        max_side = max(foreground.shape[0], foreground.shape[1])
        square_img = torch.zeros((max_side, max_side, 3), dtype=torch.float32)

        x_offset = (max_side - foreground.shape[1]) // 2
        y_offset = (max_side - foreground.shape[0]) // 2
        square_img[y_offset:y_offset+foreground.shape[0], 
                    x_offset:x_offset+foreground.shape[1]] = foreground

        white_bg = torch.ones((max_side, max_side, 3), dtype=torch.float32)
        
        mask_crop = mask_tensor[y:y+h, x:x+w]
        mask_square = torch.zeros((max_side, max_side), dtype=torch.float32)
        mask_square[y_offset:y_offset+foreground.shape[0], 
                    x_offset:x_offset+foreground.shape[1]] = mask_crop
        
        white_bg = white_bg * (1 - mask_square[..., None]) + square_img * mask_square[..., None]

        resized_img = F.interpolate(white_bg.permute(2, 0, 1).unsqueeze(0), 
                                    size=(1024, 1024), 
                                    mode='bilinear',
                                    align_corners=False).squeeze(0).permute(1, 2, 0)

        resized_img = resized_img.permute(2,0,1).contiguous()
        return resized_img