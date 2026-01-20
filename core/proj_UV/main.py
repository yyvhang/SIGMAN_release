import os
import cv2
import time
import tqdm
import numpy as np
import dearpygui.dearpygui as dpg
import kiui
import torch
import torch.nn.functional as F

from cam_utils import OrbitCamera
from mesh_renderer import Renderer

from grid_put import mipmap_linear_grid_put_2d, linear_grid_put_2d
from scipy.spatial.transform import Rotation as R

from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import binary_dilation, binary_erosion
def dilate_image(image, mask, iterations):
    # image: [H, W, C], current image
    # mask: [H, W], region with content (~mask is the region to inpaint)
    # iterations: int

    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    
    if mask.dtype != bool:
        mask = mask > 0.5

    inpaint_region = binary_dilation(mask, iterations=iterations)
    inpaint_region[mask] = 0

    search_region = mask.copy()
    not_search_region = binary_erosion(search_region, iterations=3)
    search_region[not_search_region] = 0

    search_coords = np.stack(np.nonzero(search_region), axis=-1)
    inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

    knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(search_coords)
    _, indices = knn.kneighbors(inpaint_coords)

    image[tuple(inpaint_coords.T)] = image[tuple(search_coords[indices[:, 0]].T)]
    return image


class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.gui = opt.gui # enable gui
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(W=1024, H=1024, r=1.5, fovy=0.6123773929618543)

        self.mode = "image"
        self.seed = opt.seed
        self.save_path = opt.save_path

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.buffer_overlay = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.buffer_out = None  # for 2D to 3D projection

        self.need_update = True  # update buffer_image
        self.need_update_overlay = True  # update buffer_overlay

        self.mouse_loc = np.array([0, 0])
        self.draw_mask = False
        self.draw_radius = 20
        self.mask_2d = np.zeros((self.W, self.H, 1), dtype=np.float32)

        # models
        self.device = torch.device("cuda")

        self.guidance = None
        self.guidance_embeds = None

        # renderer
        self.renderer = Renderer(self.device, opt)

        # input mesh
        # if self.opt.mesh is not None:
        #     self.renderer.load_mesh(self.opt.mesh)

        # input text
        # self.prompt = self.opt.posi_prompt + ', ' + self.opt.prompt
        # self.negative_prompt = self.opt.nega_prompt

        if self.gui:
            dpg.create_context()
            self.register_dpg()
            self.test_step()

    def __del__(self):
        if self.gui:
            dpg.destroy_context()

    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)
        
        print(f'[INFO] seed = {seed}')

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.last_seed = seed
    
    def prepare_guidance(self):
        
        if self.guidance is None:
            print(f'[INFO] loading guidance model...')
            if self.opt.enable_lcm:
                from guidance.sd_lcm_utils import StableDiffusion
            else:
                from guidance.sd_utils import StableDiffusion
            self.guidance = StableDiffusion(self.device, control_mode=self.opt.control_mode, model_key=self.opt.model_key, lora_keys=self.opt.lora_keys)
            print(f'[INFO] loaded guidance model!')

        print(f'[INFO] encoding prompt...')
        nega = self.guidance.get_text_embeds([self.negative_prompt])

        if not self.opt.text_dir:
            posi = self.guidance.get_text_embeds([self.prompt])
            self.guidance_embeds = torch.cat([nega, posi], dim=0)
        else:
            self.guidance_embeds = {}
            posi = self.guidance.get_text_embeds([self.prompt])
            self.guidance_embeds['default'] = torch.cat([nega, posi], dim=0)
            for d in ['front', 'side', 'back', 'top', 'bottom']:
                posi = self.guidance.get_text_embeds([self.prompt + f', {d} view'])
                self.guidance_embeds[d] = torch.cat([nega, posi], dim=0)
        
    def intrinsic_to_perspective(self, K, width=1024, height=1024, near=0.1, far=100.0):
        fx = K[0,0]
        fy = K[1,1]
        cx = K[0,2]
        cy = K[1,2]
        
        # 构建透视投影矩阵
        perspective = np.zeros((4,4))
        perspective[0,0] = 2 * fx / width
        perspective[1,1] = 2 * fy / height
        perspective[0,2] = 2 * (cx / width) - 1
        perspective[1,2] = 2 * (cy / height) - 1
        perspective[2,2] = -(far + near) / (far - near)
        perspective[2,3] = -2 * far * near / (far - near)
        perspective[3,2] = -1
        
        return perspective
    
    @torch.no_grad()
    def inpaint_view(self, pose, input_image, K):

        h = w = int(self.opt.texture_size)
        H = W = int(self.opt.render_resolution)

        convert_mat = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        pose = pose @ convert_mat
        cam_perspective = self.intrinsic_to_perspective(K)
        out = self.renderer.render(pose, cam_perspective, H, W)

        valid_pixels = out['alpha'].squeeze(-1).nonzero() # [N, 2]
        min_h, max_h = valid_pixels[:, 0].min().item(), valid_pixels[:, 0].max().item()
        min_w, max_w = valid_pixels[:, 1].min().item(), valid_pixels[:, 1].max().item()
        
        size = max(max_h - min_h + 1, max_w - min_w + 1) * 1.1
        h_start = min(min_h, max_h) - (size - (max_h - min_h + 1)) / 2
        w_start = min(min_w, max_w) - (size - (max_w - min_w + 1)) / 2

        min_h = int(h_start)
        min_w = int(w_start)
        max_h = int(min_h + size)
        max_w = int(min_w + size)

        # crop region is outside rendered image: do not crop at all.
        if min_h < 0 or min_w < 0 or max_h > H or max_w > W:
            min_h = 0
            min_w = 0
            max_h = H
            max_w = W

        def _zoom(x, mode='bilinear', size=(H, W)):
            return F.interpolate(x[..., min_h:max_h+1, min_w:max_w+1], size, mode=mode)

        image = _zoom(out['image'].permute(2, 0, 1).unsqueeze(0).contiguous()) # [1, 3, H, W]
        mask_generate = _zoom(out['cnt'].permute(2, 0, 1).unsqueeze(0).contiguous(), mode='nearest') < 0.1 # [1, 1, H, W] bool
        mask_np = mask_generate[0,0].cpu().numpy().astype(np.uint8) * 255

        kernel = np.ones((2,2), np.uint8)

        eroded_mask = cv2.erode(mask_np, kernel, iterations=1)

        mask_generate = torch.from_numpy(eroded_mask).to(mask_generate.device).float() / 255.0
        mask_generate = mask_generate.unsqueeze(0).unsqueeze(0)
        mask_generate = mask_generate > 0.5

        viewcos_old = _zoom(out['viewcos_cache'].permute(2, 0, 1).unsqueeze(0).contiguous()) # [1, 1, H, W]
        viewcos = _zoom(out['viewcos'].permute(2, 0, 1).unsqueeze(0).contiguous()) # [1, 1, H, W]

        mask_refine = ((viewcos_old < viewcos) & ~mask_generate)

        mask_keep = (~mask_generate & ~mask_refine)

        mask_generate = mask_generate.float()
        mask_refine = mask_refine.float()
        mask_keep = mask_keep.float()
        mask_generate_blur = mask_generate

        if not (mask_generate > 0.5).any():
            return

        control_images = {}

        # construct normal control
        if 'normal' in self.opt.control_mode:
            rot_normal = out['rot_normal'] # [H, W, 3]
            rot_normal[..., 0] *= -1 # align with normalbae: blue = front, red = left, green = top
            control_images['normal'] = _zoom(rot_normal.permute(2, 0, 1).unsqueeze(0).contiguous() * 0.5 + 0.5, size=(512, 512)) # [1, 3, H, W]
        
        # construct depth control
        if 'depth' in self.opt.control_mode:
            depth = out['depth']
            control_images['depth'] = _zoom(depth.view(1, 1, H, W), size=(512, 512)).repeat(1, 3, 1, 1) # [1, 3, H, W]
        
        # construct ip2p control
        if 'ip2p' in self.opt.control_mode:
            ori_image = _zoom(out['ori_image'].permute(2, 0, 1).unsqueeze(0).contiguous(), size=(512, 512)) # [1, 3, H, W]
            control_images['ip2p'] = ori_image

        # construct inpaint control
        if 'inpaint' in self.opt.control_mode:
            image_generate = image.clone()
            image_generate[mask_generate.repeat(1, 3, 1, 1) > 0.5] = -1 # -1 is inpaint region
            image_generate = F.interpolate(image_generate, size=(512, 512), mode='bilinear', align_corners=False)
            control_images['inpaint'] = image_generate

            # mask blending to avoid changing non-inpaint region (ref: https://github.com/lllyasviel/ControlNet-v1-1-nightly/commit/181e1514d10310a9d49bb9edb88dfd10bcc903b1)
            latents_mask = F.interpolate(mask_generate_blur, size=(64, 64), mode='bilinear') # [1, 1, 64, 64]
            latents_mask_refine = F.interpolate(mask_refine, size=(64, 64), mode='bilinear') # [1, 1, 64, 64]
            latents_mask_keep = F.interpolate(mask_keep, size=(64, 64), mode='bilinear') # [1, 1, 64, 64]
            control_images['latents_mask'] = latents_mask
            control_images['latents_mask_refine'] = latents_mask_refine
            control_images['latents_mask_keep'] = latents_mask_keep
            control_images['latents_original'] = self.guidance.encode_imgs(F.interpolate(image, (512, 512), mode='bilinear', align_corners=False).to(self.guidance.dtype)) # [1, 4, 64, 64]
        
        rgbs = _zoom(input_image)

        # apply mask to make sure non-inpaint region is not changed
        rgbs = rgbs * (1 - mask_keep) + image * mask_keep

        if self.opt.vis:
            if 'depth' in control_images:
                kiui.vis.plot_image(control_images['depth'])
            if 'normal' in control_images:
                kiui.vis.plot_image(control_images['normal'])
            if 'ip2p' in control_images:
                kiui.vis.plot_image(ori_image)
            # kiui.vis.plot_image(mask_generate)
            if 'inpaint' in control_images:
                kiui.vis.plot_image(control_images['inpaint'].clamp(0, 1))
                # kiui.vis.plot_image(control_images['inpaint_refine'].clamp(0, 1))
            if 'depth_inpaint' in control_images:
                kiui.vis.plot_image(control_images['depth_inpaint'][:, :3].clamp(0, 1))
                kiui.vis.plot_image(control_images['depth_inpaint'][:, 3:].clamp(0, 1))
            kiui.vis.plot_image(rgbs)

        # grid put
        # project-texture mask
        proj_mask = (out['alpha'] > 0) & (out['viewcos'] > self.opt.cos_thresh)  # [H, W, 1]
        proj_mask = _zoom(proj_mask.view(1, 1, H, W).float(), 'nearest').view(-1).bool()
        uvs = _zoom(out['uvs'].permute(2, 0, 1).unsqueeze(0).contiguous(), 'nearest')

        uvs = uvs.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 2)[proj_mask]
        rgbs = rgbs.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 3)[proj_mask]

        cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(h, w, uvs[..., [1, 0]] * 2 - 1, rgbs, min_resolution=128, return_count=True)

        self.backup()
        mask = cur_cnt.squeeze(-1) > 0
        self.albedo[mask] += cur_albedo[mask]
        self.cnt[mask] += cur_cnt[mask]
        # update mesh texture for rendering
        self.update_mesh_albedo()
        # kiui.vis.plot_image(cur_albedo.detach().cpu().numpy())
        # update viewcos cache
        viewcos = viewcos.view(-1, 1)[proj_mask]
        cur_viewcos = mipmap_linear_grid_put_2d(h, w, uvs[..., [1, 0]] * 2 - 1, viewcos, min_resolution=256)
        self.renderer.mesh.viewcos_cache = torch.maximum(self.renderer.mesh.viewcos_cache, cur_viewcos)
    
    @torch.no_grad()
    def backup(self):
        self.backup_albedo = self.albedo.clone()
        self.backup_cnt = self.cnt.clone()
        self.backup_viewcos_cache = self.renderer.mesh.viewcos_cache.clone()
    
    @torch.no_grad()
    def restore(self):
        self.albedo = self.backup_albedo.clone()
        self.cnt = self.backup_cnt.clone()
        self.renderer.mesh.cnt = self.cnt
        self.renderer.mesh.viewcos_cache = self.backup_viewcos_cache.clone()
        self.update_mesh_albedo()
    
    @torch.no_grad()
    def update_mesh_albedo(self):
        mask = self.cnt.squeeze(-1) > 0
        cur_albedo = self.albedo.clone()
        cur_albedo[mask] /= self.cnt[mask].repeat(1, 3)
        self.renderer.mesh.albedo = cur_albedo

    @torch.no_grad()
    def dilate_texture(self):
        h = w = int(self.opt.texture_size)

        self.backup()

        mask = self.cnt.squeeze(-1) > 0
        
        mask = mask.view(h, w)
        mask = mask.detach().cpu().numpy()

        self.albedo = dilate_image(self.albedo, mask, iterations=int(h*0.2))
        self.cnt = dilate_image(self.cnt, mask, iterations=int(h*0.2))
        
        self.update_mesh_albedo()
    
    @torch.no_grad()
    def deblur(self, ratio=2):
        h = w = int(self.opt.texture_size)

        self.backup()

        cur_albedo = self.renderer.mesh.albedo.detach().cpu().numpy()
        cur_albedo = (cur_albedo * 255).astype(np.uint8)
        cur_albedo = cv2.resize(cur_albedo, (w // ratio, h // ratio), interpolation=cv2.INTER_CUBIC)
        cur_albedo = kiui.sr.sr(cur_albedo, scale=ratio)
        cur_albedo = cur_albedo.astype(np.float32) / 255
        # kiui.vis.plot_image(cur_albedo)
        cur_albedo = torch.from_numpy(cur_albedo).to(self.device)
        self.renderer.mesh.albedo = cur_albedo

    @torch.no_grad()
    def initialize(self, keep_ori_albedo=False):
        
        h = w = int(self.opt.texture_size)

        self.albedo = torch.zeros((h, w, 3), device=self.device, dtype=torch.float32)
        self.cnt = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32)
        self.viewcos_cache = - torch.ones((h, w, 1), device=self.device, dtype=torch.float32)

        # keep original texture if using ip2p
        if 'ip2p' in self.opt.control_mode:
            self.renderer.mesh.ori_albedo = self.renderer.mesh.albedo.clone()

        if keep_ori_albedo:
            self.albedo = self.renderer.mesh.albedo.clone()
            self.cnt += 1 # set to 1
            self.viewcos_cache *= -1 # set to 1
        
        self.renderer.mesh.albedo = self.albedo
        self.renderer.mesh.cnt = self.cnt 
        self.renderer.mesh.viewcos_cache = self.viewcos_cache


    def rotate_obj(self, vertices, smplx_param, frame_index):
        device = vertices.device
        smplx_param = np.load(smplx_param, allow_pickle=True)
        global_orient = smplx_param['global_orient'][frame_index]
        transl = smplx_param['transl'][frame_index]
        transl[1] /= 4 #MvhumanNet
        vertices = vertices.cpu().numpy()
        vertices_no_trans = vertices - transl

        rot_matrix = R.from_rotvec(global_orient).as_matrix()
        inv_rot_matrix = np.linalg.inv(rot_matrix)
        vertices_original = (inv_rot_matrix @ vertices_no_trans.T).T

        return torch.from_numpy(vertices_original).float().to(device)

    @torch.no_grad()
    def generate(self,obj_path,image_folder,camera_path,smplx_param,frame_index):
        self.renderer.load_mesh(obj_path)
        self.initialize(keep_ori_albedo=False)
        import json
        camera_param = json.load(open(camera_path, 'r'))
        camera_frames = list(camera_param.keys())
        camera_params = [camera_param[camera_frame] for camera_frame in camera_frames]
        poses = []
        Ks = []
        for camera_param in camera_params:
            camera_matrix = np.eye(4)
            K = np.array(camera_param['K'])
            R = np.array(camera_param['R'])
            T = np.array(camera_param['T'])
            camera_matrix[:3, :3] = R
            camera_matrix[:3, 3] = T
            poses.append(camera_matrix)
            Ks.append(K)
        image_files = sorted(os.listdir(image_folder))
        image_paths = [os.path.join(image_folder,image_file) for image_file in image_files]

        start_t = time.time()
        camera_positions = []
        camera_directions = []
        choose_view = [30,31,34,36,37,41,45,49,53,56,60,65,67,70,75,78,85,89]
        poses_ = [poses[id] for id in choose_view]
        image_paths_ = [image_paths[id] for id in choose_view]
        Ks_ = [Ks[id] for id in choose_view]


        print(f'[INFO] start generation...')
        for pose, image_path, K in tqdm.tqdm(zip(poses_, image_paths_, Ks_)):
            # render image
            image = kiui.read_image(image_path) # kiui automatically handles RGB conversion
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).contiguous().to(self.device)
            pose = np.linalg.inv(pose) #w2c -> c2w
            self.inpaint_view(pose, image, K)
            self.need_update = True
            self.test_step()

        self.dilate_texture()
        torch.cuda.synchronize()
        end_t = time.time()
        print(f'[INFO] finished generation in {end_t - start_t:.3f}s!')

        self.need_update = True

    @torch.no_grad()
    def test_step(self):
        # ignore if no need to update
        if not self.need_update and not self.need_update_overlay:
            return

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        # should update image
        if self.need_update:
            # render image

            out = self.renderer.render(self.cam.pose, self.cam.perspective, self.H, self.W)

            buffer_image = out[self.mode]  # [H, W, 3]

            if self.mode in ['depth', 'alpha', 'viewcos', 'viewcos_cache', 'cnt']:
                buffer_image = buffer_image.repeat(1, 1, 3)
                if self.mode == 'depth':
                    buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)
            
            if self.mode in ['normal', 'rot_normal']:
                buffer_image = (buffer_image + 1) / 2

            self.buffer_image = buffer_image.contiguous().clamp(0, 1).detach().cpu().numpy()

            self.buffer_out = out

            self.need_update = False
        
        # should update overlay
        if self.need_update_overlay:
            buffer_overlay = np.zeros_like(self.buffer_overlay)

            # draw mask 2d
            buffer_overlay += self.mask_2d * 0.5
            
            self.buffer_overlay = buffer_overlay
            self.need_update_overlay = False

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        if self.gui:

            # mix image and overlay
            buffer = np.clip(
                self.buffer_image + self.buffer_overlay, 0, 1
            )  # mix mode, sometimes unclear

            dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)} FPS)")
            dpg.set_value(
                "_texture", buffer
            )  # buffer must be contiguous, else seg fault!


    def save_model(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
        path = os.path.join(out_dir, 'smplxuv.obj')
        self.renderer.export_mesh(path)

        print(f"[INFO] save model to {path}.")
        return path

    def register_dpg(self):
        ### register texture

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window

        # the rendered image, as the primary window
        with dpg.window(
            tag="_primary_window",
            width=self.W,
            height=self.H,
            pos=[0, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        # control window
        with dpg.window(
            label="Control",
            tag="_control_window",
            width=600,
            height=self.H,
            pos=[self.W, 0],
            no_move=True,
            no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            def callback_setattr(sender, app_data, user_data):
                setattr(self, user_data, app_data)

            # init stuff
            with dpg.collapsing_header(label="Inpaint", default_open=True):

                # seed stuff
                def callback_set_seed(sender, app_data):
                    self.seed = app_data
                    self.seed_everything()

                dpg.add_input_text(
                    label="seed",
                    default_value=self.seed,
                    on_enter=True,
                    callback=callback_set_seed,
                )

                # prompt stuff
                dpg.add_input_text(
                    label="prompt",
                    default_value=self.prompt,
                    callback=callback_setattr,
                    user_data="prompt",
                )
            
                dpg.add_input_text(
                    label="negative",
                    default_value=self.negative_prompt,
                    callback=callback_setattr,
                    user_data="negative_prompt",
                )

                # generate texture
                with dpg.group(horizontal=True):
                    dpg.add_text("Generate: ")

                    def callback_generate(sender, app_data):
                        self.generate()
                        self.need_update = True

                    dpg.add_button(
                        label="auto",
                        tag="_button_generate",
                        callback=callback_generate,
                    )
                    dpg.bind_item_theme("_button_generate", theme_button)

                    def callback_init(sender, app_data, user_data):
                        self.initialize(keep_ori_albedo=user_data)
                        self.need_update = True

                    dpg.add_button(
                        label="init",
                        tag="_button_init",
                        callback=callback_init,
                        user_data=False,
                    )
                    dpg.bind_item_theme("_button_init", theme_button)

                    dpg.add_button(
                        label="init_ori",
                        tag="_button_init_ori",
                        callback=callback_init,
                        user_data=True,
                    )
                    dpg.bind_item_theme("_button_init_ori", theme_button)

                    def callback_encode(sender, app_data):
                        self.prepare_guidance()

                    dpg.add_button(
                        label="encode",
                        tag="_button_encode",
                        callback=callback_encode,
                    )
                    dpg.bind_item_theme("_button_encode", theme_button)

                    def callback_inpaint(sender, app_data):
                        # inpaint current view
                        self.inpaint_view(self.cam.pose)
                        self.need_update = True

                    dpg.add_button(
                        label="inpaint",
                        tag="_button_inpaint",
                        callback=callback_inpaint,
                    )
                    dpg.bind_item_theme("_button_inpaint", theme_button)

                    def callback_undo(sender, app_data):
                        # inpaint current view
                        self.restore()
                        self.need_update = True

                    dpg.add_button(
                        label="undo",
                        tag="_button_undo",
                        callback=callback_undo,
                    )
                    dpg.bind_item_theme("_button_undo", theme_button)

                with dpg.group(horizontal=True):
                    dpg.add_text("Post-process: ")

                    def callback_dilate(sender, app_data):
                        self.dilate_texture()
                        self.need_update = True

                    dpg.add_button(
                        label="dilate",
                        tag="_button_dilate",
                        callback=callback_dilate,
                    )
                    dpg.bind_item_theme("_button_dilate", theme_button)

                    def callback_deblur(sender, app_data):
                        self.deblur()
                        self.need_update = True

                    dpg.add_button(
                        label="deblur",
                        tag="_button_deblur",
                        callback=callback_deblur,
                    )
                    dpg.bind_item_theme("_button_deblur", theme_button)
                
            
                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Save: ")

                    def callback_save_model(sender, app_data):
                        self.save_model()

                    dpg.add_button(
                        label="mesh",
                        tag="_button_save_model",
                        callback=callback_save_model,
                    )
                    dpg.bind_item_theme("_button_save_model", theme_button)

                    dpg.add_input_text(
                        label="",
                        default_value=self.opt.save_path,
                        callback=callback_setattr,
                        user_data="save_path",
                    )

            
            # draw mask 
            with dpg.collapsing_header(label="Repaint", default_open=True):
                with dpg.group(horizontal=True):

                    def callback_toggle_draw_mask(sender, app_data):
                        self.draw_mask = not self.draw_mask
                        self.need_update_overlay = True
                    
                    def callback_reset_mask(sender, app_data):
                        self.mask_2d *= 0
                        self.need_update_overlay = True
                    
                    def callback_erase_mask(sender, app_data):
                        out = self.buffer_out
                        h = w = int(self.opt.texture_size)

                        proj_mask = (out['alpha'] > 0.1).view(-1).bool()
                        uvs = out['uvs'].view(-1, 2)[proj_mask]
                        mask_2d = torch.from_numpy(self.mask_2d).to(self.device).permute(2, 0, 1).contiguous()
                        mask_2d = F.interpolate(mask_2d.unsqueeze(0), size=out['alpha'].shape[:2], mode='nearest').squeeze(0)
                        mask_2d = mask_2d.view(-1, 1)[proj_mask]
                        # mask_2d = mipmap_linear_grid_put_2d(h, w, uvs[..., [1, 0]] * 2 - 1, mask_2d, min_resolution=128)
                        mask_2d = linear_grid_put_2d(h, w, uvs[..., [1, 0]] * 2 - 1, mask_2d)
                        
                        # reset albedo and cnt
                        self.backup()
                        
                        mask = mask_2d.squeeze(-1) > 0.1
                        self.albedo[mask] = 0
                        self.cnt[mask] = 0
                        self.renderer.mesh.viewcos_cache[mask] = -1

                        # update mesh texture for rendering
                        self.update_mesh_albedo()
                        
                        # reset mask_2d too
                        self.mask_2d *= 0
                        self.need_update = True
                        self.need_update_overlay = True

                    dpg.add_checkbox(
                        label="draw",
                        default_value=self.draw_mask,
                        callback=callback_toggle_draw_mask,
                    )

                    dpg.add_button(
                        label="reset",
                        tag="_button_reset_mask",
                        callback=callback_reset_mask,
                    )
                    dpg.bind_item_theme("_button_reset_mask", theme_button)

                    dpg.add_button(
                        label="erase",
                        tag="_button_erase_mask",
                        callback=callback_erase_mask,
                    )
                    dpg.bind_item_theme("_button_erase_mask", theme_button)
                
                dpg.add_slider_int(
                    label="draw radius",
                    min_value=1,
                    max_value=100,
                    format="%d",
                    default_value=self.draw_radius,
                    callback=callback_setattr,
                    user_data="draw_radius",
                )

            # rendering options
            with dpg.collapsing_header(label="Render", default_open=True):
                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(
                    ("image", "depth", "alpha", "normal", "rot_normal", "viewcos", "viewcos_cache", "cnt"),
                    label="mode",
                    default_value=self.mode,
                    callback=callback_change_mode,
                )

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = np.deg2rad(app_data)
                    self.need_update = True

                dpg.add_slider_int(
                    label="FoV (vertical)",
                    min_value=1,
                    max_value=120,
                    format="%d deg",
                    default_value=np.rad2deg(self.cam.fovy),
                    callback=callback_set_fovy,
                )


        ### register camera handler

        def callback_camera_drag_rotate_or_draw_mask(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            if self.draw_mask:
                self.mask_2d[
                    int(self.mouse_loc[1])
                    - self.draw_radius : int(self.mouse_loc[1])
                    + self.draw_radius,
                    int(self.mouse_loc[0])
                    - self.draw_radius : int(self.mouse_loc[0])
                    + self.draw_radius,
                ] = 1

            else:
                self.cam.orbit(dx, dy)
                self.need_update = True

            self.need_update_overlay = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

        def callback_set_mouse_loc(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            # just the pixel coordinate in image
            self.mouse_loc = np.array(app_data)

        with dpg.handler_registry():
            # for camera moving
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=callback_camera_drag_rotate_or_draw_mask,
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )
            dpg.add_mouse_move_handler(callback=callback_set_mouse_loc)

        dpg.create_viewport(
            title="InteX",
            width=self.W + 600,
            height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        if os.path.exists("LXGWWenKai-Regular.ttf"):
            with dpg.font_registry():
                with dpg.font("LXGWWenKai-Regular.ttf", 18) as default_font:
                    dpg.bind_font(default_font)

        # dpg.show_metrics()

        dpg.show_viewport()

    def render(self):
        assert self.gui
        while dpg.is_dearpygui_running():
            self.test_step()
            dpg.render_dearpygui_frame()
    
    # no gui mode
    def run(self):
        wrong = []
        root_paths = np.load('./data/xxx.npy',allow_pickle=True)[args.start:args.end] #npy file contains the root path of the item
        for root_path in root_paths:
            id = root_path.split('/')[-1]
            smplx_path = os.path.join(root_path, f'mesh/{id}_smplx_modified.obj')
            rgb_folder = os.path.join(root_path,'rgb_map')
            camera_path = os.path.join(root_path,'camera_full_calibration.json')
            try:
                self.generate(smplx_path, rgb_folder, camera_path, smplx_param=None, frame_index=None)
                save_dir = os.path.join(root_path,'UV')
                self.save_model(save_dir)
            except:
                wrong.append(root_path)
        np.save(args.save_wrong, np.array(wrong))
        

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf
    import os
    os.environ['TORCH_CUDA_ARCH_LIST'] = "8.0"

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='configs/base.yaml', help="path to the yaml config file")
    parser.add_argument("--start", type=int, default=500)
    parser.add_argument("--end", type=int, default=3000)
    parser.add_argument("--save_wrong", type=str, default='wrong/wrong_id.npy')
    args, extras = parser.parse_known_args()
    
    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    print(opt)

    gui = GUI(opt)
    if opt.gui:
        gui.render()
    else:
        gui.run()
