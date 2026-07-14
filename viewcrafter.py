import sys
sys.path.append('./extern/dust3r')
from dust3r.inference import inference, load_model
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import to_numpy
import trimesh
import torch
import json
import h5py
import time
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import os
import copy
import cv2  
import glob
from PIL import Image
import pytorch3d
from torchvision.utils import save_image
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from utils.pvd_utils import *
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from utils.diffusion_utils import instantiate_from_config,load_model_checkpoint,image_guided_synthesis
from utils.colmap_utils import (
    read_cameras_text, undistorted_camera, camera_to_intrinsic
)
from pathlib import Path
from torchvision.utils import save_image
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

class ViewCrafter:
    def __init__(self, opts, gradio = False):
        self.opts = opts
        self.device = opts.device
        # self.setup_dust3r()
        self.setup_diffusion()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0) #.cuda()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True) #.cuda()
        # initialize ref images, pcd
        if not gradio:
            if self.opts.scene_json is not None and self.opts.scene_id is not None:
                assert self.opts.scene_json.endswith('.json'), "scene_json must be a json file"
                assert os.path.isfile(self.opts.scene_json), f"scene_json file {self.opts.scene_json} does not exist"
                assert os.path.isdir(self.opts.image_dir), f"image_dir {self.opts.image_dir} needs to be the base data directory of dataset"
                
                # load input and gt data for scene
                self.images, self.img_ori, self.camera_poses, self.gt_images, self.gt_img_ori, self.gt_camera_poses, self.K, self.scene_dict = \
                    self.load_data_from_json(
                        data_dir=self.opts.image_dir, 
                        json_file=self.opts.scene_json, 
                        scene_id=self.opts.scene_id
                    )

                # run dust3r
                # self.run_dust3r(input_images=self.images, poses=self.camera_poses, K=self.K, clean_pc=True)
            elif os.path.isfile(self.opts.image_dir):
                self.images, self.img_ori = self.load_initial_images(image_dir=self.opts.image_dir)
                self.run_dust3r(input_images=self.images)
            elif os.path.isdir(self.opts.image_dir):
                self.images, self.img_ori = self.load_initial_dir(image_dir=self.opts.image_dir)
                self.run_dust3r(input_images=self.images, clean_pc=True)    
            else:
                print(f"{self.opts.image_dir} doesn't exist")           
        
    def run_dust3r(self, input_images, poses=None, K=None, clean_pc=False):
        pairs = make_pairs(input_images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, self.dust3r, self.device, batch_size=self.opts.batch_size)

        mode = GlobalAlignerMode.PointCloudOptimizer #if len(self.images) > 2 else GlobalAlignerMode.PairViewer
        scene = global_aligner(output, device=self.device, mode=mode)
        if mode == GlobalAlignerMode.PointCloudOptimizer:
            if poses is not None and K is not None:
                scene.preset_pose(poses)
                focals = [K[0, 0]] * len(poses)
                scene.preset_focal(focals)
                # principal_points = [[K[0, 2], K[1, 2]]] * len(poses)
                # scene.preset_principal_point(principal_points)
                loss = scene.compute_global_alignment(init='known_poses', niter=self.opts.niter, schedule=self.opts.schedule, lr=self.opts.lr)
            else:
                loss = scene.compute_global_alignment(init='mst', niter=self.opts.niter, schedule=self.opts.schedule, lr=self.opts.lr)

        if clean_pc:
            self.scene = scene.clean_pointcloud()
        else:
            self.scene = scene

    def run_diffusion(self, renderings, conditioning_frame=None):

        prompts = [self.opts.prompt]
        videos = (renderings * 2. - 1.).permute(3,0,1,2).unsqueeze(0).to(self.device)
        if conditioning_frame is not None:
            conditioning_frame = (conditioning_frame * 2. - 1.).permute(2,0,1).unsqueeze(0).to(self.device)
        condition_index = [0]
        with torch.no_grad(), torch.cuda.amp.autocast():
            # [1,1,c,t,h,w]
            batch_samples = image_guided_synthesis(self.diffusion, prompts, videos, self.noise_shape, self.opts.n_samples, self.opts.ddim_steps, self.opts.ddim_eta, \
                               self.opts.unconditional_guidance_scale, self.opts.cfg_img, self.opts.frame_stride, self.opts.text_input, self.opts.multiple_cond_cfg, self.opts.timestep_spacing, self.opts.guidance_rescale, condition_index, conditioning_frame=conditioning_frame)

            # save_results_seperate(batch_samples[0], self.opts.save_dir, fps=8)
            # torch.Size([1, 3, 25, 576, 1024]) [-1,1]

        return torch.clamp(batch_samples[0][0].permute(1,2,3,0), -1., 1.) 

    def nvs_single_view(self, gradio=False):
        # 最后一个view为 0 pose
        c2ws = self.scene.get_im_poses().detach()[1:] 
        principal_points = self.scene.get_principal_points().detach()[1:] #cx cy
        focals = self.scene.get_focals().detach()[1:] 
        shape = self.images[0]['true_shape']
        H, W = int(shape[0][0]), int(shape[0][1])
        pcd = [i.detach() for i in self.scene.get_pts3d(clip_thred=self.opts.dpt_trd)] # a list of points of size whc
        depth = [i.detach() for i in self.scene.get_depthmaps()]
        depth_avg = depth[-1][H//2,W//2] #以图像中心处的depth(z)为球心旋转
        radius = depth_avg*self.opts.center_scale #缩放调整

        ## change coordinate
        c2ws,pcd =  world_point_to_obj(poses=c2ws, points=torch.stack(pcd), k=-1, r=radius, elevation=self.opts.elevation, device=self.device)

        imgs = np.array(self.scene.imgs)
        
        masks = None

        if self.opts.mode == 'single_view_nbv':
            ## 输入candidate->渲染mask->最大mask对应的pose作为nbv
            ## nbv模式下self.opts.d_theta[0], self.opts.d_phi[0]代表search space中的网格theta, phi之间的间距; self.opts.d_phi[0]的符号代表方向,分为左右两个方向
            ## FIXME hard coded candidate view数量, 以left为例,第一次迭代从[左,左上]中选取, 从第二次开始可以从[左,左上,左下]中选取
            num_candidates = 2
            candidate_poses,thetas,phis = generate_candidate_poses(c2ws, H, W, focals, principal_points, self.opts.d_theta[0], self.opts.d_phi[0],num_candidates, self.device)
            _, viewmask = run_render([pcd[-1]], [imgs[-1]],masks, H, W, candidate_poses,num_candidates, self.device)
            nbv_id = torch.argmin(viewmask.sum(dim=[1,2,3])).item()
            save_image( viewmask.permute(0,3,1,2), os.path.join(self.opts.save_dir,f"candidate_mask0_nbv{nbv_id}.png"), normalize=True, value_range=(0, 1))
            theta_nbv = thetas[nbv_id]
            phi_nbv = phis[nbv_id]
            # generate camera trajectory from T_curr to T_nbv
            camera_traj,num_views = generate_traj_specified(c2ws, H, W, focals, principal_points, theta_nbv, phi_nbv, self.opts.d_r[0],self.opts.video_length, self.device)
            # 重置elevation
            self.opts.elevation -= theta_nbv
        elif self.opts.mode == 'single_view_target':
            camera_traj,num_views = generate_traj_specified(c2ws, H, W, focals, principal_points, self.opts.d_theta[0], self.opts.d_phi[0], self.opts.d_r[0],self.opts.d_x[0]*depth_avg/focals.item(),self.opts.d_y[0]*depth_avg/focals.item(),self.opts.video_length, self.device)
        elif self.opts.mode == 'single_view_txt':
            if not gradio:
                with open(self.opts.traj_txt, 'r') as file:
                    lines = file.readlines()
                    phi = [float(i) for i in lines[0].split()]
                    theta = [float(i) for i in lines[1].split()]
                    r = [float(i) for i in lines[2].split()]
            else: 
                phi, theta, r = self.gradio_traj
            camera_traj,num_views = generate_traj_txt(c2ws, H, W, focals, principal_points, phi, theta, r,self.opts.video_length, self.device,viz_traj=True, save_dir = self.opts.save_dir)
        else:
            raise KeyError(f"Invalid Mode: {self.opts.mode}")

        render_results, viewmask = run_render([pcd[-1]], [imgs[-1]],masks, H, W, camera_traj,num_views, self.device)
        render_results = F.interpolate(render_results.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
        render_results[0] = self.img_ori
        if self.opts.mode == 'single_view_txt':
            if phi[-1]==0. and theta[-1]==0. and r[-1]==0.:
                render_results[-1] = self.img_ori
                
        save_video(render_results, os.path.join(self.opts.save_dir, 'render0.mp4'))
        save_pointcloud_with_normals([imgs[-1]], [pcd[-1]], msk=None, save_path=os.path.join(self.opts.save_dir,'pcd0.ply') , mask_pc=False, reduce_pc=False)
        diffusion_results = self.run_diffusion(render_results)
        save_video((diffusion_results + 1.0) / 2.0, os.path.join(self.opts.save_dir, 'diffusion0.mp4'))

        return diffusion_results

    def nvs_sparse_view(self,iter):

        c2ws = self.scene.get_im_poses().detach()
        principal_points = self.scene.get_principal_points().detach()
        focals = self.scene.get_focals().detach()
        shape = self.images[0]['true_shape']
        H, W = int(shape[0][0]), int(shape[0][1])
        pcd = [i.detach() for i in self.scene.get_pts3d(clip_thred=self.opts.dpt_trd)] # a list of points of size whc
        depth = [i.detach() for i in self.scene.get_depthmaps()]
        depth_avg = depth[0][H//2,W//2] #以ref图像中心处的depth(z)为球心旋转
        radius = depth_avg*self.opts.center_scale #缩放调整

        ## masks for cleaner point cloud
        self.scene.min_conf_thr = float(self.scene.conf_trf(torch.tensor(self.opts.min_conf_thr)))
        masks = self.scene.get_masks()
        depth = self.scene.get_depthmaps()
        bgs_mask = [dpt > self.opts.bg_trd*(torch.max(dpt[40:-40,:])+torch.min(dpt[40:-40,:])) for dpt in depth]
        masks_new = [m+mb for m, mb in zip(masks,bgs_mask)] 
        masks = to_numpy(masks_new)

        ## render, 从c2ws[0]即ref image对应的相机开始
        imgs = np.array(self.scene.imgs)

        if self.opts.mode == 'single_view_ref_iterative':
            c2ws,pcd =  world_point_to_obj(poses=c2ws, points=torch.stack(pcd), k=0, r=radius, elevation=self.opts.elevation, device=self.device)
            camera_traj,num_views = generate_traj_specified(c2ws[0:1], H, W, focals[0:1], principal_points[0:1], self.opts.d_theta[iter], self.opts.d_phi[iter], self.opts.d_r[iter],self.opts.video_length, self.device)
            render_results, viewmask = run_render(pcd, imgs,masks, H, W, camera_traj,num_views, self.device)
            render_results = F.interpolate(render_results.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
            render_results[0] = self.img_ori
        elif self.opts.mode == 'single_view_1drc_iterative':
            self.opts.elevation -= self.opts.d_theta[iter-1]
            c2ws,pcd =  world_point_to_obj(poses=c2ws, points=torch.stack(pcd), k=-1, r=radius, elevation=self.opts.elevation, device=self.device)
            camera_traj,num_views = generate_traj_specified(c2ws[-1:], H, W, focals[-1:], principal_points[-1:], self.opts.d_theta[iter], self.opts.d_phi[iter], self.opts.d_r[iter],self.opts.video_length, self.device)
            render_results, viewmask = run_render(pcd, imgs,masks, H, W, camera_traj,num_views, self.device)
            render_results = F.interpolate(render_results.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
            render_results[0] = (self.images[-1]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2.
        elif self.opts.mode == 'single_view_nbv':
            c2ws,pcd =  world_point_to_obj(poses=c2ws, points=torch.stack(pcd), k=-1, r=radius, elevation=self.opts.elevation, device=self.device)
            ## 输入candidate->渲染mask->最大mask对应的pose作为nbv
            ## nbv模式下self.opts.d_theta[0], self.opts.d_phi[0]代表search space中的网格theta, phi之间的间距; self.opts.d_phi[0]的符号代表方向,分为左右两个方向
            ## FIXME hard coded candidate view数量, 以left为例,第一次迭代从[左,左上]中选取, 从第二次开始可以从[左,左上,左下]中选取
            num_candidates = 3
            candidate_poses,thetas,phis = generate_candidate_poses(c2ws[-1:], H, W, focals[-1:], principal_points[-1:], self.opts.d_theta[0], self.opts.d_phi[0], num_candidates, self.device)
            _, viewmask = run_render(pcd, imgs,masks, H, W, candidate_poses,num_candidates, self.device, nbv=True)
            nbv_id = torch.argmin(viewmask.sum(dim=[1,2,3])).item()
            save_image(viewmask.permute(0,3,1,2), os.path.join(self.opts.save_dir,f"candidate_mask{iter}_nbv{nbv_id}.png"), normalize=True, value_range=(0, 1))
            theta_nbv = thetas[nbv_id]
            phi_nbv = phis[nbv_id]   
            # generate camera trajectory from T_curr to T_nbv
            camera_traj,num_views = generate_traj_specified(c2ws[-1:], H, W, focals[-1:], principal_points[-1:], theta_nbv, phi_nbv, self.opts.d_r[0],self.opts.video_length, self.device)
            # 重置elevation
            self.opts.elevation -= theta_nbv    
            render_results, viewmask = run_render(pcd, imgs,masks, H, W, camera_traj,num_views, self.device)
            render_results = F.interpolate(render_results.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
            render_results[0] = (self.images[-1]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2. 
        else:
            raise KeyError(f"Invalid Mode: {self.opts.mode}")

        save_video(render_results, os.path.join(self.opts.save_dir, f'render{iter}.mp4'))
        save_pointcloud_with_normals(imgs, pcd, msk=masks, save_path=os.path.join(self.opts.save_dir, f'pcd{iter}.ply') , mask_pc=True, reduce_pc=False)
        diffusion_results = self.run_diffusion(render_results)
        save_video((diffusion_results + 1.0) / 2.0, os.path.join(self.opts.save_dir, f'diffusion{iter}.mp4'))
        # torch.Size([25, 576, 1024, 3])
        return diffusion_results
    
    def nvs_sparse_view_interp(self):

        c2ws = self.scene.get_im_poses().detach()
        principal_points = self.scene.get_principal_points().detach()
        focals = self.scene.get_focals().detach()
        shape = self.images[0]['true_shape']
        H, W = int(shape[0][0]), int(shape[0][1])
        pcd = [i.detach() for i in self.scene.get_pts3d(clip_thred=self.opts.dpt_trd)] # a list of points of size whc
        depth = [i.detach() for i in self.scene.get_depthmaps()]

        if len(self.images) == 2:
            masks = None
            mask_pc = False
        else:
            ## masks for cleaner point cloud
            self.scene.min_conf_thr = float(self.scene.conf_trf(torch.tensor(self.opts.min_conf_thr)))
            masks = self.scene.get_masks()
            depth = self.scene.get_depthmaps()
            bgs_mask = [dpt > self.opts.bg_trd*(torch.max(dpt[40:-40,:])+torch.min(dpt[40:-40,:])) for dpt in depth]
            masks_new = [m+mb for m, mb in zip(masks,bgs_mask)] 
            masks = to_numpy(masks_new)
            mask_pc = True

        imgs = np.array(self.scene.imgs)

        camera_traj, num_views = generate_traj_interp(c2ws, H, W, focals, principal_points, self.opts.video_length, self.device)
        render_results, viewmask = run_render(pcd, imgs,masks, H, W, camera_traj,num_views, self.device)
        render_results = F.interpolate(render_results.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
        
        for i in range(len(self.img_ori)):
            render_results[i*(self.opts.video_length - 1)] = self.img_ori[i]
        save_video(render_results, os.path.join(self.opts.save_dir, f'render.mp4'))
        save_pointcloud_with_normals(imgs, pcd, msk=masks, save_path=os.path.join(self.opts.save_dir, f'pcd.ply') , mask_pc=mask_pc, reduce_pc=False)

        diffusion_results = []
        print(f'Generating {len(self.img_ori)-1} clips\n')
        for i in range(len(self.img_ori)-1 ):
            print(f'Generating clip {i} ...\n')
            diffusion_results.append(self.run_diffusion(render_results[i*(self.opts.video_length - 1):self.opts.video_length+i*(self.opts.video_length - 1)]))
        print(f'Finish!\n')
        diffusion_results = torch.cat(diffusion_results)
        save_video((diffusion_results + 1.0) / 2.0, os.path.join(self.opts.save_dir, f'diffusion.mp4'))
        # torch.Size([25, 576, 1024, 3])
        return diffusion_results

    def nvs_single_view_eval(self):

        # get camera trajectory of the input frames
        c2ws = self.scene.get_im_poses().detach()
        principal_points = self.scene.get_principal_points().detach()
        focals = self.scene.get_focals().detach()
        shape = self.images[0]['true_shape']
        H, W = int(shape[0][0]), int(shape[0][1])
        pcd = [i.detach() for i in self.scene.get_pts3d(clip_thred=self.opts.dpt_trd)] # a list of points of size whc
        c2ws,pcd =  world_point_to_kth(poses=c2ws, points=torch.stack(pcd), k=0, device=self.device)
        camera_traj,num_views = generate_traj(c2ws, H, W, focals, principal_points, self.device)
        
        # estimate pcd again using only one ref image
        images_ref = [self.images[0], copy.deepcopy(self.images[0])]
        images_ref[1]['idx'] = 1
        self.run_dust3r(input_images=images_ref)
        pcd_ref = self.scene.get_pts3d(clip_thred=self.opts.dpt_trd)[0].detach()
        img_ref = np.array(self.scene.imgs)[0]
        masks = None

        render_results, viewmask = run_render([pcd_ref], [img_ref],masks, H, W, camera_traj,num_views, self.device)
        render_results = F.interpolate(render_results.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
        render_results[0] = self.img_ori[0]
        save_video(render_results, os.path.join(self.opts.save_dir, f'render_ref0.mp4'))
        diffusion_results = self.run_diffusion(render_results)

        save_video((diffusion_results + 1.0) / 2.0, os.path.join(self.opts.save_dir, f'diffusion_ref0.mp4'))
        # torch.Size([25, 576, 1024, 3])
        return diffusion_results

    def nvs_sparse_view_eval(self):
        start_time = time.time()

        # Ensure rendered_image and gt_image are on cpu and in numpy format, in [0,1] for display
        def to_numpy_img(img_tensor):
            if isinstance(img_tensor, np.ndarray):
                return img_tensor
            img = img_tensor.detach().cpu().numpy()
            # If shape is (C,H,W), transpose to (H,W,C)
            if img.shape[0] == 3 and img.ndim == 3:
                img = np.transpose(img, (1,2,0))
            return np.clip(img, 0, 1)

        c2ws = torch.stack([torch.tensor(pose) for pose in self.camera_poses]).to(self.device).to(torch.float32)
        principal_points = torch.tensor([[self.K[0,2], self.K[1,2]]]).to(self.device).to(torch.float32)
        focals = torch.tensor([[self.K[0,0], self.K[1,1]]]).to(self.device).to(torch.float32)

        # set camera poses to novel views
        gt_c2ws = torch.stack([torch.tensor(pose) for pose in self.gt_camera_poses]).to(self.device).to(torch.float32)
        principal_points = principal_points[0:1].expand(len(self.gt_camera_poses) + 1, -1).to(torch.float32)
        focals = focals[0:1].expand(len(self.gt_camera_poses) + 1, -1).to(torch.float32)

        shape = self.images[0]['true_shape']
        H, W = int(shape[0][0]), int(shape[0][1])
        # pcd = [i.detach() for i in self.scene.get_pts3d(clip_thred=self.opts.dpt_trd)] # a list of points of size whc
        # depth = [i.detach() for i in self.scene.get_depthmaps()]

        # print(pcd[0].shape, pcd[0].device, pcd[0].dtype)
        # print(depth[0].shape, depth[0].device, depth[0].dtype)

        pcd = self.scene_dict['pcd']
        depth = self.scene_dict['depth']

        # print(pcd[0].shape, pcd[0].device, pcd[0].dtype)
        # print(depth[0].shape, depth[0].device, depth[0].dtype)

        if len(self.images) == 2:
            masks = None
            mask_pc = False
        else:
            ## masks for cleaner point cloud
            # self.scene.min_conf_thr = float(self.scene.conf_trf(torch.tensor(self.opts.min_conf_thr)))
            # masks = self.scene.get_masks()
            # depth = self.scene.get_depthmaps()
            # print(masks[0].shape, masks[0].device, masks[0].dtype)
            # print(depth[0].shape, depth[0].device, depth[0].dtype)
            masks = self.scene_dict['masks']
            depth = self.scene_dict['depth']
            # print(masks[0].shape, masks[0].device, masks[0].dtype)
            # print(depth[0].shape, depth[0].device, depth[0].dtype)
            bgs_mask = [dpt > self.opts.bg_trd*(torch.max(dpt[40:-40,:])+torch.min(dpt[40:-40,:])) for dpt in depth]
            masks_new = [m+mb for m, mb in zip(masks,bgs_mask)] 
            masks = to_numpy(masks_new)
            mask_pc = True

        # imgs = np.array(self.scene.imgs)
        # print(imgs.min(), imgs.max())
        # print(imgs.shape, imgs.dtype)
        imgs = np.array(self.scene_dict['images'])
        # print(imgs.min(), imgs.max())
        # print(imgs.shape, imgs.dtype)

        camera_traj, num_views, interp_c2ws = generate_traj_interp2(c2ws, gt_c2ws, H, W, focals, principal_points, self.opts.video_length, self.device)
        render_results, viewmask = run_render(pcd, imgs, masks, H, W, camera_traj, num_views, self.device)
        for i in range(len(self.images)):
            render_results[i*(self.opts.video_length - 1)] = (self.images[i]['img'].squeeze(0).permute(1,2,0)+1.)/2.
        render_results = F.interpolate(render_results.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
        
        save_video(render_results, os.path.join(self.opts.save_dir, f'render.mp4'))
        save_pointcloud_with_normals(imgs, pcd, msk=masks, save_path=os.path.join(self.opts.save_dir, f'pcd.ply') , mask_pc=mask_pc, reduce_pc=False)

        # del self.dust3r
        # del self.scene
        torch.cuda.empty_cache()

        diffusion_results = []
        print(f'Generating {len(self.img_ori)-1} clips\n')
        for i in range(len(self.img_ori)-1 ):
            print(f'Generating clip {i} ...\n')
            diffusion_results.append(
                self.run_diffusion(
                    render_results[i*(self.opts.video_length - 1):self.opts.video_length+i*(self.opts.video_length - 1)], 
                )
            )
        print(f'Finish!\n')
        end_time = time.time()
        print(f'Time taken: {end_time - start_time:.4f} s')
        diffusion_results = torch.cat(diffusion_results)
        save_video((diffusion_results + 1.0) / 2.0, os.path.join(self.opts.save_dir, f'diffusion.mp4'))
        # torch.Size([25, 576, 1024, 3])

        diffusion_results = (diffusion_results + 1.0) / 2.0
        diffusion_results = diffusion_results.to(torch.float32)

        if 87 < len(diffusion_results):
            test_poses = [12, 37, 62, 87]
        else:
            test_poses = [12, 37, 62]
        rendered_images = []
        for i in test_poses:
            rendered_image = diffusion_results[i]
            rendered_images.append(rendered_image)
        rendered_images = torch.stack(rendered_images, dim=0).float().cpu()
        gt_images = torch.stack(self.gt_img_ori, dim=0).float().cpu()

        # downsample
        rendered_images = F.interpolate(rendered_images.permute(0,3,1,2), size=(360, 480), mode='bilinear', align_corners=False)
        gt_images = F.interpolate(gt_images.permute(0,3,1,2), size=(360, 480), mode='bilinear', align_corners=False)

        for i in range(len(rendered_images)):
            rendered_image = rendered_images[i].cpu() #.permute(1,2,0)
            gt_image = gt_images[i].cpu() #.permute(1,2,0)
            
            # rendered_to_show = to_numpy_img(rendered_image)
            # gt_to_show = to_numpy_img(gt_image)

            # Save rendered and ground truth images to the save directory instead of visualizing
            rendered_path = os.path.join(self.opts.save_dir, f"rendered_{i}.png")
            gt_path = os.path.join(self.opts.save_dir, f"gt_{i}.png")
            torchvision.utils.save_image(rendered_image, rendered_path)
            torchvision.utils.save_image(gt_image, gt_path)

        # Compute PSNR
        mse = F.mse_loss(rendered_images, gt_images, reduction='none')
        mse_per_image = mse.reshape(mse.shape[0], -1).mean(dim=1)
        psnr_per_image = -10 * torch.log10(mse_per_image + 1e-8)
        psnr = psnr_per_image.mean().item()

        # SSIM
        ssim_value = self.ssim(rendered_images, gt_images)
        ssim_value = ssim_value.item()

        # LPIPS
        lpips_value = self.lpips(rendered_images, gt_images)
        lpips_value = lpips_value.item()

        print(f"Evaluation Metrics:\n"
                  f"PSNR: {psnr:.3f} dB\n"
                  f"SSIM: {ssim_value:.4f}\n"
                  f"LPIPS: {lpips_value:.4f}")

        eval_metrics = {}
        eval_metrics['psnr'] = psnr
        eval_metrics['ssim'] = ssim_value
        eval_metrics['lpips'] = lpips_value
        eval_metrics['time'] = end_time - start_time
        
        return diffusion_results, eval_metrics

    def nvs_single_view_ref_iterative(self):

        all_results = []
        sample_rate = 6
        idx = 1 #初始包含1张ref image
        for itr in range(0, len(self.opts.d_phi)):
            if itr == 0:
                self.images = [self.images[0]] #去掉后一份copy
                diffusion_results_itr = self.nvs_single_view()
                # diffusion_results_itr = torch.randn([25, 576, 1024, 3]).to(self.device)
                diffusion_results_itr = diffusion_results_itr.permute(0,3,1,2)
                all_results.append(diffusion_results_itr)
            else:
                for i in range(0+sample_rate, diffusion_results_itr.shape[0], sample_rate):
                    self.images.append(get_input_dict(diffusion_results_itr[i:i+1,...],idx,dtype = torch.float32))
                    idx += 1
                self.run_dust3r(input_images=self.images, clean_pc=True)
                diffusion_results_itr = self.nvs_sparse_view(itr)
                # diffusion_results_itr = torch.randn([25, 576, 1024, 3]).to(self.device)
                diffusion_results_itr = diffusion_results_itr.permute(0,3,1,2)
                all_results.append(diffusion_results_itr)
        return all_results

    def nvs_single_view_1drc_iterative(self):

        all_results = []
        sample_rate = 6
        idx = 1 #初始包含1张ref image
        for itr in range(0, len(self.opts.d_phi)):
            if itr == 0:
                self.images = [self.images[0]] #去掉后一份copy
                diffusion_results_itr = self.nvs_single_view()
                # diffusion_results_itr = torch.randn([25, 576, 1024, 3]).to(self.device)
                diffusion_results_itr = diffusion_results_itr.permute(0,3,1,2)
                all_results.append(diffusion_results_itr)
            else:
                for i in range(0+sample_rate, diffusion_results_itr.shape[0], sample_rate):
                    self.images.append(get_input_dict(diffusion_results_itr[i:i+1,...],idx,dtype = torch.float32))
                    idx += 1
                self.run_dust3r(input_images=self.images, clean_pc=True)
                diffusion_results_itr = self.nvs_sparse_view(itr)
                # diffusion_results_itr = torch.randn([25, 576, 1024, 3]).to(self.device)
                diffusion_results_itr = diffusion_results_itr.permute(0,3,1,2)
                all_results.append(diffusion_results_itr)
        return all_results

    def nvs_single_view_nbv(self):
        # lef and right
        # d_theta and a_phi 是搜索空间的顶点间隔
        all_results = []
        ## FIXME: hard coded
        sample_rate = 6
        max_itr = 3

        idx = 1 #初始包含1张ref image
        for itr in range(0, max_itr):
            if itr == 0:
                self.images = [self.images[0]] #去掉后一份copy
                diffusion_results_itr = self.nvs_single_view()
                # diffusion_results_itr = torch.randn([25, 576, 1024, 3]).to(self.device)
                diffusion_results_itr = diffusion_results_itr.permute(0,3,1,2)
                all_results.append(diffusion_results_itr)
            else:
                for i in range(0+sample_rate, diffusion_results_itr.shape[0], sample_rate):
                    self.images.append(get_input_dict(diffusion_results_itr[i:i+1,...],idx,dtype = torch.float32))
                    idx += 1
                self.run_dust3r(input_images=self.images, clean_pc=True)
                diffusion_results_itr = self.nvs_sparse_view(itr)
                # diffusion_results_itr = torch.randn([25, 576, 1024, 3]).to(self.device)
                diffusion_results_itr = diffusion_results_itr.permute(0,3,1,2)
                all_results.append(diffusion_results_itr)
        return all_results

    def setup_diffusion(self):
        seed_everything(self.opts.seed)

        config = OmegaConf.load(self.opts.config)
        model_config = config.pop("model", OmegaConf.create())

        ## set use_checkpoint as False as when using deepspeed, it encounters an error "deepspeed backend not set"
        model_config['params']['unet_config']['params']['use_checkpoint'] = False
        model = instantiate_from_config(model_config)
        model = model.to(self.device)
        model.cond_stage_model.device = self.device
        model.perframe_ae = self.opts.perframe_ae
        assert os.path.exists(self.opts.ckpt_path), "Error: checkpoint Not Found!"
        model = load_model_checkpoint(model, self.opts.ckpt_path)
        model.eval()
        self.diffusion = model

        h, w = self.opts.height // 8, self.opts.width // 8
        channels = model.model.diffusion_model.out_channels
        n_frames = self.opts.video_length
        self.noise_shape = [self.opts.bs, channels, n_frames, h, w]

    def setup_dust3r(self):
        self.dust3r = load_model(self.opts.model_path, self.device)
    
    def load_initial_images(self, image_dir):
        ## load images
        ## dict_keys(['img', 'true_shape', 'idx', 'instance', 'img_ori']),张量形式
        images = load_images([image_dir], size=512,force_1024 = True)
        img_ori = (images[0]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2. # [576,1024,3] [0,1]

        if len(images) == 1:
            images = [images[0], copy.deepcopy(images[0])]
            images[1]['idx'] = 1

        return images, img_ori

    def load_initial_dir(self, image_dir):

        image_files = glob.glob(os.path.join(image_dir, "*"))

        if len(image_files) < 2:
            raise ValueError("Input views should not less than 2.")
        image_files = sorted(image_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        images = load_images(image_files, size=512,force_1024 = True)

        img_gts = []
        for i in range(len(image_files)):
            img_gts.append((images[i]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2.) 

        return images, img_gts

    def load_data_from_json(self, data_dir, json_file, scene_id):
        def __blender2opencv_cam(cam_T):
            # blender camera to opengl camera
            y_z_swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
            cam_T = y_z_swap.dot(cam_T)
            cam_T[:3, 0] *= -1

            # opengl camera to opencv camera
            cam_T[:3, 1] *= -1
            cam_T[:3, 2] *= -1

            return cam_T

        def __backproject(depth, extrinsics, K):
            u, v = np.meshgrid(range(depth.shape[1]), range(depth.shape[0]))
            u = u.reshape([1, -1])[0]
            v = v.reshape([1, -1])[0]

            z = depth[v, u]

            # calculate coordinates
            x = (u - K[0][2]) * z / K[0][0]
            y = (v - K[1][2]) * z / K[1][1]

            point_cam = np.vstack([x, y, z]).T

            # camera to world coordinates
            R = extrinsics[:3, :3]
            T = extrinsics[:3, 3]
            point_canonical = point_cam.dot(R.T) + T

            point_canonical = point_canonical.reshape(depth.shape[0], depth.shape[1], 3)
            point_canonical = torch.tensor(point_canonical, dtype=torch.float32, device=self.device)

            return point_canonical
        
        if "scannet" in data_dir:
            # parse json file for image file names
            with open(json_file, 'r') as file:
                data = json.load(file)
            input_image_files = [os.path.join(data_dir, scene_id, "dslr", "render_rgb", i) for i in data[scene_id]['input_view_ids']]
            gt_novel_view_image_files = [os.path.join(data_dir, scene_id, "dslr", "render_rgb", i) for i in data[scene_id]['render_view_ids']]

            # load input images
            input_images = load_images(input_image_files, size=512, force_1024=False)
            input_images_ori = []
            for i in range(len(input_image_files)):
                input_images_ori.append((input_images[i]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2.)

            # load gt images
            gt_novel_view_images = load_images(gt_novel_view_image_files, size=512, force_1024=False)
            gt_novel_view_images_ori = []
            for i in range(len(gt_novel_view_image_files)):
                gt_novel_view_images_ori.append((gt_novel_view_images[i]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2.)

            # load camera intrinsics
            intrinsics_file = os.path.join(data_dir, scene_id, "dslr", "colmap", "cameras.txt")
            colmap_camera = list(read_cameras_text(intrinsics_file).values())[0]
            pinhole_camera = undistorted_camera(colmap_camera)
            K = camera_to_intrinsic(pinhole_camera)

            # update intrinsics based on resizing and cropping
            height, width = pinhole_camera.height, pinhole_camera.width
            s_x = 480.0 / float(width)
            s_y = 360.0 / float(height)
            K[0, 0] *= s_x
            K[1, 1] *= s_y
            K[0, 2] = 480.0 / 2.
            K[1, 2] = 360.0 / 2.
            ratio2 = 480.0 / 512.0
            scale2 = 1.0 / ratio2
            K[:2, :] *= scale2
            K[1, 2] = np.round(K[1, 2])

            # load camera poses
            input_camera_poses = []
            for i in range(len(input_image_files)):
                image_name = input_image_files[i].split('/')[-1].split('.')[0]
                pose_file = os.path.join(data_dir, scene_id, "dslr", "pose", f"{image_name}.txt")
                cam_T = np.loadtxt(pose_file)
                input_camera_poses.append(cam_T)
            gt_novel_view_camera_poses = []
            for i in range(len(gt_novel_view_image_files)):
                image_name = gt_novel_view_image_files[i].split('/')[-1].split('.')[0]
                pose_file = os.path.join(data_dir, scene_id, "dslr", "pose", f"{image_name}.txt")
                cam_T = np.loadtxt(pose_file)
                gt_novel_view_camera_poses.append(cam_T)

            scene = {}
            images = []
            depth = []
            masks = []
            pcd = []
            for i in range(len(input_image_files)):
                images.append((input_images[i]['img'].squeeze(0).permute(1,2,0)+1.)/2.)
                depth.append(torch.tensor(input_images[i]['depth'][0], dtype=torch.float32, device=self.device))
                mask = torch.ones_like(depth[-1], dtype=torch.bool, device=self.device)
                mask[depth[-1] == 0.0] = False
                masks.append(mask)
                pcd.append(__backproject(input_images[i]['depth'][0], input_camera_poses[i], K))
            scene['images'] = torch.stack(images, dim=0).numpy()
            scene['depth'] = depth
            scene['masks'] = masks
            scene['pcd'] = pcd

            return input_images, input_images_ori, input_camera_poses, gt_novel_view_images, \
                gt_novel_view_images_ori, gt_novel_view_camera_poses, K, scene
        elif "3D-FRONT" in data_dir:
            # parse json file for image file names
            with open(json_file, 'r') as file:
                data = json.load(file)
            scene_name = scene_id.split(' ')[0]
            room_name = scene_id.split(' ')[1]
            input_image_files = [os.path.join(data_dir, scene_name, room_name, i) for i in data[scene_id]['input_view_ids']]
            gt_novel_view_image_files = [os.path.join(data_dir, scene_name, room_name, i) for i in data[scene_id]['render_view_ids']]

             # load input images
            input_images = load_images(input_image_files, size=512, force_1024=False)
            input_images_ori = []
            for i in range(len(input_image_files)):
                input_images_ori.append((input_images[i]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2.)

            # load gt images
            gt_novel_view_images = load_images(gt_novel_view_image_files, size=512, force_1024=False)
            gt_novel_view_images_ori = []
            for i in range(len(gt_novel_view_image_files)):
                gt_novel_view_images_ori.append((gt_novel_view_images[i]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2.)

            K = np.load(os.path.join(data_dir, "cam_K.npy"))

            # update intrinsics based on resizing and cropping
            ratio2 = 480.0 / 512.0
            scale2 = 1.0 / ratio2
            K[:2, :] *= scale2
            K[1, 2] = np.round(K[1, 2])

            # load camera poses
            input_camera_poses = []
            for i in range(len(input_image_files)):
                with h5py.File(input_image_files[i]) as f:
                    cam_T = __blender2opencv_cam(np.array(f["cam_Ts"]))
                input_camera_poses.append(cam_T)
            gt_novel_view_camera_poses = []
            for i in range(len(gt_novel_view_image_files)):
                with h5py.File(gt_novel_view_image_files[i]) as f:
                    cam_T = __blender2opencv_cam(np.array(f["cam_Ts"]))
                gt_novel_view_camera_poses.append(cam_T)

            scene = {}
            images = []
            depth = []
            masks = []
            pcd = []
            for i in range(len(input_image_files)):
                images.append((input_images[i]['img'].squeeze(0).permute(1,2,0)+1.)/2.)
                depth.append(torch.tensor(input_images[i]['depth'][0], dtype=torch.float32, device=self.device))
                mask = torch.ones_like(depth[-1], dtype=torch.bool, device=self.device)
                mask[depth[-1] == 0.0] = False
                masks.append(mask)
                pcd.append(__backproject(input_images[i]['depth'][0], input_camera_poses[i], K))
            scene['images'] = torch.stack(images, dim=0).numpy()
            scene['depth'] = depth
            scene['masks'] = masks
            scene['pcd'] = pcd

            return input_images, input_images_ori, input_camera_poses, gt_novel_view_images, \
                gt_novel_view_images_ori, gt_novel_view_camera_poses, K, scene
        else:
            raise ValueError(f"Invalid dataset: {data_dir}")

    def run_gradio(self,i2v_input_image, i2v_elevation, i2v_center_scale, i2v_d_phi, i2v_d_theta, i2v_d_r, i2v_steps, i2v_seed):
        self.opts.elevation = float(i2v_elevation)
        self.opts.center_scale = float(i2v_center_scale)
        self.opts.ddim_steps = i2v_steps
        self.gradio_traj = [float(i) for i in i2v_d_phi.split()],[float(i) for i in i2v_d_theta.split()],[float(i) for i in i2v_d_r.split()]
        seed_everything(i2v_seed)
        torch.cuda.empty_cache()
        img_tensor = torch.from_numpy(i2v_input_image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        img_tensor = (img_tensor / 255. - 0.5) * 2

        image_tensor_resized = center_crop_image(img_tensor) #1,3,h,w
        images = get_input_dict(image_tensor_resized,idx = 0,dtype = torch.float32)
        images = [images, copy.deepcopy(images)]
        images[1]['idx'] = 1
        self.images = images
        self.img_ori = (image_tensor_resized.squeeze(0).permute(1,2,0) + 1.)/2.

        # self.images: torch.Size([1, 3, 288, 512]), [-1,1]
        # self.img_ori:  torch.Size([576, 1024, 3]), [0,1]
        # self.images, self.img_ori = self.load_initial_images(image_dir=i2v_input_image)
        self.run_dust3r(input_images=self.images)
        self.nvs_single_view(gradio=True)

        traj_dir = os.path.join(self.opts.save_dir, "viz_traj.mp4")
        gen_dir = os.path.join(self.opts.save_dir, "diffusion0.mp4")
        
        return traj_dir, gen_dir