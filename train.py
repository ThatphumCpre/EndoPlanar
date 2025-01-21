#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np
import random
import os 
import torch
from random import randint
from utils.loss_utils import l1_loss, get_img_grad_weight
from gaussian_renderer import render

import sys
from scene import  Scene
from scene.flexible_deform_model import GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from arguments import FDMHiddenParams as ModelHiddenParams
from utils.timer import Timer
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# import lpips
from utils.scene_utils import render_training_image

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# import torch

class ViewpointDataset(Dataset):
    def __init__(self, viewpoint_stack):
        # Save the list of viewpoint cameras
        self.viewpoint_stack = viewpoint_stack

    def __len__(self):
        return len(self.viewpoint_stack)

    def __getitem__(self, index):
        # Return the viewpoint at this index
        return self.viewpoint_stack[index]


def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, tb_writer, train_iter, timer):
    first_iter = 0
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None

    
    final_iter = train_iter
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    
    # lpips_model = lpips.LPIPS(net="vgg").cuda()
    """
    cameras.append(Camera(colmap_id=idx, R=R, T=T, FoVx=FovX, FoVy=FovY,image=image, depth=depth, mask=mask, gt_alpha_mask=None,
                    image_name=f"{idx}", uid=idx, data_device=torch.device("cuda"), time=time,
                    Znear=None, Zfar=None, K=self.K, h=self.img_wh[1], w=self.img_wh[0]))
    """
    video_cams = scene.getVideoCameras()
    
    if not viewpoint_stack:
        # same format as video_cams but sampling for training
        # list Camera
        viewpoint_stack = scene.getTrainCameras()
        
    for iteration in range(first_iter, final_iter+1):        

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        # sampling one idx/time_step in dataset
        # seem like stocastic GD with batch size 1?
        # ???? epoch ????
        idx = randint(0, len(viewpoint_stack)-1)
        viewpoint_cams = [viewpoint_stack[idx]]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
            
        images = []
        depths = []
        gt_images = []
        gt_depths = []
        masks = []
        
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []

        normals = []
        depth_normals = []
        # scales_finals = []
        scales_final = torch.empty(0)
        
        for viewpoint_cam in viewpoint_cams:
            # render: 
            #   (1) extract time_step find deformation using self._deformation to get deformation of that time step t
            #   (2) add 3DGuassians with its deformation (new object)
            #   (3) call CUDA kernel for rasterization via submodules/depth-diff-gaussian-rasterization
            #       rendered_image, radii, depth = rasterizer(...)
            render_pkg = render(viewpoint_cam, gaussians, pipe, background,
                            return_plane=True, return_depth_normal=True)
            image, viewspace_point_tensor, visibility_filter, radii = \
                render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
            gt_image = viewpoint_cam.original_image.cuda().float()
            gt_depth = viewpoint_cam.original_depth.cuda().float()
            mask = viewpoint_cam.mask.cuda()
            
            images.append(image.unsqueeze(0))
            depths.append(render_pkg["plane_depth"].unsqueeze(0))
            gt_images.append(gt_image.unsqueeze(0))
            gt_depths.append(gt_depth.unsqueeze(0))
            masks.append(mask.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            normals.append(render_pkg["rendered_normal"].unsqueeze(0))
            depth_normals.append(render_pkg["depth_normal"].unsqueeze(0))
            scales_final = render_pkg["scales_final"]
            # seem like it not being used at all <= 2Dmeans projection with thier gradient
            viewspace_point_tensor_list.append(viewspace_point_tensor)
        
        # written like it can be multiple scene trained it one iteration, but actually was one ...
        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images,0)
        depth_tensor = torch.cat(depths, 0)
        gt_image_tensor = torch.cat(gt_images,0)
        gt_depth_tensor = torch.cat(gt_depths, 0)
        mask_tensor = torch.cat(masks, 0)

        normal = torch.cat(normals, 0)
        depth_normal = torch.cat(depth_normals, 0)
        
        # abs error
        Ll1 = l1_loss(image_tensor, gt_image_tensor, mask_tensor)
        loss = Ll1.clone()
        
        if (gt_depth_tensor!=0).sum() < 10:
            depth_loss = torch.tensor(0.).cuda()
        else:
            depth_tensor[depth_tensor!=0] = 1 / depth_tensor[depth_tensor!=0]
            gt_depth_tensor[gt_depth_tensor!=0] = 1 / gt_depth_tensor[gt_depth_tensor!=0]

            # abs error but depth
            depth_loss = l1_loss(depth_tensor, gt_depth_tensor, mask_tensor)

        # scale loss
        if visibility_filter.sum() > 0:
            # TODO: change get_scaling to deform_scaling
            # deform = exp(self._scale + sum(basis*weight)) >= 0
            scale = scales_final[visibility_filter]
            sorted_scale, _ = torch.sort(scale, dim=-1)
            min_scale_loss = sorted_scale[...,0]
            scaling_loss = opt.scale_loss_weight * min_scale_loss.mean()

        # single-view loss
        normal_loss = 0
        if iteration > opt.single_view_weight_from_iter:
            weight = opt.single_view_weight

            image_weight = (1.0 - get_img_grad_weight(gt_image))
            image_weight = (image_weight).clamp(0,1).detach() ** 2
            if not opt.wo_image_weight:
                # image_weight = erode(image_weight[None,None]).squeeze()
                normal_loss = weight * (image_weight * (((depth_normal - normal)).abs().sum(0))).mean()
            else:
                normal_loss = weight * (((depth_normal - normal)).abs().sum(0)).mean()
        # catch psnr
        psnr_ = psnr(image_tensor, gt_image_tensor, mask_tensor).mean().double()
        
        # combine loss
        loss = Ll1 + depth_loss + scaling_loss + normal_loss
        
        # cal grad
        loss.backward()

        # seem like it tery to copy the grad out for further use?
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss.item():.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point":f"{total_point}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            timer.pause()
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, [pipe, background])
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, 'fine')
            timer.start()
            
            # Densification
            # opt.densify_until_iter: few last step we may not want to densify it anymore since it will need amount of iteration after doing it, 
            #                         so, the last few step we focus on optimize it directly
            if iteration < opt.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                # this occur EVERY ITERATION , it catch the max radii2D for prunning
                mask = (render_pkg["out_observe"] > 0) & visibility_filter
                gaussians.max_radii2D[mask] = torch.max(gaussians.max_radii2D[mask], radii[mask])
                viewspace_point_tensor_abs = render_pkg["viewspace_points_abs"]
                gaussians.add_densification_stats(viewspace_point_tensor_grad, viewspace_point_tensor_abs, visibility_filter)
                # gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                # calculate thds for opacity prunning and densification
                # they are linear function that decrease over time. 
                # prune_mask = (self.get_opacity < min_opacity).squeeze(): opacity_threshold decreasing => less prunning over time
                opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                # torch.norm(grads, dim=-1) >= grad_threshold: grad_threshold decreasing => more densify over time
                densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter )
                densify_threshold_abs = opt.densify_grad_threshold_abs_fine_init - iteration*(opt.densify_grad_threshold_abs_fine_init - opt.densify_grad_threshold_abs_after)/(opt.densify_until_iter )  

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify(densify_threshold, densify_threshold_abs, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0:
                    size_threshold = 40 if iteration > opt.opacity_reset_interval else None
                    gaussians.prune(densify_threshold, densify_threshold_abs, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    print("reset opacity")
                    gaussians.reset_opacity()
                    
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname, extra_mark):
    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians)
    timer.start()
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, tb_writer, opt.iterations,timer)

def prepare_output_and_logger(expname):    
    if not args.model_path:
        unique_str = expname
        args.model_path = os.path.join("./output/", unique_str)
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    
    if tb_writer:
        tb_writer.add_scalar(f'train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'iter_time', elapsed, iteration)


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i*500 for i in range(0,120)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3000,])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "endonerf/pulling_fdm")
    parser.add_argument("--configs", type=str, default = "arguments/endonerf/default.py")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, \
        args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname, args.extra_mark)

    # All done
    print("\nTraining complete.")
