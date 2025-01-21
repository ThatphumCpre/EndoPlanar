#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import imageio
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, FDMHiddenParams
from scene.flexible_deform_model import GaussianModel
from time import time
import copy
import open3d as o3d
from utils.graphics_utils import fov2focal
import cv2

def gen_pseudo_pcd_gt(rgb, depth, K, pose, depth_scale=1., depth_truc=3., depth_filter=None):
    """Generate point cloud.
    """
    if depth_filter is not None:
         depth = cv2.bilateralFilter(depth, depth_filter[0], depth_filter[1], depth_filter[2])
    h, w = rgb.shape[:-1]
    rgb_im = o3d.geometry.Image(rgb.astype(np.uint8))
    depth_im = o3d.geometry.Image(depth)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_im, 
        depth_im, 
        depth_scale=depth_scale,
        depth_trunc=depth_truc/depth_scale,
        convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(w, h, K[:3, :3]),
        pose,
        project_valid_depth_only=True,
    )
    return pcd

def clean_mesh(mesh, min_len=1000):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_len
    mesh_0 = copy.deepcopy(mesh)
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    return mesh_0

def post_process_mesh(mesh, cluster_to_keep=1):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0

def get_intrinsics(camera):
    """
    Return a 3x3 pinhole intrinsic matrix from either:
      - The camera.K parameter (if provided), or
      - The camera's FoVx/FoVy and image width/height.
    """

    # Otherwise, derive from FoV and image size
    w = camera.image_width
    h = camera.image_height
    fx = w / (2.0 * np.tan(camera.FoVx / 2.0))  # focal length x
    fy = h / (2.0 * np.tan(camera.FoVy / 2.0))  # focal length y
    cx = w / 2.0
    cy = h / 2.0
    
    K = np.array([
        [fx,  0,  cx],
        [ 0, fy,  cy],
        [ 0,  0,   1]
    ], dtype=np.float32)
    return K

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background,  voxel_size, num_cluster,\
    no_fine, render_test=False, reconstruct=False, crop_size=0, max_depth=5.0, volume=None, use_depth_filter=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    gtdepth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_depth")
    masks_path = os.path.join(model_path, name, "ours_{}".format(iteration), "masks")
    mesh_path = os.path.join(model_path, name, "ours_{}".format(iteration), "meshs")
    processed_mesh_path = os.path.join(model_path, name, "ours_{}".format(iteration), "processed_meshs")
    normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normals")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(gtdepth_path, exist_ok=True)
    makedirs(masks_path, exist_ok=True)
    makedirs(mesh_path, exist_ok=True)
    makedirs(processed_mesh_path, exist_ok=True)
    makedirs(normal_path, exist_ok=True)
    
    render_images = []
    render_depths = []
    gt_list = []
    gt_depths = []
    mask_list = []
    pseudo_gt_pcds = []
    depths_tsdf_fusion = []
    normals = []
    count = 0

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        stage = 'coarse' if no_fine else 'fine'
        # rgb, depth, K, pose
        gt_rgb = view.original_image[0:3, :, :]
        gt_depth = view.original_depth
        K = get_intrinsics(view)
        pose = view.world_view_transform.inverse().cpu().numpy()  # shape [4,4]
        # TODO: add apply mask
        pseudo_pcd_gt = gen_pseudo_pcd_gt(gt_rgb, gt_depth, K, pose)

        # render
        rendering = render(view, gaussians, pipeline, background)
        render_rgb = rendering["render"].cpu()
        depth = rendering["plane_depth"].squeeze()
        depth_tsdf = depth.clone()
        depth = depth.detach().cpu().numpy()

        # PGSR normalize?
        # depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
        # depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
        # depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)

        normal = rendering["rendered_normal"].permute(1,2,0)
        normal = normal/(normal.norm(dim=-1, keepdim=True)+1.0e-8)
        normal = normal.detach().cpu().numpy()
        normal = ((normal+1) * 127.5).astype(np.uint8).clip(0, 255)

        # reject depth measurements for pixels whose surface normal is too oblique relative to the camera’s viewing direction. This often corresponds to areas where the geometry is nearly parallel to the line of sight
        if use_depth_filter:
            view_dir = torch.nn.functional.normalize(view.get_rays(), p=2, dim=-1)
            depth_normal = rendering["depth_normal"].permute(1,2,0)
            depth_normal = torch.nn.functional.normalize(depth_normal, p=2, dim=-1)
            dot = torch.sum(view_dir*depth_normal, dim=-1).abs()
            angle = torch.acos(dot)
            mask = angle > (80.0 / 180 * 3.14159)
            depth_tsdf[mask] = 0
        depths_tsdf_fusion.append(depth_tsdf.squeeze().cpu())

        if name in ["train", "test", "video"]:
            gt_list.append(gt_rgb)
            mask = view.mask
            mask_list.append(mask)
            gt_depths.append(gt_depth)
            pseudo_gt_pcds.append(pseudo_pcd_gt)
            normals.append(normal)
    
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=voxel_size,
                sdf_trunc=4.0*voxel_size,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

        depths_tsdf_fusion = torch.stack(depths_tsdf_fusion, dim=0)
        for idx, view in enumerate(tqdm(views, desc="TSDF Fusion progress")):
            ref_depth = depths_tsdf_fusion[idx].cuda()

            if view.mask is not None:
                ref_depth[view.mask.squeeze() < 0.5] = 0
            ref_depth[ref_depth>max_depth] = 0
            ref_depth = ref_depth.detach().cpu().numpy()
            
            pose = np.identity(4)
            pose[:3,:3] = view.R.transpose(-1,-2)
            pose[:3, 3] = view.T
            color = o3d.io.read_image(os.path.join(render_path, view.image_name + ".jpg"))
            depth = o3d.geometry.Image((ref_depth*1000).astype(np.uint16))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_scale=1000.0, depth_trunc=max_depth, convert_rgb_to_intensity=False)
            volume.integrate(
                rgbd,
                o3d.camera.PinholeCameraIntrinsic(view.image_width, view.image_height, view.Fx, view.Fy, view.Cx, view.Cy),
                pose)
            
        mesh = volume.extract_triangle_mesh()
        
        o3d.io.write_triangle_mesh(os.path.join(mesh_path, '{0:05d}'.format(count) + "tsdf_fusion.ply") , mesh,
                                    write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
        
        processed_mesh = post_process_mesh(mesh, num_cluster)
        o3d.io.write_triangle_mesh(os.path.join(processed_mesh_path, '{0:05d}'.format(count) + "tsdf_fusion.ply"), processed_mesh, 
                                    write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
        count += 1

    if render_test:
        test_times = 20
        for i in range(test_times):
            for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
                if idx == 0 and i == 0:
                    time1 = time()
                stage = 'coarse' if no_fine else 'fine'
                rendering = render(view, gaussians, pipeline, background)
        time2=time()
        print("FPS:",(len(views)-1)*test_times/(time2-time1))
    
    count = 0
    print("writing training images.")
    if len(gt_list) != 0:
        for image in tqdm(gt_list):
            torchvision.utils.save_image(image, os.path.join(gts_path, '{0:05d}'.format(count) + ".png"))
            count+=1
            
    count = 0
    print("writing rendering images.")
    if len(render_images) != 0:
        for image in tqdm(render_images):
            torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(count) + ".png"))
            count +=1
    
    count = 0
    print("writing mask images.")
    if len(mask_list) != 0:
        for image in tqdm(mask_list):
            image = image.float()
            torchvision.utils.save_image(image, os.path.join(masks_path, '{0:05d}'.format(count) + ".png"))
            count +=1
    
    count = 0
    print("writing rendered depth images.")
    if len(render_depths) != 0:
        for image in tqdm(render_depths):
            image = np.clip(image.cpu().squeeze().numpy().astype(np.uint8), 0, 255)
            cv2.imwrite(os.path.join(depth_path, '{0:05d}'.format(count) + ".png"), image)
            count += 1
    
    count = 0
    print("writing gt depth images.")
    if len(gt_depths) != 0:
        for image in tqdm(gt_depths):
            image = image.cpu().squeeze().numpy().astype(np.uint8)
            cv2.imwrite(os.path.join(gtdepth_path, '{0:05d}'.format(count) + ".png"), image)
            count += 1

    count = 0
    print("writing gt normal images.")
    if len(normals) != 0:
        for image in tqdm(normals):
            image = image.cpu().squeeze().numpy().astype(np.uint8)
            cv2.imwrite(os.path.join(normal_path, '{0:05d}'.format(count) + ".jpg"), image)
            count += 1
            
    render_array = torch.stack(render_images, dim=0).permute(0, 2, 3, 1)
    render_array = (render_array*255).clip(0, 255).cpu().numpy().astype(np.uint8) # BxHxWxC
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'ours_video.mp4'), render_array, fps=30, quality=8)
    
    gt_array = torch.stack(gt_list, dim=0).permute(0, 2, 3, 1)
    gt_array = (gt_array*255).clip(0, 255).cpu().numpy().astype(np.uint8)
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'gt_video.mp4'), gt_array, fps=30, quality=8)
                    
    FoVy, FoVx, height, width = view.FoVy, view.FoVx, view.image_height, view.image_width
    focal_y, focal_x = fov2focal(FoVy, height), fov2focal(FoVx, width)
    camera_parameters = (focal_x, focal_y, width, height)
    

    if reconstruct:
        print('file name:', name)
        reconstruct_point_cloud(render_images, mask_list, render_depths, camera_parameters, name, crop_size)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,
                 max_depth : float, voxel_size : float, num_cluster: int, use_depth_filter : bool, skip_video: bool, reconstruct_train: bool, reconstruct_test: bool, reconstruct_video: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        # load model from model save_point at iteration load_iteration=iteration
        scene = Scene(dataset, gaussians, load_iteration=iteration)
        # after this line the guassian model is ready to use

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, False, reconstruct=reconstruct_train)
        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, False, reconstruct=reconstruct_test, crop_size=20)
        if not skip_video:
            render_set(dataset.model_path,"video",scene.loaded_iter, scene.getVideoCameras(),gaussians,pipeline,background, False, render_test=True, reconstruct=reconstruct_video, crop_size=20)

def reconstruct_point_cloud(images, masks, depths, camera_parameters, name, crop_left_size=0):
    import cv2
    import copy
    output_frame_folder = os.path.join("reconstruct", name)
    os.makedirs(output_frame_folder, exist_ok=True)
    frames = np.arange(len(images))
    # frames = [0]
    focal_x, focal_y, width, height = camera_parameters
    if crop_left_size > 0:
        width = width - crop_left_size
        height = height - crop_left_size//2
    for i_frame in frames:
        rgb_tensor = images[i_frame]
        rgb_np = rgb_tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).contiguous().to("cpu").numpy()
        depth_np = depths[i_frame].cpu().numpy()
        if len(depth_np.shape) == 3:
            depth_np = depth_np[0]
        # depth_np = depth_np.squeeze(0)
        if crop_left_size > 0:
            rgb_np = rgb_np[:, crop_left_size:, :]
            depth_np = depth_np[:, crop_left_size:]
            rgb_np = rgb_np[:-crop_left_size//2, :, :]
            depth_np = depth_np[:-crop_left_size//2, :]
            
        # mask = masks[i_frame]
        # mask = mask.squeeze(0).cpu().numpy()
        
        rgb_new = copy.deepcopy(rgb_np)
        # depth_np[mask == 0] =0
        # rgb_new[mask ==0] = np.asarray([0,0,0]) 
        depth_smoother = (32, 64, 32) # (128, 64, 64) #[24, 64, 32]
        # print(depth_np.shape)
        depth_np = cv2.bilateralFilter(depth_np, depth_smoother[0], depth_smoother[1], depth_smoother[2])
        
        close_depth = np.percentile(depth_np[depth_np!=0], 5)
        inf_depth = np.percentile(depth_np, 95)
        depth_np = np.clip(depth_np, close_depth, inf_depth)

        rgb_im = o3d.geometry.Image(rgb_new.astype(np.uint8))
        depth_im = o3d.geometry.Image(depth_np)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_im, depth_im, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(int(width), int(height), focal_x, focal_y, width / 2, height / 2),
            project_valid_depth_only=True
        )
        o3d.io.write_point_cloud(os.path.join(output_frame_folder, 'frame_{}.ply'.format(i_frame)), pcd)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = FDMHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--reconstruct_train", action="store_true")
    parser.add_argument("--reconstruct_test", action="store_true")
    parser.add_argument("--reconstruct_video", action="store_true")
    parser.add_argument("--configs", type=str)
    args = get_combined_args(parser)
    print("Rendering ", args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, 
        pipeline.extract(args), 
        args.skip_train, args.skip_test, args.skip_video,
        args.reconstruct_train,args.reconstruct_test,args.reconstruct_video)