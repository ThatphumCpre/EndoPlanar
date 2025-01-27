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

import torch
import math
from diff_plane_rasterization import GaussianRasterizationSettings as PlaneGaussianRasterizationSettings
from diff_plane_rasterization import GaussianRasterizer as PlaneGaussianRasterizer
# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.flexible_deform_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import normal_from_depth_image

def render_normal(viewpoint_cam, depth, offset=None, normal=None, scale=1):
    # depth: (H, W), bg_color: (3), alpha: (H, W)
    # normal_ref: (3, H, W)
    w = viewpoint_cam.image_width
    h = viewpoint_cam.image_height
    fx = w / (2.0 * math.tan(viewpoint_cam.FoVx / 2.0))  # focal length x
    fy = h / (2.0 * math.tan(viewpoint_cam.FoVy / 2.0))  # focal length y
    cx = w / 2.0
    cy = h / 2.0
    
    intrinsic_matrix = torch.tensor([
        [fx/scale,  0,        cx/scale],
        [ 0      , fy/scale,  cy/scale],
        [ 0      ,  0,               1]
    ]).float()
    extrinsic_matrix = viewpoint_cam.world_view_transform.transpose(0,1).contiguous() # cam2world
    # return intrinsic_matrix, extrinsic_matrix
    # intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf(scale=scale)
    st = max(int(scale/2)-1,0)
    if offset is not None:
        offset = offset[st::scale,st::scale]
    normal_ref = normal_from_depth_image(depth[st::scale,st::scale], 
                                            intrinsic_matrix.to(depth.device), 
                                            extrinsic_matrix.to(depth.device), offset)

    normal_ref = normal_ref.permute(2,0,1)
    return normal_ref

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, 
           return_plane =True, return_depth_normal = True):

    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # seem like they initial 2Dmeans with zeros and set it to retain grad, 
    # send the its pointer to the CUDA rasterization kernel, 
    # and the CUDA rasterization kernel update/store the projection in this 2Dmeans so that we can inspect the gradient in python env if needed
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_abs = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
        screenspace_points_abs.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = PlaneGaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            render_geo=return_plane,
            debug=pipe.debug
        )

    rasterizer = PlaneGaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    # add deformation to each points
    # deformation = pc.get_deformation
    means3D = pc.get_xyz
    ori_time = torch.tensor(viewpoint_camera.time).to(means3D.device)
    means2D = screenspace_points
    means2D_abs = screenspace_points_abs
    opacity = pc._opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling 
        rotations = pc._rotation
    deformation_point = pc._deformation_table

    means3D_deform, scales_deform, rotations_deform = pc.deformation(means3D[deformation_point], scales[deformation_point], 
                                                                         rotations[deformation_point],
                                                                         ori_time)
    opacity_deform = opacity[deformation_point]
        
    # print(time.max())
    with torch.no_grad():
        pc._deformation_accum[deformation_point] += torch.abs(means3D_deform - means3D[deformation_point])

    means3D_final = torch.zeros_like(means3D)
    rotations_final = torch.zeros_like(rotations)
    scales_final = torch.zeros_like(scales)
    opacity_final = torch.zeros_like(opacity)
    means3D_final[deformation_point] =  means3D_deform
    rotations_final[deformation_point] =  rotations_deform
    scales_final[deformation_point] =  scales_deform
    opacity_final[deformation_point] = opacity_deform
    means3D_final[~deformation_point] = means3D[~deformation_point]
    rotations_final[~deformation_point] = rotations[~deformation_point]
    scales_final[~deformation_point] = scales[~deformation_point]
    opacity_final[~deformation_point] = opacity[~deformation_point]

    scales_final = pc.scaling_activation(scales_final)
    # print("\n\n\n\n", scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity_final = pc.opacity_activation(opacity)

    # means3D_final = means3D_final.cuda()
    # rotations_final = rotations_final.cuda()
    # scales_final = scales_final.cuda()
    # opacity_final = opacity_final.cuda()

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    if not return_plane:
        rendered_image, radii, out_observe, out_all_map, plane_depth = rasterizer(
            means3D = means3D_final,
            means2D = means2D,
            means2D_abs = means2D_abs,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity_final,
            scales = scales_final,
            rotations = rotations_final,
            cov3D_precomp = cov3D_precomp)
        
        return_dict =  {"render": rendered_image,
                        "viewspace_points": screenspace_points,
                        "viewspace_points_abs": screenspace_points_abs,
                        "visibility_filter" : radii > 0,
                        "radii": radii,
                        "out_observe": out_observe}

        return return_dict

    # TODO: modify get_normal from deform

    # print(means3D_final.shape, means3D_final.device)
    # print(means2D.shape, means2D.device)
    # print(means2D_abs.shape, means2D_abs.device)
    # print(shs.shape, shs.device)
    # # print(colors_precomp.shape, colors_precomp.device)
    # print(opacity_final.shape, opacity_final.device)
    # print(scales_final.shape, scales_final.device)
    # print(rotations_final.shape, rotations_final.device)
    # print(input_all_map.shape, input_all_map.device)
    # # print(cov3D_precomp.shape, cov3D_precomp.device)
    # camera_center = viewpoint_camera.camera_center.cuda()
    # world_view_transform = viewpoint_camera.world_view_transform.cuda()

    # global_normal = pc.get_normal(means3D_final, scales_final, rotations_final, camera_center)
    # local_normal = global_normal @ world_view_transform[:3,:3]
    # pts_in_cam = means3D_final @ world_view_transform[:3,:3] + world_view_transform[3,:3]
    # depth_z = pts_in_cam[:, 2]
    # # local_normal @ pts_in_cam
    # local_distance = (local_normal * pts_in_cam).sum(-1).abs()
    # input_all_map = torch.zeros((means3D_final.shape[0], 5)).cuda().float()
    # input_all_map[:, :3] = local_normal
    # input_all_map[:, 3] = 1.0
    # input_all_map[:, 4] = local_distance
    

    rendered_image, radii, out_observe, out_all_map, plane_depth = rasterizer(
            means3D = means3D_final,
            means2D = means2D,
            means2D_abs = means2D_abs,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity_final,
            scales = scales_final,
            rotations = rotations_final,
            # all_map = input_all_map,
            cov3D_precomp = cov3D_precomp)

    rendered_normal = out_all_map[0:3]
    rendered_alpha = out_all_map[3:4, ]
    rendered_distance = out_all_map[4:5, ]

    return_dict =  {
                "render": rendered_image,
                "viewspace_points": screenspace_points,
                "viewspace_points_abs": screenspace_points_abs,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "out_observe": out_observe,
                "rendered_normal": rendered_normal,
                "plane_depth": plane_depth,
                "rendered_distance": rendered_distance,
                "scales_final": scales_final
                }
    
    if return_depth_normal:
        depth_normal = render_normal(viewpoint_camera, plane_depth.squeeze()) * (rendered_alpha).detach()
        return_dict.update({"depth_normal": depth_normal})
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return return_dict
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # rendered_image, radii, depth = rasterizer(
    #     means3D = means3D_final,
    #     means2D = means2D,
    #     shs = shs,
    #     colors_precomp = colors_precomp,
    #     opacities = opacity,
    #     scales = scales_final,
    #     rotations = rotations_final,
    #     cov3D_precomp = cov3D_precomp)
    


    # # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # # They will be excluded from value updates used in the splitting criteria.
    # return {"render": rendered_image,
    #         "depth": depth,
    #         "viewspace_points": screenspace_points,
    #         "visibility_filter" : radii > 0,
    #         "radii": radii,}
