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
from typing import List, Tuple

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

def render(viewpoint_camera, pcs, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, 
           return_plane =True, return_depth_normal = True):

    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # seem like they initial 2Dmeans with zeros and set it to retain grad, 
    # send the its pointer to the CUDA rasterization kernel, 
    # and the CUDA rasterization kernel update/store the projection in this 2Dmeans so that we can inspect the gradient in python env if needed

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
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pcs[0].active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            render_geo=return_plane,
            debug=pipe.debug
        )

    rasterizer = PlaneGaussianRasterizer(raster_settings=raster_settings)

    means3D_finals = []
    rotations_finals = []
    scales_finals = []
    opacity_finals = []
    means2Ds = []
    means2D_abss = []
    shss = []
    # means3D = pc.get_xyz
    # add deformation to each points

    
    means3D = torch.cat([pcs[0].get_xyz, pcs[1].get_xyz], dim=0)
    ori_time_first = torch.tensor(pcs[0].get_normalized_time_with_offset(viewpoint_camera.time)).to(means3D.device)
    ori_time_seconds = torch.tensor(pcs[1].get_normalized_time_with_offset(viewpoint_camera.time)).to(means3D.device)
    opacity = torch.cat([pcs[0]._opacity, pcs[1]._opacity], dim=0)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None


    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = torch.cat([pcs[0]._scaling, pcs[1]._scaling], dim=0)
        rotations = torch.cat([pcs[0]._rotation, pcs[1]._rotation], dim=0)

    deformation_point =  torch.cat([pcs[0]._deformation_table, pcs[1]._deformation_table], dim=0)

    means3D_deform, scales_deform, rotations_deform = efficient_deformation(xyz=means3D[deformation_point],
                                                                            scales=scales[deformation_point], 
                                                                            rotations=rotations[deformation_point],
                                                                            time=[ori_time_first,ori_time_seconds],
                                                                            ch_num=pcs[0].args.ch_num,
                                                                            basis_num=pcs[0].args.curve_num,
                                                                            set1=pcs[0],
                                                                            set2=pcs[1]
                                                                            )
    # opacity_final = pc.get_deform_opacity(ori_time)
    opacity_deform = torch.cat([pcs[0]._opacity, pcs[1]._opacity], dim=0)
        
    # print(time.max())
    with torch.no_grad():
        
        # pc._deformation_accum[deformation_point] += torch.abs(means3D_deform - means3D[deformation_point])
        pcs[0]._deformation_accum[pcs[0]._deformation_table] += torch.abs(means3D_deform[:len(pcs[0].get_xyz)]
                                                                            - means3D[:len(pcs[0].get_xyz)][pcs[0]._deformation_table])
        pcs[1]._deformation_accum[pcs[1]._deformation_table] += torch.abs(means3D_deform[len(pcs[0].get_xyz):]
                                                                            - means3D[len(pcs[0].get_xyz):][pcs[1]._deformation_table])

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

    scales_final = pcs[0].scaling_activation(scales_final)
    # print("\n\n\n\n", scales_final)
    rotations_final = pcs[0].rotation_activation(rotations_final)
    opacity_final = pcs[0].opacity_activation(opacity)

    
    for pc in pcs:
    # pc = pcs[0]
      
        # deformation = pc.get_deformation
        
    
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
        shss.append(shs)
    

  
    # screenspace_points = torch.cat(means2Ds, 0)
    # screenspace_points_abs = torch.cat(means2D_abss, 0)
    shs = torch.cat(shss, 0)

 
    
    # Now, 'shs' (and/or 'colors_precomp') contains the processed data from all pcs.

    
    # print(means3D_final.shape)
    screenspace_points = torch.zeros_like(means3D_final, dtype=means3D_final.dtype, requires_grad=True, device="cuda") + 0 # [N, ]
    screenspace_points_abs = torch.zeros_like(means3D_final, dtype=means3D_final.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
        screenspace_points_abs.retain_grad()
    except:
        pass
    # deformation = pc.get_deformation
    means2D = screenspace_points
    means2D_abs = screenspace_points_abs
    # if not return_plane:
    #     rendered_image, radii, out_observe, out_all_map, plane_depth = rasterizer(
    #         means3D = means3D_final,
    #         means2D = means2D,
    #         means2D_abs = means2D_abs,
    #         shs = shs,
    #         colors_precomp = colors_precomp,
    #         opacities = opacity_final,
    #         scales = scales_final,
    #         rotations = rotations_final,
    #         cov3D_precomp = cov3D_precomp)
        
    #     return_dict =  {"render": rendered_image,
    #                     "viewspace_points": screenspace_points,
    #                     "viewspace_points_abs": screenspace_points_abs,
    #                     "visibility_filter" : radii > 0,
    #                     "radii": radii,
    #                     "out_observe": out_observe}

    #     return return_dict

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
                "scales_final": scales_final,
                "opacity_final": opacity_final,
                }
    
    if return_depth_normal:
        depth_normal = render_normal(viewpoint_camera, plane_depth.squeeze()) * (rendered_alpha).detach()
        return_dict.update({"depth_normal": depth_normal})
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return return_dict


def efficient_deformation(
                xyz: torch.Tensor,
                scales: torch.Tensor,
                rotations: torch.Tensor,
                time: List,
                ch_num,
                basis_num,
                set1,
                set2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply flexible deformation modeling to the Gaussian model. Only the pistions, scales, and rotations are
        considered deformable in this work.

        Args:
            xyz (torch.Tensor): The current positions of the model vertices. (shape: [N, 3])
            scales (torch.Tensor): The current scales per Gaussian primitive. (shape: [N, 3])
            rotations (torch.Tensor): The current rotations of the model. (shape: [N, 4])
            time (float): The current time.

        Returns:
            tuple: A tuple containing the updated positions, scaling factors, and rotations of the model.
                   (xyz: torch.Tensor, scales: torch.Tensor, rotations: torch.Tensor)
        """
        #deform = self.gaussian_deformation(time, ch_num=self.args.ch_num, basis_num=self.args.curve_num)
        deform = efficient_gaussian_deformation(N=len(xyz), coef_first_set=set1._coefs, coef_second_set=set2._coefs, t1=time[0], t2=time[1], ch_num=ch_num, basis_num=basis_num)

        deform_xyz = deform[:, :3]
       
        xyz += deform_xyz
        deform_rot = deform[:, 3:7]
        rotations += deform_rot
        try:
            # when ch_num is 10
            deform_scaling = deform[:, 7:10]
            scales += deform_scaling
            return xyz, scales, rotations
        except:
            return xyz, scales, rotations
        
def efficient_gaussian_deformation(N, coef_first_set, coef_second_set, t1: float, t2: float, ch_num=10, basis_num=17):
    """
    Applies linear combination of learnable Gaussian basis functions to model the surface deformation.
    Args:
        t1 (float): First input value.
        t2 (float): Second input value.
        ch_num (int): Number of channels in the deformation tensor.
        basis_num (int): Number of Gaussian basis functions.
    Returns:
        torch.Tensor: The deformed model tensor for both t1 and t2.
    """
    # Stack both coefficient sets into one tensor for parallel computation
    coefs = torch.cat([coef_first_set, coef_second_set], dim=0)  # Shape: (X+Y, ch_num, 3, basis_num)

    # Reshape coefficients
    coefs = coefs.reshape(N, ch_num, 3, basis_num).contiguous()
    weight, mu, sigma = torch.chunk(coefs, 3, dim=2)  # Split along dimension 2 (the 3 in shape)
    
    # Create tensor t with the same length as corresponding coefficient sets
    t = torch.cat([
        torch.full((coef_first_set.shape[0], ch_num, 1, basis_num), t1, device=coefs.device),
        torch.full((coef_second_set.shape[0], ch_num, 1, basis_num), t2, device=coefs.device)
    ], dim=0)

    
    # Compute Gaussian weights
    exponent = (t - mu) ** 2 / (sigma ** 2 + 1e-4)
    gaussian = torch.exp(-exponent)  # Removed extra square in exponent
    
    # Apply weight and sum over basis functions
    return (gaussian * weight).sum(dim=-1).squeeze()


