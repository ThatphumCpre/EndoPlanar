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
import numpy as np
from scene.flexible_deform_model import BasicPointCloud

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.flexible_deform_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from torch.utils.data import Dataset
from typing import List

class Scene:

    gaussians_sets : List[GaussianModel]
    
    def __init__(self, args : ModelParams, gaussians_sets : List[GaussianModel], frames_per_set, opt, load_iteration=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians_sets = gaussians_sets

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))
        
        if os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")) and args.extra_mark == 'endonerf':
            scene_info, endo_dataset = sceneLoadTypeCallbacks["endonerf"](args.source_path)
            print("Found poses_bounds.py and extra marks with EndoNeRf")
        elif os.path.exists(os.path.join(args.source_path, "point_cloud.obj")) or os.path.exists(os.path.join(args.source_path, "left_point_cloud.obj")):
            scene_info = sceneLoadTypeCallbacks["scared"](args.source_path, args.white_background, args.eval)
            print("Found point_cloud.obj, assuming SCARED data!")
        else:
            assert False, "Could not recognize scene type!"
                
        self.maxtime = scene_info.maxtime
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        # self.cameras_extent = args.camera_extent
        print("self.cameras_extent is ", self.cameras_extent)

        print("Loading Training Cameras")
        self.train_camera = scene_info.train_cameras 
        print("Loading Test Cameras")
        self.test_camera = scene_info.test_cameras 
        print("Loading Video Cameras")
        self.video_camera =  scene_info.video_cameras 

        if opt.bidirectional:
            # [0, 1]
            gaussians_sets[0].center_time_idx = 0
            gaussians_sets[0].min_time_idx = 0
            gaussians_sets[0].max_time_idx = len(self.video_camera)


            # gaussians_sets[0].center_time_idx = len(self.video_camera) - 1
            # gaussians_sets[0].min_time_idx = 0
            # gaussians_sets[0].max_time_idx = len(self.video_camera) - 1
        
            # # [0, -1]
            # gaussians_sets[1].center_time_idx = len(self.video_camera) - 1
            # gaussians_sets[1].min_time_idx = 0
            # gaussians_sets[1].max_time_idx = len(self.video_camera) - 1

        else:
            middle = int(len(self.video_camera)/2)
            max_sharp_score = 0
            search_range = int(0.12*len(self.video_camera)/2)
            for i, cam in enumerate(self.video_camera[middle-search_range: middle+search_range]):
                score = (cam.mask * cam.sharp_map).mean()
                if score > max_sharp_score:
                    print(i, score, max_sharp_score)
                    max_sharp_score = score
                    cen = i
            print(i)
            gaussians_sets[0].center_time_idx = middle + cen - search_range
            gaussians_sets[0].center_time_idx = 0
            gaussians_sets[0].min_time_idx = 0
            gaussians_sets[0].max_time_idx = len(self.video_camera)
        # else:
        #     # parse_sharp_frame [0, 65] [65, 130], [130, 195], [195, 260]
        #     for set_offset in range(num_set):
        #         max_sharp_score = 0
        #         st = frames_per_set*set_offset
        #         ed = frames_per_set*(set_offset+1)
        #         cen = 0
        #         for i, cam in enumerate(self.video_camera[st: ed]):
        #             score = (cam.mask * cam.sharp_map).mean()
        #             if score > max_sharp_score:
        #                 max_sharp_score = score
        #                 cen = i
        #         gaussians_sets[set_offset].center_time_idx = cen
        #         gaussians_sets[set_offset].min_time_idx = st
        #         gaussians_sets[set_offset].max_time_idx = ed


        for i, gaussians in enumerate(gaussians_sets):
    
            if self.loaded_iter:
                # model is saved as GS unique ply file format
                # brief summary of format: representing Guassians as vertices(point) at its center 
                #                          with custom attributes being all learnable paramters except 3Dmean instead of general attr such as color
                #                          (including _coef since we also have one for each guassian)
                # method .load_ply() used parse this file format
                try:
                    gaussians.load_ply(os.path.join(self.model_path,
                                                   "point_cloud",
                                                   "iteration_" + str(self.loaded_iter),
                                                    f"set_{i}_point_cloud.ply"))
                    gaussians.load_model(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                           ))
                except: 
                    gaussians.load_ply(os.path.join(self.model_path,
                                               "point_cloud",
                                               "iteration_" + str(self.loaded_iter),
                                               "point_cloud.ply"))
                    gaussians.load_model(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                           ))
                    gaussians.max_time_idx = len(self.video_camera)
            else:
                xyz, rgb, normals = endo_dataset.get_sparse_pts(gaussians.min_time_idx, gaussians.center_time_idx, gaussians.max_time_idx)
                gaussians.center_time_idx = gaussians.center_time_idx + gaussians.min_time_idx
                # xyz_max = scene_info.point_cloud.points.max(axis=0)
                # xyz_min = scene_info.point_cloud.points.min(axis=0)
                # self.gaussians._deformation.deformation_net.grid.set_aabb(xyz_max,xyz_min)
                
                normals = np.random.random((xyz.shape[0], 3))
                pcd = BasicPointCloud(points=xyz, colors=rgb, normals=normals)

                if gaussians.max_time_idx == len(self.video_camera) - 1:
                    gaussians.max_time_idx += 1
                gaussians.create_from_pcd(pcd, args.camera_extent, self.maxtime)
                print("set ", i, ": ", gaussians.min_time_idx, gaussians.center_time_idx, gaussians.max_time_idx)

    def save(self, iteration, stage):
        if stage == "coarse":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_iteration_{}".format(iteration))
        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        for i, gaussians in enumerate(self.gaussians_sets):
            gaussians.save_ply(os.path.join(point_cloud_path, f"set_{i}_point_cloud.ply"))
        # self.gaussians.save_deformation(point_cloud_path)
    
    def getTrainCameras(self, scale=1.0):
        return self.train_camera

    def getTestCameras(self, scale=1.0):
        return self.test_camera

    def getVideoCameras(self, scale=1.0):
        return self.video_camera
    
