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

import os
from enum import Enum
from typing import Any, Optional

import numpy as np
import torch
import trimesh
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from arguments import ModelParams, OptimizationParams, PipelineParams
from scene.appearance_network import AppearanceNetwork
from utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    inverse_sigmoid,
    strip_symmetric,
)
from utils.graphics_utils import BasicPointCloud
from utils.sh_utils import RGB2SH
from utils.system_utils import mkdir_p


class GaussianModel:
    class App_model(Enum):
        NO = 0
        GS = 1
        GOF = 2
        PGSR = 3

    # use mip-splatting filters
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.sharpness_activateion = torch.nn.functional.softplus

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int, sg_degree: int):
        self.active_sh_degree = 0
        self.active_sg_degree = 0
        self.max_sh_degree = sh_degree
        self.max_sg_degree = sg_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._sg_axis = torch.empty(0)
        self._sg_sharpness = torch.empty(0)
        self._sg_color = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.percent_dense = 0.0
        self.spatial_lr_scale = 0.0
        self.setup_functions()
        self.appearance_network: Optional[AppearanceNetwork] = None
        self._appearance_embeddings: Optional[torch.nn.Parameter] = None

    def capture(self):
        if self.app_model == self.App_model.GOF:
            app_model_param = self.appearance_network.state_dict()
        else:
            app_model_param = None
        return (
            self.active_sh_degree,
            self.active_sg_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._sg_axis,
            self._sg_sharpness,
            self._sg_color,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.app_model,
            app_model_param,
            self._appearance_embeddings,
        )

    def restore(self, model_args, training_args):
        (
            self.active_sh_degree,
            self.active_sg_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._sg_axis,
            self._sg_sharpness,
            self._sg_color,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            self.app_model,
            app_dict,
            _appearance_embeddings,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        if self.app_model == self.App_model.GOF:
            self.appearance_network = AppearanceNetwork(3 + 64, 3).cuda()
            self.appearance_network.load_state_dict(app_dict)
        self._appearance_embeddings = _appearance_embeddings

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_scaling_with_3D_filter(self):
        scales = self.get_scaling
        scales = torch.square(scales) + torch.square(self.filter_3D)
        scales = torch.sqrt(scales)
        return scales

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_sg_sharpness(self):
        return self.sharpness_activateion(self._sg_sharpness)

    @property
    def get_sg_axis(self):
        axis = torch.nn.functional.normalize(self._sg_axis, dim=2)
        return axis

    @property
    def get_sg_color(self):
        return self._sg_color

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def _filtered_scaling_and_coef(self):
        scales = self.get_scaling
        filtered_scales = torch.hypot(scales, self.filter_3D)
        coef = (scales / filtered_scales).prod(dim=1, keepdim=True)
        return filtered_scales, coef

    @property
    def get_opacity_with_3D_filter(self):
        _, coef = self._filtered_scaling_and_coef()
        return self.get_opacity * coef

    @property
    def get_scaling_n_opacity_with_3D_filter(self):
        scales, coef = self._filtered_scaling_and_coef()
        return scales, self.get_opacity * coef

    def get_appearance_embedding(self, idx):
        return self._appearance_embeddings[idx]

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    @torch.no_grad()
    def reset_3D_filter(self):
        xyz = self.get_xyz
        self.filter_3D = torch.zeros([xyz.shape[0], 1], device=xyz.device)

    @torch.no_grad()
    def compute_3D_filter(self, cameras):
        # print("Computing 3D filter")
        # TODO consider focal length and image width
        xyz = self.get_xyz
        distance = torch.full((xyz.shape[0],), torch.inf, device=xyz.device)
        valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)

        # we should use the focal length of the highest resolution camera
        focal_length = 0.0
        for camera in cameras:
            # transform points to camera space
            R = camera.R
            T = camera.T
            # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here
            xyz_cam = torch.addmm(T[None, :], xyz, R)
            z = xyz_cam[:, 2]

            # project to screen space
            valid_depth = z > 0.2

            uv = xyz_cam[:, :2] / z.unsqueeze(-1)
            uv_abs = torch.abs(uv)

            boundry_x = camera.image_width / camera.Fx * 0.575
            boundry_y = camera.image_height / camera.Fy * 0.575
            in_screen = torch.logical_and(uv_abs[:, 0] <= boundry_x, uv_abs[:, 1] <= boundry_y)

            valid = torch.logical_and(valid_depth, in_screen)

            distance = torch.where(valid, torch.minimum(distance, z), distance)
            valid_points = torch.logical_or(valid_points, valid)
            focal_length = max(focal_length, camera.Fx)

        distance[~valid_points] = distance[valid_points].max()

        filter_3D = distance / focal_length * (0.2**0.5)
        self.filter_3D = filter_3D[..., None]

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def unlockSGdegree(self, N):
        self.active_sg_degree = min(self.active_sg_degree + N, self.max_sg_degree)

    def create_app_model(self, num_cameras: int, app_model: App_model) -> None:
        self.app_model = self.App_model(app_model)

        init_dispatch = {
            self.App_model.GS: self._init_gs_embeddings,
            self.App_model.GOF: self._init_gof_embeddings,
            self.App_model.PGSR: self._init_pgsr_embeddings,
            self.App_model.NO: self._init_no_embeddings,
        }

        try:
            init_dispatch[self.app_model](num_cameras)
        except KeyError as exc:
            raise ValueError(f"Unsupported appearance model: {self.app_model}") from exc

    def _init_gs_embeddings(self, num_cameras: int) -> None:
        exposure = torch.eye(3, 4, device="cuda").expand(num_cameras, -1, -1).clone()
        self._appearance_embeddings = nn.Parameter(exposure.requires_grad_(True))

    def _init_gof_embeddings(self, num_cameras: int) -> None:
        self.appearance_network = AppearanceNetwork(3 + 64, 3).cuda()
        std = 1e-4
        embeddings = torch.empty(num_cameras, 64, device="cuda").normal_(0, std)
        self._appearance_embeddings = nn.Parameter(embeddings)

    def _init_pgsr_embeddings(self, num_cameras: int) -> None:
        embeddings = torch.zeros(num_cameras, 2, device="cuda")
        self._appearance_embeddings = nn.Parameter(embeddings)

    def _init_no_embeddings(self, num_cameras: int) -> None:
        self._appearance_embeddings = None
        self.appearance_network = None

    def create_from_pcd(self, pcd, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        if isinstance(pcd, BasicPointCloud):
            fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        else:
            fused_point_cloud = torch.tensor(np.asarray(pcd._xyz)).float().cuda()
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd._rgb)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        sg_axis = torch.randn(fused_point_cloud.shape[0], self.max_sg_degree, 3).float().cuda()
        sg_axis = torch.nn.functional.normalize(sg_axis, dim=2)
        sg_sharpness = torch.zeros(fused_point_cloud.shape[0], self.max_sg_degree).float().cuda()
        sg_color = torch.zeros(fused_point_cloud.shape[0], self.max_sg_degree, 3).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud.detach().clone().float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._sg_axis = nn.Parameter(sg_axis.requires_grad_(True))
        self._sg_sharpness = nn.Parameter(sg_sharpness.requires_grad_(True))
        self._sg_color = nn.Parameter(sg_color.requires_grad_(True))

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args: OptimizationParams) -> None:
        self._init_gradient_accumulators()
        base_param_groups = self._build_base_param_groups(training_args)
        appearance_groups = self._build_appearance_param_groups(training_args)

        self.optimizer = torch.optim.Adam(
            base_param_groups + appearance_groups,
            lr=0.0,
            eps=1e-15,
        )

    # --- helpers -------------------------------------------------------------

    def _init_gradient_accumulators(self) -> None:
        num_points = self.get_xyz.shape[0]
        device = self.get_xyz.device

        self.xyz_gradient_accum = torch.zeros((num_points, 1), device=device)
        self.xyz_gradient_accum_abs = torch.zeros((num_points, 1), device=device)
        self.denom = torch.zeros((num_points, 1), device=device)

    def _build_base_param_groups(self, training_args: OptimizationParams) -> list[dict[str, Any]]:
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )
        spatial_lr = training_args.position_lr_init * self.spatial_lr_scale
        return [
            {"params": [self._xyz], "lr": spatial_lr, "name": "xyz"},
            {"params": [self._features_dc], "lr": training_args.feature_dc_lr, "name": "f_dc"},
            {"params": [self._features_rest], "lr": training_args.feature_rest_lr, "name": "f_rest"},
            {"params": [self._opacity], "lr": training_args.opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": training_args.scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": training_args.rotation_lr, "name": "rotation"},
            {"params": [self._sg_axis], "lr": training_args.sg_axis_lr, "name": "sg_axis"},
            {"params": [self._sg_sharpness], "lr": training_args.sg_sharpness_lr, "name": "sg_sharpness"},
            {"params": [self._sg_color], "lr": training_args.sg_color, "name": "sg_color"},
        ]

    def _build_appearance_param_groups(self, training_args: OptimizationParams) -> list[dict[str, Any]]:
        dispatch = {
            self.App_model.GS: self._gs_param_groups,
            self.App_model.GOF: self._gof_param_groups,
            self.App_model.PGSR: self._pgsr_param_groups,
            self.App_model.NO: self._noop_param_groups,
        }
        try:
            builder = dispatch[self.app_model]
        except KeyError as exc:
            raise ValueError(f"Unsupported appearance model: {self.app_model}") from exc
        return builder(training_args)

    def _gs_param_groups(self, training_args: OptimizationParams) -> list[dict[str, Any]]:
        self.exposure_scheduler_args = get_expon_lr_func(
            training_args.gs_appearance_lr_init,
            training_args.gs_appearance_lr_final,
            lr_delay_steps=training_args.gs_appearance_lr_delay_steps,
            lr_delay_mult=training_args.gs_appearance_lr_delay_mult,
            max_steps=training_args.iterations,
        )
        return [
            {
                "params": [self._appearance_embeddings],
                "lr": training_args.gs_appearance_lr_init,
                "name": "appearance_embeddings",
            }
        ]

    def _gof_param_groups(self, training_args: OptimizationParams) -> list[dict[str, Any]]:
        if self.appearance_network is None:
            raise RuntimeError("appearance_network must be initialized before calling training_setup")
        return [
            {
                "params": [self._appearance_embeddings],
                "lr": training_args.appearance_embeddings_lr,
                "name": "appearance_embeddings",
            },
            {
                "params": self.appearance_network.parameters(),
                "lr": training_args.appearance_network_lr,
                "name": "appearance_network",
            },
        ]

    def _pgsr_param_groups(self, training_args: OptimizationParams) -> list[dict[str, Any]]:
        return [
            {
                "params": [self._appearance_embeddings],
                "lr": training_args.pgsr_appearance_lr,
                "beta": (0.9, 0.99),
                "name": "appearance_embeddings",
            }
        ]

    def _noop_param_groups(self, training_args: OptimizationParams) -> list[dict[str, Any]]:
        return []

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "appearance_embeddings" and self.app_model == self.App_model.GS:
                param_group["lr"] = self.exposure_scheduler_args(iteration)

    def construct_list_of_attributes(self, exclude_filter=False):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        for i in range(self._sg_axis.shape[1] * self._sg_axis.shape[2]):
            l.append("sg_axis_{}".format(i))
        for i in range(self._sg_sharpness.shape[1]):
            l.append("sg_sharpness_{}".format(i))
        for i in range(self._sg_color.shape[1] * self._sg_color.shape[2]):
            l.append("sg_color_{}".format(i))
        if not exclude_filter:
            l.append("filter_3D")
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        sg_axis = self._sg_axis.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        sg_sharpness = self._sg_sharpness.detach().cpu().numpy()
        sg_color = self._sg_color.detach().flatten(start_dim=1).contiguous().cpu().numpy()

        filter_3D = self.filter_3D.detach().cpu().numpy()
        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, sg_axis, sg_sharpness, sg_color, filter_3D), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    @torch.no_grad()
    def get_tetra_points(self):
        NEAR_SCALE_FACTOR = 1.5
        FAR_SCALE_FACTOR = 3
        M = trimesh.creation.box()
        M.vertices *= 2

        rots = build_rotation(self._rotation)
        xyz = self.get_xyz
        scale = self.get_scaling_with_3D_filter

        vertices_near = M.vertices.T * NEAR_SCALE_FACTOR
        vertices_far = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]).T * FAR_SCALE_FACTOR
        vertices = np.concatenate([vertices_near, vertices_far], axis=1)
        vertices = torch.from_numpy(vertices).float().cuda().unsqueeze(0).repeat(xyz.shape[0], 1, 1)
        # scale vertices first
        vertices = vertices * scale.unsqueeze(-1)
        vertices = torch.bmm(rots, vertices).squeeze(-1) + xyz.unsqueeze(-1)
        vertices = vertices.permute(0, 2, 1).reshape(-1, 3).contiguous()
        # concat center points
        vertices = torch.cat([vertices, xyz], dim=0)
        scale = scale.max(dim=-1, keepdim=True)[0] * 3.0
        scale_corner = scale.repeat(1, 8 + 6).reshape(-1, 1)
        vertices_scale = torch.cat([scale_corner, scale], dim=0)
        return vertices, vertices_scale

    def reset_opacity(self):
        # reset opacity to by considering 3D filter
        current_opacity_with_filter = self.get_opacity_with_3D_filter
        opacities_new = torch.min(current_opacity_with_filter, torch.ones_like(current_opacity_with_filter) * 0.01)

        scales = self.get_scaling
        filtered_scales = torch.hypot(scales, self.filter_3D)
        rcoef = (filtered_scales / scales).prod(dim=1, keepdim=True)
        opacities_new = opacities_new * rcoef
        opacities_new = self.inverse_opacity_activation(torch.clamp(opacities_new, 0, 0.9999))

        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]), np.asarray(plydata.elements[0]["y"]), np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        filter_3D = np.asarray(plydata.elements[0]["filter_3D"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        sg_axis_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("sg_axis_")]
        sg_axis_names = sorted(sg_axis_names, key=lambda x: int(x.split("_")[-1]))
        sg_axis = np.zeros((xyz.shape[0], self.max_sg_degree * 3))
        for idx, attr_name in enumerate(sg_axis_names):
            sg_axis[:, idx] = np.asarray(plydata.elements[0][attr_name])
        sg_axis = sg_axis.reshape((sg_axis.shape[0], self.max_sg_degree, 3))

        sg_sharpness_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("sg_sharpness_")]
        sg_sharpness_names = sorted(sg_sharpness_names, key=lambda x: int(x.split("_")[-1]))
        sg_sharpness = np.zeros((xyz.shape[0], self.max_sg_degree))
        for idx, attr_name in enumerate(sg_sharpness_names):
            sg_sharpness[:, idx] = np.asarray(plydata.elements[0][attr_name])

        sg_color_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("sg_color_")]
        sg_color_names = sorted(sg_color_names, key=lambda x: int(x.split("_")[-1]))
        sg_color = np.zeros((xyz.shape[0], self.max_sg_degree * 3))
        for idx, attr_name in enumerate(sg_color_names):
            sg_color[:, idx] = np.asarray(plydata.elements[0][attr_name])
        sg_color = sg_color.reshape((sg_axis.shape[0], self.max_sg_degree, 3))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._sg_axis = nn.Parameter(torch.tensor(sg_axis, dtype=torch.float, device="cuda").requires_grad_(True))
        self._sg_sharpness = nn.Parameter(torch.tensor(sg_sharpness, dtype=torch.float, device="cuda").requires_grad_(True))
        self._sg_color = nn.Parameter(torch.tensor(sg_color, dtype=torch.float, device="cuda").requires_grad_(True))
        self.filter_3D = torch.tensor(filter_3D, dtype=torch.float, device="cuda")

        self.active_sh_degree = self.max_sh_degree
        self.active_sg_degree = self.max_sg_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["appearance_embeddings", "appearance_network"]:
                continue
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["appearance_embeddings", "appearance_network"]:
                continue
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._sg_axis = optimizable_tensors["sg_axis"]
        self._sg_sharpness = optimizable_tensors["sg_sharpness"]
        self._sg_color = optimizable_tensors["sg_color"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def prune_points_inference(self, mask):
        valid_points_mask = ~mask

        self._xyz = self._xyz[valid_points_mask]
        self._features_dc = self._features_dc[valid_points_mask]
        self._features_rest = self._features_rest[valid_points_mask]
        self._opacity = self._opacity[valid_points_mask]
        self._scaling = self._scaling[valid_points_mask]
        self._rotation = self._rotation[valid_points_mask]
        self._sg_axis = self._sg_axis[valid_points_mask]
        self._sg_sharpness = self._sg_sharpness[valid_points_mask]
        self._sg_color = self._sg_color[valid_points_mask]
        self.filter_3D = self.filter_3D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["appearance_embeddings", "appearance_network"]:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_sg_axis, new_sg_sharpness, new_sg_color
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "sg_axis": new_sg_axis,
            "sg_sharpness": new_sg_sharpness,
            "sg_color": new_sg_color,
        }
        # extension_num = new_xyz.shape[0]
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._sg_axis = optimizable_tensors["sg_axis"]
        self._sg_sharpness = optimizable_tensors["sg_sharpness"]
        self._sg_color = optimizable_tensors["sg_color"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, grads_abs, grad_abs_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        padded_grad_abs = torch.zeros((n_init_points), device="cuda")
        padded_grad_abs[: grads_abs.shape[0]] = grads_abs.squeeze()
        selected_pts_mask_abs = torch.where(padded_grad_abs >= grad_abs_threshold, True, False)
        # selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacities = self._opacity[selected_pts_mask].repeat(N, 1)
        new_sg_axis = self._sg_axis[selected_pts_mask].repeat(N, 1, 1)
        new_sg_sharpness = self._sg_sharpness[selected_pts_mask].repeat(N, 1)
        new_sg_color = self._sg_color[selected_pts_mask].repeat(N, 1, 1)
        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_sg_axis, new_sg_sharpness, new_sg_color
        )

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, grads_abs, grad_abs_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        # sample a new gaussian instead of fixing position
        stds = self.get_scaling[selected_pts_mask]
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask])
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask]

        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_sg_axis = self._sg_axis[selected_pts_mask]
        new_sg_sharpness = self._sg_sharpness[selected_pts_mask]
        new_sg_color = self._sg_color[selected_pts_mask]

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_sg_axis, new_sg_sharpness, new_sg_color
        )

    # use the same densification strategy as GOF https://github.com/autonomousvision/gaussian-opacity-fields
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0
        ratio = (torch.norm(grads, dim=-1) >= max_grad).float().mean()
        Q = torch.quantile(grads_abs.reshape(-1), 1 - ratio)

        before = self._xyz.shape[0]
        self.densify_and_clone(grads, max_grad, grads_abs, Q, extent)
        clone = self._xyz.shape[0]

        self.densify_and_split(grads, max_grad, grads_abs, Q, extent)
        split = self._xyz.shape[0]

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        self.prune_points(prune_mask)
        prune = self._xyz.shape[0]
        return clone - before, split - clone, split - prune

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, 2:], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
