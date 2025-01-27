/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

__device__ __forceinline__
float3 transform3x3(const float3& p, const float* matrix)
{
    // matrix is assumed to be 16 floats in column-major order, i.e.:
    // [ 0,  1,  2,  3,
    //   4,  5,  6,  7,
    //   8,  9, 10, 11,
    //  12, 13, 14, 15 ]
    // where the top-left 3x3 is at indices:
    // matrix[0], matrix[4], matrix[8],
    // matrix[1], matrix[5], matrix[9],
    // matrix[2], matrix[6], matrix[10].

	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ bool in_frustum(int idx,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Bring points to screen space
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	p_view = transformPoint4x3(p_orig, viewmatrix);

	if (p_view.z <= 0.2f)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}

// Inlines a CUDA device function to invert an orthonormal view matrix.
//
// M_in is a 4x4 matrix (column-major). It must be of the form:
//   [ R^T   -R^T*C ]
//   [  0       1   ]
// where R is orthonormal (rotation), and C is the camera position in world space.
//
// M_out will be the inverse (camera->world), i.e.:
//   [ R    C ]
//   [ 0    1 ].
//
// This does NOT handle general scale/shear/perspective. 
// For a full matrix inverse, see a cofactor or Gauss-Jordan approach.
__device__ __forceinline__
void invertOrthonormalView4x4(const float* M_in, float* M_out)
{
    // For reference, in column-major:
    //
    // M_in[ 0] = R^T_{0,0}, M_in[ 4] = R^T_{0,1}, M_in[ 8] = R^T_{0,2}, M_in[12] = -R^T*C_x
    // M_in[ 1] = R^T_{1,0}, M_in[ 5] = R^T_{1,1}, M_in[ 9] = R^T_{1,2}, M_in[13] = -R^T*C_y
    // M_in[ 2] = R^T_{2,0}, M_in[ 6] = R^T_{2,1}, M_in[10] = R^T_{2,2}, M_in[14] = -R^T*C_z
    // M_in[ 3] = 0.0f,      M_in[ 7] = 0.0f,      M_in[11] = 0.0f,      M_in[15] = 1.0f
    //
    // We want M_out = M_in^{-1}, which for an orthonormal view is:
    //    [ R    C ]
    //    [ 0    1 ]
    // Where R = (R^T)^T, and C = -R * t, with t = last column of M_in.

    // 1) Extract R^T from M_in:
    float rtx0 = M_in[0],  rtx1 = M_in[4],  rtx2 = M_in[ 8];
    float rty0 = M_in[1],  rty1 = M_in[5],  rty2 = M_in[ 9];
    float rtz0 = M_in[2],  rtz1 = M_in[6],  rtz2 = M_in[10];

    // 2) Extract translation t = -R^T*C = (M_in[12], M_in[13], M_in[14])
    //    (since M_in is world->camera).
    float tx = M_in[12];
    float ty = M_in[13];
    float tz = M_in[14];

    // 3) Compute R = (R^T)^T:
    //    R_{0,0} = rtx0, R_{0,1} = rty0, R_{0,2} = rtz0, etc.
    //    We'll place it directly into M_out in column-major:
    M_out[ 0] = rtx0;  // R_{0,0}
    M_out[ 1] = rty0;  // R_{1,0}
    M_out[ 2] = rtz0;  // R_{2,0}
    M_out[ 3] = 0.0f;

    M_out[ 4] = rtx1;  // R_{0,1}
    M_out[ 5] = rty1;  // R_{1,1}
    M_out[ 6] = rtz1;  // R_{2,1}
    M_out[ 7] = 0.0f;

    M_out[ 8] = rtx2;  // R_{0,2}
    M_out[ 9] = rty2;  // R_{1,2}
    M_out[10] = rtz2;  // R_{2,2}
    M_out[11] = 0.0f;

    // 4) Compute the camera center C in world space:
    //    We have t = - R^T C  =>  C = - R * t
    //    (R is 3x3, t is a 3x1).
    float Cx = -(rtx0*tx + rtx1*ty + rtx2*tz);
    float Cy = -(rty0*tx + rty1*ty + rty2*tz);
    float Cz = -(rtz0*tx + rtz1*ty + rtz2*tz);

    // Place in last column of M_out:
    M_out[12] = Cx;  // camera center x
    M_out[13] = Cy;  // camera center y
    M_out[14] = Cz;  // camera center z
    M_out[15] = 1.0f;
}

__device__ __forceinline__
float sign(float x)
{
    if (x > 0.0f)  return 1.0f;
    if (x < 0.0f)  return -1.0f;
    return 0.0f;
}

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif