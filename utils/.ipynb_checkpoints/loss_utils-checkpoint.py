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
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp, sqrt
import cv2
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

# If you can, run this example on a GPU, it will be a lot faster.
device = "cuda" if torch.cuda.is_available() else "cpu"

model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
# credit ====> deblur 4dgs
from flow3d.pwcnet import PWCNet, get_backwarp
import torch.nn as nn
class AlignedLoss(nn.Module):
    def __init__ (self, loss_weight=1.0):
        super(AlignedLoss, self).__init__()
        self.lrec = torch.nn.L1Loss()
        self.alignnet = model
        self.alignnet.eval()
        self.loss_weight = loss_weight
    def forward(self, pred, target, mask=None):
        if target.ndim == 3:  # [C, H, W]
            target = target.unsqueeze(0)  # [1, C, H, W]
        if pred.ndim == 3:  # [C, H, W]
            target = target.unsqueeze(0)  # [1, C, H, W]
        with torch.no_grad():
            # print(pred.shape, target.shape)
            offset = self.alignnet(pred, target)  # 144 3 32 32
        align_pred, flow_mask = get_backwarp(pred, offset[-1])

        if mask is not None:
             l_rec = self.lrec(align_pred*flow_mask*mask, target*flow_mask*mask)
        else:
            l_rec = self.lrec(align_pred*flow_mask, target*flow_mask) 
        
        return l_rec 

def TV_loss(x, mask=None):
    B, C, H, W = x.shape
    tv_h = torch.abs(x[:,:,1:,:] - x[:,:,:-1,:]).sum()
    tv_w = torch.abs(x[:,:,:,1:] - x[:,:,:,:-1]).sum()
    return (tv_h + tv_w) / (B * C * H * W)

def masked_TV_loss(x, mask):
    """
    x:    (B, C, H, W)
    mask: (B, 1, H, W) or (B, C, H, W) with values in {0,1} (or bool)
    """
    # print()
    B, C, H, W = x.shape
        # If the mask is 3D: [B, H, W], insert a channel dim -> [B, 1, H, W]
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)  # => [B, 1, H, W]
    elif mask.ndim == 2:
        # If [H, W], then also add batch & channel => [1, 1, H, W]
        mask = mask.unsqueeze(0).unsqueeze(0)
    # If mask is (B,1,H,W), broadcast to match the channel dimension
    if mask.ndim == 4 and mask.shape[1] == 1 and C > 1:
        mask = mask.repeat(1, C, 1, 1)
    
    # Horizontal differences: shape [B, C, H-1, W]
    diff_h = x[:,:,1:,:] - x[:,:,:-1,:]
    # Create horizontal mask to match shape
    mask_h = mask[:,:,1:,:] * mask[:,:,:-1,:]
    
    # Vertical differences: shape [B, C, H, W-1]
    diff_w = x[:,:,:,1:] - x[:,:,:,:-1]
    # Create vertical mask
    mask_w = mask[:,:,:,1:] * mask[:,:,:,:-1]
    
    # Apply masks and sum
    tv_h = torch.abs(diff_h) * mask_h  # keep only within the mask
    tv_w = torch.abs(diff_w) * mask_w
    
    # Sum up the values
    sum_tv = tv_h.sum() + tv_w.sum()
    
    # Normalize by the count of valid pairs in mask
    # You can compute how many pairs are non-zero in the mask (float(...).sum())
    denom = (mask_h.sum() + mask_w.sum()).clamp(min=1e-6)  # prevent division by 0
    
    return sum_tv / denom



def lpips_loss(img1, img2, lpips_model):
    loss = lpips_model(img1,img2)
    return loss.mean()

def l1_loss(network_output, gt, mask=None, sharp_map=None):
    loss = torch.abs((network_output - gt))
    if mask is not None:
        if mask.ndim == 4:
            mask = mask.repeat(1, network_output.shape[1], 1, 1)
        elif mask.ndim == 3:
            mask = mask.repeat(network_output.shape[1], 1, 1)
        else:
            raise ValueError('the dimension of mask should be either 3 or 4')
    
        try:
            if sharp_map is not None:
                if sharp_map.shape != loss.shape:
                    raise ValueError('sharp_map and loss must have the same shape')
                loss *=  sharp_map**2
            loss = loss[mask!=0]
        except Exception as e:
            print(e)
            print(sharp_map.shape)
            print(loss.shape)
            print(mask.shape)
            print(loss.dtype)
            print(mask.dtype)
    return loss.mean()

# def l1_loss(network_output, gt, mask=None):
#     """Deform L1"""
#     loss = torch.abs((network_output - gt))
#     print(gt.cpu().numpy().shape)
#     if mask is not None:
#         if mask.ndim == 4:
#             mask = mask.repeat(1, network_output.shape[1], 1, 1)
#         elif mask.ndim == 3:
#             mask = mask.repeat(network_output.shape[1], 1, 1)
#         else:
#             raise ValueError('the dimension of mask should be either 3 or 4')
    
#         try:
#             loss = loss[mask!=0]
#         except:
#             print(loss.shape)
#             print(mask.shape)
#             print(loss.dtype)
#             print(mask.dtype)
#     return loss.mean()
    
def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def get_img_grad_weight(img, beta=2.0):
    _, hd, wd = img.shape 
    bottom_point = img[..., 2:hd,   1:wd-1]
    top_point    = img[..., 0:hd-2, 1:wd-1]
    right_point  = img[..., 1:hd-1, 2:wd]
    left_point   = img[..., 1:hd-1, 0:wd-2]
    grad_img_x = torch.mean(torch.abs(right_point - left_point), 0, keepdim=True)
    grad_img_y = torch.mean(torch.abs(top_point - bottom_point), 0, keepdim=True)
    grad_img = torch.cat((grad_img_x, grad_img_y), dim=0)
    grad_img, _ = torch.max(grad_img, dim=0)
    grad_img = (grad_img - grad_img.min()) / (grad_img.max() - grad_img.min())
    grad_img = torch.nn.functional.pad(grad_img[None,None], (1,1,1,1), mode='constant', value=1.0).squeeze()
    return grad_img

def fft2d(image):
    fft_image = torch.fft.fft2(image, dim=(-2, -1))
    return torch.fft.fftshift(fft_image, dim=(-2, -1))

def make_frequency_filters(H, W, cutoff_low, cutoff_high):
    # Create a meshgrid in frequency coordinates:
    # We'll shift so that (0,0) in freq coords is at image center.
    yy, xx = torch.meshgrid(
        torch.arange(H), torch.arange(W), indexing='ij'
    )
    center_y, center_x = H // 2, W // 2
    dist = torch.sqrt((yy - center_y)**2 + (xx - center_x)**2)
    max_radius = torch.tensor((H / 2)**2 + (W / 2)**2, dtype=dist.dtype, device=dist.device)
    max_radius = torch.sqrt(max_radius)  # now a 1-element tensor
    # print(max_radius)
    # Low-pass filter mask
    LP_mask = (dist <= cutoff_low*max_radius).float()
    
    # High-pass filter mask
    # everything <= cutoff_high is 1, then we subtract the LP_mask
    HP_mask_full = (dist <= cutoff_high*max_radius).float()
    HP_mask = HP_mask_full - LP_mask
    
    return LP_mask, HP_mask

def amplitude_phase(freq):
    amplitude = torch.abs(freq)
    phase = torch.angle(freq)  # in radians
    return amplitude, phase

def freq_discrepancies(pred_freq, gt_freq, LP_mask, HP_mask):
    # The masks (H,W) broadcast to (B, C, H, W) automatically.
    LP_mask, HP_mask = LP_mask.to(pred_freq.device), HP_mask.to(pred_freq.device)

    pred_LF = pred_freq * LP_mask
    gt_LF   = gt_freq   * LP_mask

    pred_HF = pred_freq * HP_mask
    gt_HF   = gt_freq   * HP_mask

    # Get amplitude + phase for each
    pred_LF_amp, pred_LF_phase = amplitude_phase(pred_LF)
    gt_LF_amp,   gt_LF_phase   = amplitude_phase(gt_LF)

    pred_HF_amp, pred_HF_phase = amplitude_phase(pred_HF)
    gt_HF_amp,   gt_HF_phase   = amplitude_phase(gt_HF)

    # Compute amplitude discrepancies (L1) for LF, HF
    d_la = torch.mean(torch.abs(gt_LF_amp - pred_LF_amp))  # low-freq amplitude
    d_ha = torch.mean(torch.abs(gt_HF_amp - pred_HF_amp))  # high-freq amplitude
    
    # Compute phase discrepancies (L1) for LF, HF
    d_lp = torch.mean(torch.abs(gt_LF_phase - pred_LF_phase))  # low-freq phase
    d_hp = torch.mean(torch.abs(gt_HF_phase - pred_HF_phase))  # high-freq phase

    return d_la, d_lp, d_ha, d_hp

def progressive_frequency_loss(
    pred_image, gt_image, mask,
    iteration, T0, T,
    D0, D,
    w_l=1.0, w_h=1.0,
):
    # pred_image: (B, C, H, W)
    # B could be 1 in your use-case, but we handle the general case.
    B, C, H, W = pred_image.shape

    # 1) Replace masked region of gt with corresponding region from pred
    #    so that only the masked regions from gt remain, everything else from pred.
    inv_mask = ~mask
    pred_image = pred_image * mask
    # gt_image = pred_image * inv_mask + gt_image * mask
    gt_image = gt_image * mask

    # 2) Convert images to frequency domain
    pred_fft = fft2d(pred_image)
    gt_fft   = fft2d(gt_image)

    # 3) Determine how much high frequency to allow at this iteration
    if iteration < T0:
        alpha = 0.0
    elif iteration > T:
        alpha = 1.0
    else:
        alpha = float(iteration - T0) / float(T - T0)
    
    # Now interpolate the cutoff in [D0, D] based on alpha
    cutoff_high = D0 + alpha * (D - D0)
    cutoff_low  = D0  

    # 4) Build the low-pass & high-pass masks
    LP_mask, HP_mask = make_frequency_filters(H, W, cutoff_low, cutoff_high)

    # 5) Compute the four frequency-domain discrepancy terms
    d_la, d_lp, d_ha, d_hp = freq_discrepancies(pred_fft, gt_fft, LP_mask, HP_mask)

    # 6) Combine them according to iteration
    if iteration <= T0:
        # Only low-freq terms
        loss = w_l * (d_la + d_lp)
    else:
        # Include high-freq terms
        loss = w_l * (d_la + d_lp) + w_h * (d_ha + d_hp)

    return loss