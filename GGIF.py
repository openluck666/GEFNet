from math import exp
import torch.nn.functional as F
import torch.nn as nn
import torch

from einops import rearrange

class L2Loss_W(nn.Module):
    def __init__(self, eps=1e-6):
        super(L2Loss_W, self).__init__()
        self.eps = eps
    def forward(self, y1, y2, weight):
        norm_factor = torch.mean(weight.detach()) + self.eps
        dis = torch.pow((y1-y2+self.eps),2)
        return torch.mean(dis*weight) / norm_factor
class L1Loss_W(nn.Module):
    def __init__(self, eps=1e-6):
        super(L1Loss_W, self).__init__()
        self.eps = eps
    def forward(self, y1, y2, weight):
        diff = y1 - y2
        adaptive_weight = 1.0 + torch.sigmoid(10 * (diff.abs() - 0.1))  # 对显著差异区域加权
        return torch.mean(torch.abs(diff) * weight * adaptive_weight)

class GGIF_Loss(nn.Module):
    def __init__(self,gamma=1.8,exposure_thresh=0.75):
        super(GGIF_Loss, self).__init__()
        self.l1 = L1Loss_W()
        self.l2 = L2Loss_W()
        self.gamma = gamma
        self.exposure_thresh = exposure_thresh
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.register_buffer('scharr_kernel_x',
                             torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=torch.float32) / 255.0)
        self.register_buffer('scharr_kernel_y',
                             torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=torch.float32) / 255.0)

        self.detail_gate = nn.Parameter(torch.tensor(0.15))
        self.exposure_adapt = nn.Parameter(torch.tensor(1.0))

    def gradient_pyramid(self, x, ksize=3):
        scharr_x = self.scharr_kernel_x.to(device=x.device)
        scharr_y = self.scharr_kernel_y.to(device=x.device)
        scharr_x = scharr_x.view(1, 1, 3, 3).repeat(x.size(1), 1, 1, 1)
        scharr_y = scharr_y.view(1, 1, 3, 3).repeat(x.size(1), 1, 1, 1)

        grad_x = F.conv2d(x, scharr_x, padding=1, groups=x.size(1))
        grad_y = F.conv2d(x, scharr_y, padding=1, groups=x.size(1))
        return (grad_x.pow(2) + grad_y.pow(2)).sqrt()

    def highlight_constraint(self, fused, y0, y1):
        highlight_mask = (fused > 0.8).float()
        grad_fused = self.scharr(fused)
        grad_ref = torch.max(self.scharr(y0), self.scharr(y1))
        return 2*torch.mean(highlight_mask * torch.abs(grad_fused - grad_ref))

    def gamma_correction(self, x):
        safe_x = torch.where(x < 1e-6, 1e-6 * torch.ones_like(x), x)
        return torch.where(x < 0, x, safe_x.pow(1 / self.gamma))
    def dynamic_exposure_mask(self, fused, y0, y1):
        over_mask = (self.window(fused, 7, 1, 3, "max") > self.exposure_thresh).float()
        under_mask = (self.window(fused, 7, 1, 3, "min") < 0.12).float()
        ref_brightness = 0.4 * (self.adaptive_pool(y0) + 0.6 * self.adaptive_pool(y1))
        exposure_diff = torch.sigmoid(10 * (fused - ref_brightness)) - 0.5
        fused_detail = self.gradient_pyramid(fused, 3)
        detail_penalty = torch.sigmoid(6 * (self.detail_gate - fused_detail))
        weights = 1.0 + 2.2 * torch.relu(exposure_diff) * over_mask + \
                  1.8 * torch.relu(-exposure_diff) * under_mask
        return (weights * (1 + detail_penalty)).clamp(1, 6)
    def luminance_consistency(self, fused, y0, y1):
        loss = 0
        for win_size in [3, 5, 7, 9, 11]:
            kernel = self.gaussian_kernel(win_size).to(fused.device)
            fused_win = F.conv2d(fused, kernel, padding=win_size // 2)
            y0_win = F.conv2d(y0, kernel, padding=win_size // 2)
            y1_win = F.conv2d(y1, kernel, padding=win_size // 2)
            loss += F.l1_loss(fused_win, 0.5 * (y0_win + y1_win))
        return loss / 5
    def gaussian_kernel(self, size, sigma=1.0):
        coords = torch.arange(size).float() - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g[None, None, :, None] * g[None, None, None, :]
    def window(self, x, windows, stride, padding, opt):
        b, c, h, w = x.shape
        kx = F.unfold(x, kernel_size=windows, stride=stride, padding=padding)  # b, k*k, n_H*n_W
        kx = rearrange(kx, "b c (h w) -> b c h w", h=h, w=w)
        if opt == "avg":
            kx = torch.mean(kx, 1, keepdim=True)
        if opt == "max":
            kx, i = torch.max(kx, 1, keepdim=True)
        if opt == "min":
            kx, i = torch.min(kx, 1, keepdim=True)
        return kx

    def Gmask(self, x, threshold):
        mask = x >= threshold
        x[mask] = 1
        mask = x < threshold
        x[mask] = 0
        return x

    def scharr(self, x):
        b, c, h, w = x.shape
        pad = nn.ReplicationPad2d(padding=(1, 1, 1, 1))
        x = pad(x)
        kx = F.unfold(x, kernel_size=3, stride=1, padding=0)  # b,n*k*k,n_H*n_W
        kx = kx.permute([0, 2, 1])
        w1 = torch.tensor([-3, 0, 3, -10, 0, 10, -3, 0, 3]).float().cuda()
        w2 = torch.tensor([-3, -10, -3, 0, 0, 0, 3, 10, 3]).float().cuda()

        y1 = torch.matmul(kx * 255.0, w1)
        y2 = torch.matmul(kx * 255.0, w2)
        y1 = y1.unsqueeze(-1).permute([0, 2, 1])
        y2 = y2.unsqueeze(-1).permute([0, 2, 1])  # b,1,n_H*n_W
        y1 = F.fold(y1, output_size=(h, w), kernel_size=1)  # b,m,n_H,n_W
        y2 = F.fold(y2, output_size=(h, w), kernel_size=1)  # b,m,n_H,n_W
        y1 = y1.clamp(-255, 255)
        y2 = y2.clamp(-255, 255)
        return (0.5 * torch.abs(y1) + 0.5 * torch.abs(y2)) / 255.0

    def softmax(self, x1, x2, a):
        x1 = torch.exp(x1 * a) / (torch.exp(x1 * a) + torch.exp(x2 * a))
        x2 = torch.exp(x2 * a) / (torch.exp(x1 * a) + torch.exp(x2 * a))
        return x1, x2


    def forward(self, y0, g0, y1, g1, y0_Cb, y0_Cr, y1_Cb, y1_Cr, fused, vis):
        g_max, i = torch.max(torch.cat([g0, g1], 1), 1)
        g_max = g_max.unsqueeze(1)
        g_fus = self.scharr(fused)
        y0_g = self.gamma_correction(y0)
        y1_g = self.gamma_correction(y1)
        fused_g = self.gamma_correction(fused)
        exposure_weights = self.dynamic_exposure_mask(fused_g, y0_g, y1_g)
        y_avg = (y0 + y1) / 2
        weight_0, weight_1 = self.softmax(torch.abs(y0_Cb - 0.5) + torch.abs(y0_Cr - 0.5),
                               torch.abs(y1_Cb - 0.5) + torch.abs(y1_Cr - 0.5), 1)
        weight_0, weight_1 = self.softmax(weight_0, weight_1, 1)
        weighted_avg = weight_0 * y0 + weight_1 * y1
        fused_Cb = weight_0 * y0_Cb + weight_1 * y1_Cb
        fused_Cr = weight_0 * y0_Cr + weight_1 * y1_Cr
        g0_max3 = self.window(g0, 3, 1, 1, "max")
        g1_max3 = self.window(g1, 3, 1, 1, "max")
        g0_max3_01m = self.Gmask(g0_max3, 0.5)
        g1_max3_01m = self.Gmask(g1_max3, 0.5)
        g_smooth = 1 - (g0_max3_01m + g1_max3_01m).clamp(0, 1)
        loss_gra_guided = self.l1(y0, fused, g0_max3_01m) + self.l1(y1, fused, g1_max3_01m)
        loss_gra_else = self.l1(weighted_avg, fused, g_smooth)
        loss_gra_max = self.l1(g_max, g_fus, 1)
        loss_lum_consist = self.luminance_consistency(fused_g, y0_g, y1_g)
        loss_color_preserve = F.l1_loss(y0_Cb + y0_Cr, fused_Cb + fused_Cr) * 0.5
        loss_highlight = self.highlight_constraint(fused, y0, y1)
        adaptive_weights = torch.sigmoid(self.exposure_adapt * exposure_weights)
        total_loss = (loss_gra_guided * 1.6 + loss_gra_else * 1.1 +
                      loss_gra_max * 1.7 + loss_lum_consist * 0.4 +
                      loss_color_preserve * 0.35 + loss_highlight * 0.7) * adaptive_weights
        return total_loss.mean()