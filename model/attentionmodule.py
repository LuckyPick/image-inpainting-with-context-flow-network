import torch
from torch import nn
import torch.nn.functional as F
from utils.tools import same_padding


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out+avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class ContextualFlow(nn.Module):
    def __init__(self, ksize=3, stride=1, rate=1, fuse_k=3, softmax_scale=10,
                 fuse=False, use_cuda=True, device_ids=None):
        super().__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse
        self.use_cuda = use_cuda
        self.device_ids = device_ids

    def forward(self, f, b, patch_group=None):
        # get shapes
        shape_f = list(f.size())   # bs*c*h*w
        shape_b = list(b.size())   # bs*c*h*w

        # extract patches from background with stride and rate
        b_pad = same_padding(b, ksizes=(self.ksize, self.ksize), strides=(
            self.stride, self.stride), rates=(self.rate, self.rate))
        # shape:[bs,c*k*k,h*w]
        b_img_patch = F.unfold(b_pad, kernel_size=self.ksize, dilation=self.rate,
                               padding=0, stride=self.stride)
        # re_shape: [bs, c, k, k, L]
        b_img_patch = b_img_patch.reshape(
            shape_b[0], shape_b[1], self.ksize, self.ksize, -1)
        b_img_patch = b_img_patch.permute(
            0, 4, 1, 2, 3)    # raw_shape: [bs, L, C, k, k]
        if type(patch_group) == torch.Tensor:
            b_img_patch_flow = b_img_patch.reshape(
                shape_b[0], -1, self.ksize, self.ksize)  # bs,LC,k,k

            temp = torch.cat([b_img_patch_flow, patch_group],
                             dim=1)  # bs,3LC,k,k
            if self.use_cuda:
                temp = nn.Conv2d(
                    temp.shape[1], b_img_patch_flow.shape[1], 1).cuda(self.device_ids)(temp)  # bs,LC,k,k
            else:
                temp = nn.Conv2d(
                    temp.shape[1], b_img_patch_flow.shape[1], 1)(temp)

            b_img_patch = temp.reshape(
                shape_b[0], -1, shape_b[1], self.ksize, self.ksize)  # bs L C k k

        b_img_patch_groups = torch.split(b_img_patch, 1, dim=0)  # bs (L C k k)
        # split foreground to groups
        f_groups = torch.split(f, 1, dim=0)  # bs (c,h,w)

        y = []
        k = self.fuse_k
        scale = self.softmax_scale    # to fit the PyTorch tensor image value range
        fuse_weight = torch.eye(k).view(1, 1, k, k)  # 1*1*k*k
        if self.use_cuda:
            fuse_weight = fuse_weight.cuda(self.device_ids)

        for xi, wi in zip(f_groups, b_img_patch_groups):
            # conv for compare
            escape_NaN = torch.FloatTensor([1e-4])
            if self.use_cuda:
                escape_NaN = escape_NaN.cuda(self.device_ids)
            wi = wi[0]  # [L, C, k, k]
            xi = same_padding(xi, [self.ksize, self.ksize], [
                              1, 1], [1, 1])  # xi: 1*c*H*W

            yi = F.conv2d(xi, wi, stride=1)   # [1, L, H, W]
            # print(yi.shape)
            # conv implementation for fuse scores to encourage large patches
            if self.fuse:
                # make all of depth to spatial resolution
                # (B=1, I=1, H=32*32, W=32*32)
                yi = yi.view(1, 1, shape_b[2] *
                             shape_b[3], shape_f[2]*shape_f[3])
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                # (B=1, C=1, H=32*32, W=32*32)
                yi = F.conv2d(yi, fuse_weight, stride=1)
                # (B=1, 32, 32, 32, 32)
                yi = yi.contiguous().view(
                    1, shape_b[2], shape_b[3], shape_f[2], shape_f[3])
                yi = yi.permute(0, 2, 1, 4, 3)
                yi = yi.contiguous().view(
                    1, 1, shape_b[2]*shape_b[3], shape_f[2]*shape_f[3])
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)
                yi = yi.contiguous().view(
                    1, shape_b[3], shape_b[2], shape_f[3], shape_f[2])
                yi = yi.permute(0, 2, 1, 4, 3).contiguous()

            yi = yi.view(1, -1, shape_f[2], shape_f[3])
            # softmax to match
            yi = F.softmax(yi*scale, dim=1)

            wi_center = wi
            yi = F.conv_transpose2d(
                yi, wi_center, stride=self.rate, padding=1,) / 9.  # (B=1, C=128, H=64, W=64)
            # print(yi.shape)
            y.append(yi)

        y = torch.cat(y, dim=0)  # back to the mini-batch
        # print(y.shape)
        y.contiguous().view(shape_f)
        # (N,LC,K,K)
        return y, b_img_patch.reshape(shape_b[0], -1, self.ksize, self.ksize)
