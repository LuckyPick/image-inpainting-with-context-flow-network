import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as spectral_norm_fn
from torch.nn.utils import weight_norm as weight_norm_fn
from .attentionmodule import ContextualFlow
from . import attentionmodule


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.input_dim = config.input_dim
        self.cnum = config.cnum
        self.use_cuda = config.use_cuda

        self.coarse_generator = CoarseGenerator(
            self.input_dim, self.cnum, self.use_cuda, )
        self.fine_generator = FineGenerator(
            self.input_dim, self.cnum, self.use_cuda, )

    def forward(self, x, mask):
        x_stage1 = self.coarse_generator(x, mask)
        x_stage2 = self.fine_generator(x, x_stage1, mask)
        return x_stage1, x_stage2


class Conv2dBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding=0,
                 conv_padding=0, dilation=1, weight_norm='none', norm='none',
                 activation='relu', pad_type='zero',
                 gating=False, transpose=False, use_bias=False):
        super().__init__()
        self.gating = gating
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'none':
            self.pad = None
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = c_out
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        if weight_norm == 'sn':
            self.weight_norm = spectral_norm_fn
        elif weight_norm == 'wn':
            self.weight_norm = weight_norm_fn
        elif weight_norm == 'none':
            self.weight_norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(weight_norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2,)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if gating:
            self.conv = nn.Conv2d(c_in, 2 * c_out,
                                  kernel_size, stride,
                                  padding=conv_padding,
                                  dilation=dilation,
                                  bias=use_bias
                                  )
            self.activation2 = nn.Sigmoid()

        else:
            if transpose:
                self.conv = nn.ConvTranspose2d(c_in, c_out,
                                               kernel_size, stride,
                                               padding=conv_padding,
                                               output_padding=conv_padding,
                                               dilation=dilation,
                                               bias=use_bias)
            else:
                self.conv = nn.Conv2d(c_in, c_out, kernel_size, stride,
                                      padding=conv_padding, dilation=dilation,
                                      bias=use_bias)

        if self.weight_norm:
            self.conv = self.weight_norm(self.conv)

    def forward(self, x):
        if self.gating:
            if self.pad:
                x = self.pad(x)
            x = self.conv(x)
            x, y = torch.chunk(x, 2, dim=1)
            y = self.activation2(y)
            if self.activation:
                x = self.activation(x) * y
            else:
                x = x * y
            if self.norm:
                x = self.norm(x)
        else:
            if self.pad:
                x = self.pad(x)
            x = self.conv(x)
            if self.activation:
                x = self.activation(x)
            if self.norm:
                x = self.norm(x)
        return x


class AttentionResBlock(nn.Module):
    def __init__(self, channel, c_reduction=16, s_ksize=7):
        super().__init__()
        self.ca = attentionmodule.ChannelAttention(
            channel=channel, reduction=c_reduction)
        self.sa = attentionmodule.SpatialAttention(
            kernel_size=s_ksize
        )

    def forward(self, x):
        x1 = x * self.ca(x)
        x2 = x1 * self.sa(x1)
        return x+x2


def gen_conv(c_in, c_out, kernel_size=3, stride=1, padding=0,
             rate=1, activation='relu'):
    return Conv2dBlock(c_in, c_out, kernel_size, stride, conv_padding=padding,
                       dilation=rate, activation=activation, use_bias=True)


def dis_conv(c_in, c_out, kernel_size=5, stride=2, padding=0, rate=1,
             activation='lrelu'):
    return Conv2dBlock(c_in, c_out, kernel_size, stride, conv_padding=padding,
                       dilation=rate, activation=activation, use_bias=True)


def gen_deconv(c_in, c_out, kernel_size=3, stride=2, padding=1, rate=1,
               activation='relu'):
    return Conv2dBlock(c_in, c_out, kernel_size, stride, conv_padding=padding,
                       dilation=rate, activation=activation, transpose=True, use_bias=True)


def gen_gatedconv(c_in, c_out, kernel_size=3, stride=1, padding=0,
                  rate=1, activation='relu'):
    return Conv2dBlock(c_in, c_out, kernel_size, stride, conv_padding=padding,
                       dilation=rate, activation=activation, gating=True, use_bias=True)


class CoarseGenerator(nn.Module):
    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None):
        super().__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.conv1 = gen_gatedconv(input_dim+2, cnum, 5, 1, 2)
        self.conv2_downsample = gen_gatedconv(cnum, cnum*2, 3, 2, 1)
        self.conv3 = gen_gatedconv(cnum*2, cnum*4, 3, 1, 1)
        self.conv4_downsample = gen_gatedconv(cnum*4, cnum*4, 3, 2, 1)
        self.conv5 = gen_gatedconv(cnum*4, cnum*4, 3, 1, 1)
        self.conv6 = gen_gatedconv(cnum*4, cnum*4, 3, 1, 1)

        self.conv7_atrous = gen_gatedconv(cnum*4, cnum*4, 3, 1, 2, rate=2)
        self.conv8_atrous = gen_gatedconv(cnum*4, cnum*4, 3, 1, 4, rate=4)
        self.conv9_atrous = gen_gatedconv(cnum*4, cnum*4, 3, 1, 8, rate=8)
        self.conv10_atrous = gen_gatedconv(cnum*4, cnum*4, 3, 1, 16, rate=16)

        self.conv11 = gen_gatedconv(cnum*4, cnum*4, 3, 1, 1)
        self.conv12 = gen_gatedconv(cnum*4, cnum*4, 3, 1, 1)

        self.conv13 = gen_gatedconv(cnum*4*2, cnum*2, 3, 1, 1)
        self.conv14 = gen_gatedconv(cnum*2, cnum*2, 3, 1, 1)
        self.conv15 = gen_gatedconv(cnum*2, cnum, 3, 1, 1)
        self.conv16 = gen_gatedconv(cnum, cnum//2, 3, 1, 1)
        self.conv17 = gen_conv(cnum//2, input_dim, 3, 1, 1, activation='none')

    def forward(self, x, mask):
        x = x * (1-mask) + mask
        ones = torch.ones(x.size(0), 1, x.size(2), x.size(3))
        if self.use_cuda:
            ones = ones.cuda()
            mask = mask.cuda()

        # 5 x 256 x 256
        x = self.conv1(torch.cat([x, ones, mask], dim=1))
        x = self.conv2_downsample(x)
        # cnum*2 x 128 x 128
        x_skip1 = self.conv3(x)
        x = self.conv4_downsample(x_skip1)
        # cnum*4 x 64 x 64
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x = self.conv11(x)
        x = self.conv12(x)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x_skip1, x], dim=1)
        # cnum*2 x 128 x 128
        x = self.conv13(x)
        x = self.conv14(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        # cnum x 256 x 256
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        # 3 x 256 x 256
        # x_stage1 = torch.clamp(x, -1., 1.)
        x_stage1 = torch.tanh(x)

        return x_stage1


class FineGenerator(nn.Module):
    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None):
        super().__init__()
        self.use_cuda = use_cuda
        self.devices_ids = device_ids

        # 3 x 256 x 256
        self.conv1 = gen_conv(input_dim + 1, cnum, 7, 1, 3)
        
        self.conv18 = gen_conv(cnum, cnum, 5, 1, 2)
        self.conv2_downsample = gen_conv(cnum, cnum, 3, 2, 1)
        # cnum*2 x 128 x 128
        self.conv3 = gen_conv(cnum, cnum*2, 3, 1, 1)
        self.conv4_downsample = gen_conv(cnum*2, cnum*2, 3, 2, 1)
        self.down_c1 = gen_conv(cnum*2, cnum, 1, 1)
        # cnum*4 x 64 x 64 cat
        self.conv5 = gen_conv(cnum*(2+1), cnum*4, 3, 1, 1)
        self.down_c2 = gen_conv(cnum*4, cnum, 1, 1)
        self.conv6 = gen_conv(cnum*(4+1), cnum*4, 3, 1, 1)
        self.down_c3 = gen_conv(cnum*4, cnum, 1, 1)

        self.conv7_atrous = gen_conv(cnum*(4+1), cnum*4, 3, 1, 2, rate=2)
        self.conv8_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 4, rate=4)
        self.conv9_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 8, rate=8)
        self.conv10_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 16, rate=16)
        self.down_c4 = gen_conv(cnum*4, cnum, 1, 1)

        # attention flow branch
        self.cf1 = ContextualFlow(
            use_cuda=True, stride=8, device_ids=self.devices_ids)
        self.cf2 = ContextualFlow(
            use_cuda=True, stride=8, device_ids=self.devices_ids)
        self.cf3 = ContextualFlow(
            use_cuda=True, stride=8, device_ids=self.devices_ids)
        self.cf4 = ContextualFlow(
            use_cuda=True, stride=8, device_ids=self.devices_ids)
        self.cf5 = ContextualFlow(
            use_cuda=True, stride=8, device_ids=self.devices_ids)

        self.conv11 = gen_conv(cnum*(4+1), cnum*4, 3, 1, 1)
        self.down_c5 = gen_conv(cnum*4, cnum, 1, 1,)
        self.conv12 = gen_conv(cnum*(4+1), cnum*4, 3, 1, 1)
        self.conv13 = gen_conv(cnum*4, cnum*2, 3, 1, 1)
        self.conv14 = gen_conv(cnum*2, cnum*2, 3, 1, 1)
        self.conv15 = gen_conv(cnum*2, cnum, 3, 1, 1)
        self.conv16 = gen_conv(cnum, cnum//2, 3, 1, 1)
        self.conv17 = gen_conv(
            cnum//2, input_dim, 3, 1, 1, activation='none')

    def forward(self, xin, x_stage1, mask):
        x1_inpaint = x_stage1 * mask + xin * (1. - mask)
        # x1_inpaint = x_stage1top
        # ones = torch.ones(xin.size(0), 1, xin.size(2), xin.size(3))
        if self.use_cuda:
            # ones = ones.cuda(self.devices_ids)
            mask = mask.cuda(self.devices_ids)

        x = torch.cat([x1_inpaint, mask], dim=1)
        x = self.conv1(x)
        x = self.conv18(x)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x_cat1 = self.conv4_downsample(x)
        x = self.down_c1(x_cat1)
        x, flow = self.cf1(x, x)
        x = torch.cat([x_cat1, x], dim=1)
        x_cat2 = self.conv5(x)
        x = self.down_c2(x_cat2)
        x, flow = self.cf2(x, x, flow)
        x = torch.cat([x_cat2, x], dim=1)
        x_cat3 = self.conv6(x)
        x = self.down_c3(x_cat3)
        x, flow = self.cf3(x, x, flow)
        x = torch.cat([x_cat3, x], dim=1)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x_cat4 = self.conv10_atrous(x)
        x = self.down_c4(x_cat4)
        x, flow = self.cf4(x, x, flow)
        x = torch.cat([x_cat4, x], dim=1)
        x_cat5 = self.conv11(x)
        x = self.down_c5(x_cat5)
        x, _ = self.cf5(x, x, flow)
        x = torch.cat([x_cat5, x], dim=1)
        x = self.conv12(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv13(x)
        x = self.conv14(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        # x_stage2 = torch.clamp(x, -1., 1.)
        x_stage2 = torch.tanh(x)

        return x_stage2


class Discriminator(nn.Module):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4,
                          stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4,
                          stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4,
                          stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4,
                          stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4,
                          stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        # if init_weights:
        #     self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module
