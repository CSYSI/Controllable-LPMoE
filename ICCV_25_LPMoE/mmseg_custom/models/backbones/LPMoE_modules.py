import logging
from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from ops.modules import MSDeformAttn
from timm.models.layers import DropPath

import torch.nn.functional as F
import pywt
import pywt.data

_logger = logging.getLogger(__name__)

def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x

def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)

class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x
def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points

def deform_inputs(x):
    bs, c, h, w = x.shape
    spatial_shapes = torch.as_tensor([(h // 8, w // 8),
                                      (h // 16, w // 16),
                                      (h // 32, w // 32)],
                                     dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 16, w // 16)], x.device)
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]
    
    spatial_shapes = torch.as_tensor([(h // 16, w // 16)], dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 8, w // 8),
                                             (h // 16, w // 16),
                                             (h // 32, w // 32)], x.device)
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]
    
    return deform_inputs1, deform_inputs2

class MFMoE(nn.Module): # Channel-oriented adaptive scale enhancement
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConvWithMoE(hidden_features,n_experts=2)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x= self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CA(nn.Module):
    def __init__(self):
        super(CA, self).__init__()
        self.ap = nn.AdaptiveAvgPool2d(1)
        self.mp = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.ap(x)+self.mp(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class RA(nn.Module):
    def __init__(self):
        super(RA, self).__init__()
        self.ap = nn.AdaptiveAvgPool2d(1)
        self.mp = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.ap(x)+self.mp(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = -1 * self.sigmoid(y) + 1
        return x * y.expand_as(x)

class MoEAttention(nn.Module):
    def __init__(self, in_features, n_experts=2):
        super(MoEAttention, self).__init__()
        self.ca = CA()
        self.ra = RA()


        self.Gating = GatingNetwork(in_features=in_features,experts=n_experts)
    def forward(self, x):
        batch_size, channels, height, width = x.size()

        x_flattened = x.view(batch_size, channels, -1)
        gating_weights = self.Gating(x_flattened.mean(dim=-1))

        ca_output = self.ca(x)
        ra_output = self.ra(x)

        gating_weights = gating_weights.view(batch_size, -1, 1, 1)

        ca_weighted = gating_weights[:, 0:1, :, :] * ca_output
        ra_weighted = gating_weights[:, 1:, :, :] * ra_output

        output = ca_weighted + ra_weighted

        return output

class DWConvWithMoE(nn.Module):
    def __init__(self, dim=768, n_experts=2):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.moe_attention = MoEAttention(in_features=dim, n_experts=n_experts)

    def forward(self, x, H, W):
        B, N, C = x.shape
        n = N // 21

        x1 = x[:, 0:16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        x2 = x[:, 16 * n:20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()
        x3 = x[:, 20 * n:, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()

        x1 = self.dwconv(x1)
        x2 = self.dwconv(x2)
        x3 = self.dwconv(x3)

        x1 = self.moe_attention(x1)
        x2 = self.moe_attention(x2)
        x3 = self.moe_attention(x3)

        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x3 = x3.flatten(2).transpose(1, 2)

        x = torch.cat([x1, x2, x3], dim=1)

        return x



class Module2_1(nn.Module): # First Interaction
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index):
        
        def _inner_forward(query, feat):

            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)  # Cosine-aligned deformable attention
            return query + self.gamma * attn
        
        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)
            
        return query


class Module2_2(nn.Module):  # Second Interaction
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = MFMoE(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):

        def _inner_forward(query, feat):

            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None) # Cosine-aligned deformable attention
            query = query + attn

            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W)) # Channel-oriented adaptive scale enhancement
            return query

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query



class Module2(nn.Module): # Bi-directional Interaction Adapter
    def __init__(self, dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,
                 deform_ratio=1.0, extra_extractor=False, with_cp=False):
        super().__init__()

        self.module2_1 = Module2_1(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
                                 n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                                 with_cp=with_cp) # first interaction
        self.module2_2 = Module2_2(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                   norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp) # second inteaction
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Module2_2(dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
                          with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                          drop=drop, drop_path=drop_path, with_cp=with_cp)
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, H, W):
        x = self.module2_1(query=x, reference_points=deform_inputs1[0],
                          feat=c, spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])
        for idx, blk in enumerate(blocks):
            x = blk(x, H, W)
        c = self.module2_2(query=c, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                              feat=x, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2], H=H, W=W)
        return x, c


class Module2_cls(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,
                 deform_ratio=1.0, extra_extractor=False, with_cp=False):
        super().__init__()

        self.module2_1 = Module2_1(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
                                 n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                                 with_cp=with_cp)
        self.module2_2 = Module2_2(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                   norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Module2_2(dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
                          with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                          drop=drop, drop_path=drop_path, with_cp=with_cp)
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(self, x, c, cls, blocks, deform_inputs1, deform_inputs2, H, W):
        x = self.module2_1(query=x, reference_points=deform_inputs1[0],
                          feat=c, spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])
        x = torch.cat((cls, x), dim=1)
        for idx, blk in enumerate(blocks):
            x = blk(x, H, W)
        cls, x = x[:, :1, ], x[:, 1:, ]
        c = self.module2_2(query=c, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                              feat=x, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2], H=H, W=W)
        return x, c, cls


class ConvExpert_ATConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvExpert_ATConv, self).__init__()

        self.Conv_1 = nn.Sequential(*[
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=1, bias=True),
            nn.SyncBatchNorm(out_channels // 4),
            nn.ReLU(inplace=True),
        ])

        self.ATConv_3 = nn.Sequential(*[
            nn.Conv2d(in_channels//4, out_channels//4, kernel_size=3, stride=1, dilation=3, padding=3, bias=True),
            nn.SyncBatchNorm(out_channels//4),
            nn.ReLU(inplace=True),
        ])
        self.ATConv_5 = nn.Sequential(*[
            nn.Conv2d(in_channels//4, out_channels//4, kernel_size=3, stride=1, dilation=5, padding=5, bias=True),
            nn.SyncBatchNorm(out_channels//4),
            nn.ReLU(inplace=True),
        ])
        self.ATConv_7 = nn.Sequential(*[
            nn.Conv2d(in_channels//4, out_channels//4, kernel_size=3, stride=1, dilation=7, padding=7, bias=True),
            nn.SyncBatchNorm(out_channels//4),
            nn.ReLU(inplace=True),
        ])

        self.reduce = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=True),
            nn.SyncBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):
        x  = self.Conv_1(x)
        x1 = self.ATConv_3(x)
        x2 = self.ATConv_5(x + x1)
        x3 = self.ATConv_7(x + x2)
        x = self.reduce(torch.cat([x1, x2, x3, x], 1))
        return  x


class ConvExpert_DConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvExpert_DConv, self).__init__()

        self.Conv_1 = nn.Sequential(*[
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1, stride=1, bias=True),
            nn.SyncBatchNorm(out_channels//4),
            nn.ReLU(inplace=True),
        ])
        self.DConv_3 = nn.Sequential(*[
            nn.Conv2d(in_channels//4, out_channels//4, kernel_size=3, stride=1, padding=1, groups=in_channels//4, bias=True),
            nn.SyncBatchNorm(out_channels//4),
            nn.ReLU(inplace=True),
        ])
        self.DConv_5 = nn.Sequential(*[
            nn.Conv2d(in_channels//4, out_channels//4, kernel_size=5, stride=1, padding=2, groups=out_channels//4, bias=True),
            nn.SyncBatchNorm(out_channels//4),
            nn.ReLU(inplace=True),
        ])
        self.DConv_7 = nn.Sequential(*[
            nn.Conv2d(in_channels//4, out_channels//4, kernel_size=7, stride=1, padding=3, groups=out_channels//4, bias=True),
            nn.SyncBatchNorm(out_channels//4),
            nn.ReLU(inplace=True),
        ])

        self.reduce = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=True),
            nn.SyncBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.Conv_1(x)
        x1 = self.DConv_3(x)
        x2 = self.DConv_5(x + x1)
        x3 = self.DConv_7(x + x2)
        x  = self.reduce(torch.cat([x1,x2,x3,x],1))
        return x


class ConvExpert_ASConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvExpert_ASConv, self).__init__()

        self.Conv_1 = nn.Sequential(*[
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1, stride=1, bias=True),
            nn.SyncBatchNorm(out_channels//4),
            nn.ReLU(inplace=True),
        ])

        self.ASConv_1_3 = nn.Sequential(*[
            nn.Conv2d(in_channels//4, out_channels//4, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True),
            nn.SyncBatchNorm(out_channels//4),
            nn.ReLU(inplace=True),
        ])
        self.ASConv_3_1 = nn.Sequential(*[
            nn.Conv2d(out_channels//4, out_channels//4, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=True),
            nn.SyncBatchNorm(out_channels//4),
            nn.ReLU(inplace=True),
        ])
        self.ASConv_1_5 = nn.Sequential(*[
            nn.Conv2d(in_channels//4, out_channels//4, kernel_size=(1, 5), stride=1, padding=(0, 2), bias=True),
            nn.SyncBatchNorm(out_channels//4),
            nn.ReLU(inplace=True),
        ])
        self.ASConv_5_1 = nn.Sequential(*[
            nn.Conv2d(out_channels//4, out_channels//4, kernel_size=(5, 1), stride=1, padding=(2, 0), bias=True),
            nn.SyncBatchNorm(out_channels//4),
            nn.ReLU(inplace=True),
        ])
        self.ASConv_1_7 = nn.Sequential(*[
            nn.Conv2d(in_channels//4, out_channels//4, kernel_size=(1, 7), stride=1, padding=(0, 3), bias=True),
            nn.SyncBatchNorm(out_channels//4),
            nn.ReLU(inplace=True),
        ])
        self.ASConv_7_1 = nn.Sequential(*[
            nn.Conv2d(out_channels//4, out_channels//4, kernel_size=(7, 1), stride=1, padding=(3, 0), bias=True),
            nn.SyncBatchNorm(out_channels//4),
            nn.ReLU(inplace=True),
        ])

        self.reduce = nn.Sequential(
            nn.Conv2d(out_channels , out_channels, kernel_size=1, stride=1, bias=True),
            nn.SyncBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):

        x      = self.Conv_1(x)

        x1_1   = self.ASConv_1_3(x)
        x1     = self.ASConv_3_1(x1_1)

        x2_1   = self.ASConv_1_5(x+x1)
        x2     = self.ASConv_5_1(x2_1)

        x3_1   = self.ASConv_1_7(x+x2)
        x3     = self.ASConv_7_1(x3_1)

        x = self.reduce(torch.cat([x1, x2, x3, x], 1))
        return x

class ConvExpert_WTConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvExpert_WTConv, self).__init__()

        self.Conv_1 = nn.Sequential(*[
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1, stride=1, bias=True),
            nn.SyncBatchNorm(out_channels//4),
            nn.ReLU(inplace=True),
        ])

        self.WTConv_3 = nn.Sequential(*[
            WTConv2d(in_channels//4, out_channels//4, kernel_size=3, bias=True),
            nn.SyncBatchNorm(out_channels//4),
            nn.ReLU(inplace=True),
        ])
        self.WTConv_5 = nn.Sequential(*[
            WTConv2d(in_channels//4, out_channels//4, kernel_size=5, bias=True),
            nn.SyncBatchNorm(out_channels//4),
            nn.ReLU(inplace=True),
        ])
        self.WTConv_7 = nn.Sequential(*[
            WTConv2d(in_channels//4, out_channels//4, kernel_size=7, bias=True),
            nn.SyncBatchNorm(out_channels//4),
            nn.ReLU(inplace=True),
        ])

        self.reduce = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=True),
            nn.SyncBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.Conv_1(x)
        x1 = self.WTConv_3(x)
        x2 = self.WTConv_5(x + x1)
        x3 = self.WTConv_7(x + x2)
        x  = self.reduce(torch.cat([x1, x2, x3, x],1))
        return x



class GatingNetwork(nn.Module):
    def __init__(self, in_features,experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(in_features, experts)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        return self.softmax(x)



class MoEModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MoEModel, self).__init__()

        self.expert_a = ConvExpert_DConv(in_channels, out_channels)
        self.expert_b = ConvExpert_ATConv(in_channels, out_channels)
        self.expert_c = ConvExpert_ASConv(in_channels, out_channels)
        self.expert_d = ConvExpert_WTConv(in_channels, out_channels)


        self.gating_network = GatingNetwork(in_features=in_channels, experts=4)

    def forward(self, x):
        batch_size, channels, _, _ = x.size()


        x_flattened = x.view(batch_size, channels, -1)
        gate_weights = self.gating_network(x_flattened.mean(dim=-1))


        output_a = self.expert_a(x)
        output_b = self.expert_b(x)
        output_c = self.expert_c(x)
        output_d = self.expert_d(x)




        output = (gate_weights[:, 0].view(-1, 1, 1, 1) * output_a +
                  gate_weights[:, 1].view(-1, 1, 1, 1) * output_b +
                  gate_weights[:, 2].view(-1, 1, 1, 1) * output_c +
                  gate_weights[:, 3].view(-1, 1, 1, 1) * output_d)

        return output


class Module1(nn.Module): # Dynamic Mixed Local Priors Extractor
    def __init__(self, inplanes=64, embed_dim=384, with_cp=False):
        super().__init__()
        self.with_cp = with_cp

        self.stem = nn.Sequential(*[
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1,bias=True),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.conv2 = nn.Sequential(*[
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, padding=1, bias=True),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
        ])

        self.conv3 = nn.Sequential(*[
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, padding=1, bias=True),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv4 = nn.Sequential(*[
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, padding=1, bias=True),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True)
        ])

        self.MoEConv = MoEModel(inplanes,inplanes)

        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)


    def forward(self, x):
        
        def _inner_forward(x):
            c1 = self.stem(x)
            c1 = c1 + self.MoEConv(c1)
            c2 = self.conv2(c1)
            c2 = c2 + self.MoEConv(c2)
            c3 = self.conv3(c2)
            c3 = c3 + self.MoEConv(c3)
            c4 = self.conv4(c3)
            c4 = c4 + self.MoEConv(c4)

            c1 = self.fc1(c1)
            c2 = self.fc1(c2)
            c3 = self.fc1(c3)
            c4 = self.fc1(c4)
    
            bs, dim, _, _ = c1.shape

            c2 = c2.view(bs, dim, -1).transpose(1, 2)
            c3 = c3.view(bs, dim, -1).transpose(1, 2)
            c4 = c4.view(bs, dim, -1).transpose(1, 2)
    
            return c1, c2, c3, c4
        
        if self.with_cp and x.requires_grad:
            outs = cp.checkpoint(_inner_forward, x)
        else:
            outs = _inner_forward(x)
        return outs