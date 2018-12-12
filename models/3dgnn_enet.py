"""
underlying enet structure adapted from https://gist.github.com/ndronen/19154831c2049a69e8d53dea8cf3e744
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import median_pool
import numpy as np

ENCODER_PARAMS = [
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,
        'input_channels': 32,
        'output_channels': 64,
        'downsample': True,
        'dropout_prob': 0.01
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,
        'input_channels': 64,
        'output_channels': 64,
        'downsample': False,
        'dropout_prob': 0.01
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,
        'input_channels': 64,
        'output_channels': 64,
        'downsample': False,
        'dropout_prob': 0.01
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,
        'input_channels': 64,
        'output_channels': 64,
        'downsample': False,
        'dropout_prob': 0.01
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,
        'input_channels': 64,
        'output_channels': 64,
        'downsample': False,
        'dropout_prob': 0.01
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,
        'input_channels': 64,
        'output_channels': 128,
        'downsample': True,
        'dropout_prob': 0.1
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,
        'input_channels': 128,
        'output_channels': 128,
        'downsample': False,
        'dropout_prob': 0.1
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,  # 2
        'input_channels': 128,
        'output_channels': 128,
        'downsample': False,
        'dropout_prob': 0.1
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,
        'input_channels': 128,
        'output_channels': 128,
        'downsample': False,
        'dropout_prob': 0.1
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,  # 4
        'input_channels': 128,
        'output_channels': 128,
        'downsample': False,
        'dropout_prob': 0.1
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,
        'input_channels': 128,
        'output_channels': 128,
        'downsample': False,
        'dropout_prob': 0.1
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,  # 8
        'input_channels': 128,
        'output_channels': 128,
        'downsample': False,
        'dropout_prob': 0.1
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,
        'input_channels': 128,
        'output_channels': 128,
        'downsample': False,
        'dropout_prob': 0.1
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,  # 16
        'input_channels': 128,
        'output_channels': 128,
        'downsample': False,
        'dropout_prob': 0.1
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,
        'input_channels': 128,
        'output_channels': 128,
        'downsample': False,
        'dropout_prob': 0.1
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,
        'input_channels': 128,
        'output_channels': 128,
        'downsample': False,
        'dropout_prob': 0.1
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,  # 2
        'input_channels': 128,
        'output_channels': 128,
        'downsample': False,
        'dropout_prob': 0.1
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,
        'input_channels': 128,
        'output_channels': 128,
        'downsample': False,
        'dropout_prob': 0.1
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,  # 4
        'input_channels': 128,
        'output_channels': 128,
        'downsample': False,
        'dropout_prob': 0.1
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,
        'input_channels': 128,
        'output_channels': 128,
        'downsample': False,
        'dropout_prob': 0.1
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,  # 8
        'input_channels': 128,
        'output_channels': 128,
        'downsample': False,
        'dropout_prob': 0.1
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,
        'input_channels': 128,
        'output_channels': 128,
        'downsample': False,
        'dropout_prob': 0.1
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,  # 16
        'input_channels': 128,
        'output_channels': 128,
        'downsample': False,
        'dropout_prob': 0.1
    }
]

DECODER_PARAMS = [
    {
        'input_channels': 128,
        'output_channels': 128,
        'upsample': False,
        'pooling_module': None
    },
    {
        'input_channels': 128,
        'output_channels': 64,
        'upsample': True,
        'pooling_module': None
    },
    {
        'input_channels': 64,
        'output_channels': 64,
        'upsample': False,
        'pooling_module': None
    },
    {
        'input_channels': 64,
        'output_channels': 64,
        'upsample': False,
        'pooling_module': None
    },
    {
        'input_channels': 64,
        'output_channels': 32,
        'upsample': True,
        'pooling_module': None
    },
    {
        'input_channels': 32,
        'output_channels': 32,
        'upsample': False,
        'pooling_module': None
    }
]


class EnetInitialBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            6, 32, (3, 3),
            stride=2, padding=1, bias=True)
        # self.pool = nn.MaxPool2d(2, stride=2)
        self.batch_norm = nn.BatchNorm2d(32, eps=1e-3)
        self.actf = nn.PReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.batch_norm(output)
        return self.actf(output)


class EnetEncoderMainPath(nn.Module):
    def __init__(self, internal_scale=None, use_relu=None, asymmetric=None, dilated=None, input_channels=None,
                 output_channels=None, downsample=None, dropout_prob=None):
        super().__init__()

        internal_channels = output_channels // internal_scale
        input_stride = downsample and 2 or 1

        self.__dict__.update(locals())
        del self.self

        self.input_conv = nn.Conv2d(
            input_channels, internal_channels, input_stride,
            stride=input_stride, padding=0, bias=False)

        self.input_batch_norm = nn.BatchNorm2d(internal_channels, eps=1e-03)

        self.middle_conv = nn.Conv2d(
            internal_channels, internal_channels, 3, stride=1, bias=True,
            dilation=1 if (not dilated or dilated is None) else dilated,
            padding=1 if (not dilated or dilated is None) else dilated)

        self.middle_batch_norm = nn.BatchNorm2d(internal_channels, eps=1e-03)

        self.output_conv = nn.Conv2d(
            internal_channels, output_channels, 1,
            stride=1, padding=0, bias=False)

        self.output_batch_norm = nn.BatchNorm2d(output_channels, eps=1e-03)

        self.dropout = nn.Dropout2d(dropout_prob)

        self.input_actf = nn.PReLU()
        self.middle_actf = nn.PReLU()

    def forward(self, input):
        output = self.input_conv(input)

        output = self.input_batch_norm(output)

        output = self.input_actf(output)

        output = self.middle_conv(output)

        output = self.middle_batch_norm(output)

        output = self.middle_actf(output)

        output = self.output_conv(output)

        output = self.output_batch_norm(output)

        output = self.dropout(output)

        return output


class EnetEncoderOtherPath(nn.Module):
    def __init__(self, internal_scale=None, use_relu=None, asymmetric=None, dilated=None, input_channels=None,
                 output_channels=None, downsample=None, **kwargs):
        super().__init__()

        self.__dict__.update(locals())
        del self.self

        if downsample:
            self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)

    def forward(self, input):
        output = input

        if self.downsample:
            output, self.indices = self.pool(input)

        if self.output_channels != self.input_channels:
            new_size = [1, 1, 1, 1]
            new_size[1] = self.output_channels // self.input_channels
            output = output.repeat(*new_size)

        return output


class EnetEncoderModule(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.main = EnetEncoderMainPath(**kwargs)
        self.other = EnetEncoderOtherPath(**kwargs)
        self.actf = nn.PReLU()

    def forward(self, input):
        main = self.main(input)
        other = self.other(input)
        # print("EnetEncoderModule main size:", main.size())
        # print("EnetEncoderModule other size:", other.size())
        return self.actf(main + other)


class EnetEncoder(nn.Module):
    def __init__(self, params, nclasses):
        super().__init__()
        self.initial_block = EnetInitialBlock()

        self.layers = []
        for i, params in enumerate(params):
            layer_name = 'encoder_{:02d}'.format(i)
            layer = EnetEncoderModule(**params)
            super().__setattr__(layer_name, layer)
            self.layers.append(layer)

        self.output_conv = nn.Conv2d(
            128, nclasses, 1,
            stride=1, padding=0, bias=True)

    def forward(self, input):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        return output


class EnetDecoderMainPath(nn.Module):
    def __init__(self, input_channels=None, output_channels=None, upsample=None, pooling_module=None):
        super().__init__()

        internal_channels = output_channels // 4
        input_stride = 2 if upsample is True else 1

        self.__dict__.update(locals())
        del self.self

        self.input_conv = nn.Conv2d(
            input_channels, internal_channels, 1,
            stride=1, padding=0, bias=False)

        self.input_batch_norm = nn.BatchNorm2d(internal_channels, eps=1e-03)

        if not upsample:
            self.middle_conv = nn.Conv2d(
                internal_channels, internal_channels, 3,
                stride=1, padding=1, bias=True)
        else:
            self.middle_conv = nn.ConvTranspose2d(
                internal_channels, internal_channels, 3,
                stride=2, padding=1, output_padding=1,
                bias=True)

        self.middle_batch_norm = nn.BatchNorm2d(internal_channels, eps=1e-03)

        self.output_conv = nn.Conv2d(
            internal_channels, output_channels, 1,
            stride=1, padding=0, bias=False)

        self.output_batch_norm = nn.BatchNorm2d(output_channels, eps=1e-03)

        self.input_actf = nn.PReLU()
        self.middle_actf = nn.PReLU()

    def forward(self, input):
        output = self.input_conv(input)

        output = self.input_batch_norm(output)

        output = self.input_actf(output)

        output = self.middle_conv(output)

        output = self.middle_batch_norm(output)

        output = self.middle_actf(output)

        output = self.output_conv(output)

        output = self.output_batch_norm(output)

        return output


class EnetDecoderOtherPath(nn.Module):
    def __init__(self, input_channels=None, output_channels=None, upsample=None, pooling_module=None):
        super().__init__()

        self.__dict__.update(locals())
        del self.self

        if output_channels != input_channels or upsample:
            self.conv = nn.Conv2d(
                input_channels, output_channels, 1,
                stride=1, padding=0, bias=False)
            self.batch_norm = nn.BatchNorm2d(output_channels, eps=1e-03)
            if upsample and pooling_module:
                self.unpool = nn.MaxUnpool2d(2, stride=2, padding=0)

    def forward(self, input):
        output = input

        if self.output_channels != self.input_channels or self.upsample:
            output = self.conv(output)
            output = self.batch_norm(output)
            if self.upsample and self.pooling_module:
                output_size = list(output.size())
                output_size[2] *= 2
                output_size[3] *= 2
                output = self.unpool(
                    output, self.pooling_module.indices,
                    output_size=output_size)

        return output


class EnetDecoderModule(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.main = EnetDecoderMainPath(**kwargs)
        self.other = EnetDecoderOtherPath(**kwargs)
        self.actf = nn.PReLU()

    def forward(self, input):
        main = self.main(input)
        other = self.other(input)
        # print("EnetDecoderModule main size:", main.size())
        # print("EnetDecoderModule other size:", other.size())
        return self.actf(main + other)


class EnetDecoder(nn.Module):
    def __init__(self, params, nclasses, encoder):
        super().__init__()

        self.encoder = encoder

        self.pooling_modules = []

        for mod in self.encoder.modules():
            try:
                if mod.other.downsample:
                    self.pooling_modules.append(mod.other)
            except AttributeError:
                pass

        self.layers = []
        for i, params in enumerate(params):
            if params['upsample']:
                params['pooling_module'] = self.pooling_modules.pop(-1)
            layer = EnetDecoderModule(**params)
            self.layers.append(layer)
            layer_name = 'decoder{:02d}'.format(i)
            super().__setattr__(layer_name, layer)

        self.output_conv = nn.ConvTranspose2d(
            32, nclasses, 2,
            stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output


class EnetGnn(nn.Module):
    def __init__(self, mlp_num_layers):
        super().__init__()

        self.median_pool = median_pool.MedianPool2d(kernel_size=8, stride=8, padding=0, same=False)

        self.g_rnn_layers = nn.ModuleList([nn.Linear(128, 128) for l in range(mlp_num_layers)])
        self.g_rnn_actfs = nn.ModuleList([nn.PReLU() for l in range(mlp_num_layers)])
        self.q_rnn_layer = nn.Linear(256, 128)
        self.q_rnn_actf = nn.PReLU()

        self.output_conv = nn.Conv2d(256, 128, 3, stride=1, padding=1, bias=True)

    # adapted from https://discuss.pytorch.org/t/build-your-own-loss-function-in-pytorch/235/6
    # (x - y)^2 = x^2 - 2*x*y + y^2
    def get_knn_indices(self, batch_mat, k):
        r = torch.bmm(batch_mat, batch_mat.permute(0, 2, 1))
        N = r.size()[0]
        HW = r.size()[1]
        batch_indices = torch.zeros((N, HW, k)).cuda()
        for idx, val in enumerate(r):
            # get the diagonal elements
            diag = val.diag().unsqueeze(0)
            diag = diag.expand_as(val)
            # compute the distance matrix
            D = (diag + diag.t() - 2 * val).sqrt()
            topk, indices = torch.topk(D, k=k, largest=False)
            batch_indices[idx] = indices
        return batch_indices

    def forward(self, cnn_encoder_output, original_input, gnn_iterations, k, xy, use_half_precision):

        # extract for convenience
        N = cnn_encoder_output.size()[0]
        C = cnn_encoder_output.size()[1]
        H = cnn_encoder_output.size()[2]
        W = cnn_encoder_output.size()[3]
        K = k

        # extract and resize depth image as horizontal disparity channel from HHA encoded image
        depth = original_input[:, 3, :, :]  # N 8H 8W
        depth = depth.view(depth.size()[0], 1, depth.size()[1], depth.size()[2])  # N 1 8H 8W
        depth_resize = self.median_pool(depth)  # N 1 H W
        x_coords = xy[:, 0, :, :]
        x_coords = x_coords.view(x_coords.size()[0], 1, x_coords.size()[1], x_coords.size()[2])
        y_coords = xy[:, 1, :, :]
        y_coords = y_coords.view(y_coords.size()[0], 1, y_coords.size()[1], y_coords.size()[2])
        x_coords = self.median_pool(x_coords)  # N 1 H W
        y_coords = self.median_pool(y_coords)  # N 1 H W

        # 3D projection
        proj_3d = torch.cat((x_coords, y_coords, depth_resize), 1)  # N 3 W H
        proj_3d = proj_3d.view(N, 3, H * W).permute(0, 2, 1).contiguous()  # N HW 3

        # get k nearest neighbors
        knn = self.get_knn_indices(proj_3d, k=K)  # N HW K
        knn = knn.view(N, H * W * K).long()  # N HWK

        # prepare CNN encoded features for RNN
        h = cnn_encoder_output  # N C H W
        h = h.permute(0, 2, 3, 1).contiguous()  # N H W C
        h = h.view(N, (H * W), C)  # N HW C

        # aggregate and iterate messages in m, keep original CNN features h for later
        m = h.clone()  # N HW C

        # loop over timestamps to unroll
        for i in range(gnn_iterations):
            # do this for every sample in batch, not nice, but I don't know how to use index_select batchwise
            for n in range(N):
                # fetch features from nearest neighbors
                neighbor_features = torch.index_select(h[n], 0, knn[n]).view(H * W, K, C)  # HW K C
                # run neighbor features through MLP g and activation function
                for idx, g_layer in enumerate(self.g_rnn_layers):
                    neighbor_features = self.g_rnn_actfs[idx](g_layer(neighbor_features))  # HW K C
                # average over activated neighbors
                m[n] = torch.mean(neighbor_features, dim=1)  # HW C

            # concatenate current state with messages
            concat = torch.cat((h, m), 2)  # N HW 2C

            # get new features by running MLP q and activation function
            h = self.q_rnn_actf(self.q_rnn_layer(concat))  # N HW C

        # format RNN activations back to image, concatenate original CNN embedding, return
        h = h.view(N, H, W, C).permute(0, 3, 1, 2).contiguous()  # N C H W
        output = self.output_conv(torch.cat((cnn_encoder_output, h), 1))  # N 2C H W
        return output


class Model(nn.Module):
    def __init__(self, nclasses, mlp_num_layers):
        super().__init__()

        self.encoder = EnetEncoder(ENCODER_PARAMS, nclasses)
        self.gnn = EnetGnn(mlp_num_layers)
        self.decoder = EnetDecoder(DECODER_PARAMS, nclasses, self.encoder)

    def forward(self, input, gnn_iterations, k, xy, use_gnn, only_encode=False, use_half_precision=False):
        x = self.encoder.forward(input)

        if only_encode:
            return x

        if use_gnn:
            x = self.gnn.forward(x, input, gnn_iterations, k, xy, use_half_precision)

        return self.decoder.forward(x)
