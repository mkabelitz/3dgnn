"""
adapted from https://github.com/Eromera/erfnet_pytorch/blob/master/train/erfnet.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import median_pool


class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
                                   dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True,
                                   dilation=(1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.relu(output + input)  # +input = identity (residual connection)


class ErfnetEncoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.initial_block = DownsamplerBlock(6, 16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16, 64))

        for x in range(0, 5):  # 5 times
            self.layers.append(non_bottleneck_1d(64, 0.03, 1))

        self.layers.append(DownsamplerBlock(64, 128))

        for x in range(0, 2):  # 2 times
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))

        # Only in encoder mode:
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        if predict:
            output = self.output_conv(output)

        return output


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class ErfnetDecoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output


class ErfnetGnn(nn.Module):
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

        # preallocate for neighbor pooling
        m = torch.zeros_like(h)  # N HW C

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


# ERFNet
class Model(nn.Module):
    def __init__(self, num_classes, mlp_num_layers, encoder=None, gnn=None):  # use encoder to pass pretrained encoder
        super().__init__()

        if (encoder == None):
            self.encoder = ErfnetEncoder(num_classes)
        else:
            self.encoder = encoder

        if (gnn == None):
            self.gnn = ErfnetGnn(mlp_num_layers)
        else:
            self.gnn = gnn

        self.decoder = ErfnetDecoder(num_classes)

    def forward(self, input, gnn_iterations, k, xy, use_gnn, only_encode=False, use_half_precision=False):

        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            x = self.encoder(input)  # predict=False by default

            if use_gnn:
                x = self.gnn.forward(x, input, gnn_iterations, k, xy, use_half_precision)

            return self.decoder.forward(x)
