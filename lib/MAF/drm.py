from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

# Scale Reduce Module (SRM)
class DRM(nn.Module):
    def __init__(self, in_dim, inner_channel, scale):
        super(DRM, self).__init__()
        self.in_dim = in_dim
        self.inner_channel = inner_channel
        self.scale = scale

        self.conv_low_dim = nn.Conv2d(self.in_dim, self.inner_channel, kernel_size=1, stride=1,bias=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        low_dim = self.conv_low_dim(x)
        low_dim = self.relu(low_dim)
        h_num = int(low_dim.size(2) / self.scale)
        w_num = int(low_dim.size(3) / self.scale)

        low_dim = low_dim[:, :, :int(self.scale* h_num), :int(self.scale* w_num)]
#        print(self.scale* h_num, self.scale* w_num)
#        print(low_dim.shape)

        low_dim_spliting = list(torch.chunk(low_dim, h_num, dim=2))
        for i in range(len(low_dim_spliting)):
            low_dim_spliting[i] = list(torch.chunk(low_dim_spliting[i], w_num, dim=3))

        for i in range(len(low_dim_spliting)):
            for j in range(len(low_dim_spliting[i])):
                low_dim_spliting[i][j] = low_dim_spliting[i][j].reshape(low_dim_spliting[i][j].size(0), low_dim_spliting[i][j].size(1)* self.scale* self.scale, 1, 1)

        for i in range(len(low_dim_spliting)):
            low_dim_spliting[i] = torch.cat(low_dim_spliting[i], dim=3)
        low_dim_splited = torch.cat(low_dim_spliting, dim=2)

        return low_dim_splited