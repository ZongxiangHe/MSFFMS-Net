import math
import torch
import torch.nn as nn
from torch.nn import init

BATCHNORM_TRACK_RUNNING_STATS = False
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997


class BNorm_init(nn.BatchNorm2d):
    def reset_parameters(self):
        init.uniform_(self.weight, 0, 1)
        init.zeros_(self.bias)


class Conv2d_init(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros"):
        super(Conv2d_init, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

    def reset_parameters(self):
        init.xavier_normal_(self.weight)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        return self.conv(x)


def _conv_block(in_chanels, out_chanels, kernel_size, padding):
    return nn.Sequential(Conv2d_init(in_channels=in_chanels, out_channels=out_chanels,
                                     kernel_size=kernel_size, padding=padding, bias=False),
                         FeatureNorm(num_features=out_chanels, eps=0.001),
                         nn.ReLU())


class FeatureNorm(nn.Module):
    def __init__(self, num_features, feature_index=1, rank=4, reduce_dims=(2, 3), eps=0.001, include_bias=True):
        super(FeatureNorm, self).__init__()
        self.shape = [1] * rank
        self.shape[feature_index] = num_features
        self.reduce_dims = reduce_dims

        self.scale = nn.Parameter(torch.ones(self.shape, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.zeros(self.shape, requires_grad=True, dtype=torch.float)) if include_bias else nn.Parameter(
            torch.zeros(self.shape, requires_grad=False, dtype=torch.float))

        self.eps = eps

    def forward(self, features):
        f_std = torch.std(features, dim=self.reduce_dims, keepdim=True)
        f_mean = torch.mean(features, dim=self.reduce_dims, keepdim=True)
        return self.scale * ((features - f_mean) / (f_std + self.eps).sqrt()) + self.bias


class SpatialAttention(nn.Module):
    # 初始化，卷积核大小为7*7
    def __init__(self, kernel_size=7):
        # 继承父类初始化方法
        super(SpatialAttention, self).__init__()

        # 为了保持卷积前后的特征图shape相同，卷积时需要padding
        padding = kernel_size // 2
        # 7*7卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        self.conv = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        # sigmoid函数
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs1, inputs2):
        # 在通道维度上最大池化 [b,1,h,w]  keepdim保留原有深度
        # 返回值是在某维度的最大值和对应的索引
        x_maxpool1, _ = torch.max(inputs1, dim=1, keepdim=True)
        x_maxpool2, _ = torch.max(inputs2, dim=1, keepdim=True)

        # 在通道维度上平均池化 [b,1,h,w]
        x_avgpool1 = torch.mean(inputs1, dim=1, keepdim=True)
        x_avgpool2 = torch.mean(inputs2, dim=1, keepdim=True)
        # 池化后的结果在通道维度上堆叠 [b,2,h,w]
        x = torch.cat([x_maxpool1, x_maxpool2, x_avgpool1, x_avgpool2], dim=1)

        # 卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        x = self.conv(x)
        # 空间权重归一化
        x = self.sigmoid(x)
        # 输入特征图和空间权重相乘
        outputs = inputs2 * x
        return outputs


class SegDecNet(nn.Module):
    def __init__(self, device, input_width, input_height, input_channels):
        super(SegDecNet, self).__init__()
        if input_width % 8 != 0 or input_height % 8 != 0:
            raise Exception(f"Input size must be divisible by 8! width={input_width}, height={input_height}")
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels

        self.conv1 = _conv_block(self.input_channels,32, 5, 2)#512
        self.maxp1 = nn.MaxPool2d(2)                          #256
        self.conv2 = _conv_block(32, 64, 5, 2)
        self.conv3 = _conv_block(64, 64, 5, 2)
        self.conv4 = _conv_block(64, 64, 5, 2)
        self.maxp2 = nn.MaxPool2d(2)                          #128
        self.conv5 = _conv_block(64, 64, 5, 2)
        self.conv6 = _conv_block(64, 64, 5, 2)                                 #32+64+64+64+1024+1024
        self.conv7 = _conv_block(64, 64, 5, 2)
        self.conv8 = _conv_block(64, 64, 5, 2)
        self.maxp3 = nn.MaxPool2d(2)                          #64
        self.conv9 = _conv_block(64, 1024, 15, 7)
        self.conv9a = _conv_block(1024, 1024, 5, 2)
        self.upsa3 = nn.Upsample(scale_factor=2,
                                 mode='bilinear')
        #self.upsa3 = nn.functional.interpolate(self.conv9a, scale_factor=2, mode='nearest')  #128
        self.conv6a = _conv_block(1024, 64, 5, 2)
        self.upsa2 = nn.Upsample(scale_factor=2,
                                 mode='bilinear')
        #self.upsa2 = nn.functional.interpolate(self.conv6a, scale_factor=2, mode='nearest') #256
        self.conv3a = _conv_block(64, 32, 5, 2)
        self.upsa1 = nn.Upsample(scale_factor=2,
                                 mode='bilinear')
        #self.upsa1 = nn.functional.interpolate(self.conv3a, scale_factor=2, mode='nearest')  #512
        self.conv1a = _conv_block(32, 16, 5, 2)

        self.conv_concat1 = _conv_block(65, 65, 5, 2)
        self.conv_concat1s = _conv_block(65, 32, 5, 2)
        self.conv_concat2 = _conv_block(65, 65, 5, 2)
        self.conv_concat3 = _conv_block(130, 130, 5, 2)
        self.conv_concat3s = _conv_block(130, 64, 5, 2)
        self.conv_concat4 = _conv_block(1025, 1025, 5, 2)
        self.conv_concat5 = _conv_block(1155, 1024, 5, 2)
        self.conv_concat5s = _conv_block(1024, 128, 5, 2)
        self.seg_mask1 = nn.Sequential(
            Conv2d_init(in_channels=1024, out_channels=1, kernel_size=1, padding=0, bias=False),
            FeatureNorm(num_features=1, eps=0.001, include_bias=False))
        self.seg_mask2 = nn.Sequential(
            Conv2d_init(in_channels=64, out_channels=1, kernel_size=1, padding=0, bias=False),
            FeatureNorm(num_features=1, eps=0.001, include_bias=False))
        self.seg_mask3 = nn.Sequential(
            Conv2d_init(in_channels=32, out_channels=1, kernel_size=1, padding=0, bias=False),
            FeatureNorm(num_features=1, eps=0.001, include_bias=False))

        self.extractor = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                       _conv_block(in_chanels=1024, out_chanels=8, kernel_size=5, padding=2),
                                       nn.MaxPool2d(kernel_size=2),
                                       _conv_block(in_chanels=8, out_chanels=16, kernel_size=5, padding=2),
                                       nn.MaxPool2d(kernel_size=2),
                                       _conv_block(in_chanels=16, out_chanels=32, kernel_size=5, padding=2))
        # self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # self.convs1 = _conv_block(in_chanels=1024, out_chanels=8, kernel_size=5, padding=2)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # self.convs2 = _conv_block(in_chanels=40, out_chanels=32, kernel_size=5, padding=2)
        # self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        # self.convs3 = _conv_block(in_chanels=96, out_chanels=48, kernel_size=5, padding=2)
        # self.convs = _conv_block(in_chanels=176, out_chanels=128, kernel_size=5, padding=2)
        # self.convss = _conv_block(in_chanels=128, out_chanels=64, kernel_size=5, padding=2)
        # self.convsss = _conv_block(in_chanels=64, out_chanels=32, kernel_size=5, padding=2)
        self.global_max_pool_feat = nn.MaxPool2d(kernel_size=32)
        self.global_avg_pool_feat = nn.AvgPool2d(kernel_size=32)
        self.global_max_pool_seg = nn.MaxPool2d(kernel_size=(self.input_height / 8, self.input_width / 8))
        self.global_avg_pool_seg = nn.AvgPool2d(kernel_size=(self.input_height / 8, self.input_width / 8))

        self.fc = nn.Linear(in_features=70, out_features=1)

        self.volume_lr_multiplier_layer = GradientMultiplyLayer().apply
        self.glob_max_lr_multiplier_layer = GradientMultiplyLayer().apply
        self.glob_avg_lr_multiplier_layer = GradientMultiplyLayer().apply

        self.device = device

        self.spatial_attention = SpatialAttention()
        self.conv1x1a = Conv1x1(in_channels=96,out_channels=8)

        self.conv1x1b = Conv1x1(in_channels=128, out_channels=8)
        self.conv1x1c = Conv1x1(in_channels=2048, out_channels=16)
    def set_gradient_multipliers(self, multiplier):
        self.volume_lr_multiplier_mask = (torch.ones((1,)) * multiplier).to(self.device)
        self.glob_max_lr_multiplier_mask = (torch.ones((1,)) * multiplier).to(self.device)
        self.glob_avg_lr_multiplier_mask = (torch.ones((1,)) * multiplier).to(self.device)

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.maxp1(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        x5 = self.conv4(x4)
        x6 = self.maxp2(x5)
        x7 = self.conv5(x6)
        x8 = self.conv6(x7)
        x9 = self.conv7(x8)
        x10 = self.conv8(x9)
        x11 = self.maxp3(x10)
        x12 = self.conv9(x11)
        x13 = self.conv9a(x12)
        x14 = self.upsa3(x13)
        x15 = self.conv6a(x14)
        x16 = self.upsa2(x15)
        x17 = self.conv3a(x16)
        x18 = self.upsa1(x17)
        # x19 = self.conv1a(x18)

        seg_mask1 = self.seg_mask1(x13)
        seg_mask2 = self.seg_mask2(x15)
        seg_mask3 = self.seg_mask3(x17)
        concats1 = torch.cat((x5, x17), dim=1)
        concats1 = self.conv1x1a(concats1)
        x17 = self.spatial_attention(inputs1=concats1, inputs2=x5)
        concats2 = torch.cat((x10, x15), dim=1)
        concats2 = self.conv1x1b(concats2)
        x15 = self.spatial_attention(inputs1=concats2, inputs2=x10)
        concats3 = torch.cat((x12, x13), dim=1)
        concats3 = self.conv1x1c(concats3)
        x13 = self.spatial_attention(inputs1=concats3, inputs2=x12)
        #
        concat1 = torch.cat((x17,  seg_mask3), dim=1)   #65
        concat1 = self.conv_concat1(concat1)
        concat1 = nn.functional.interpolate(concat1, scale_factor=0.5, mode='bilinear', align_corners=False)
        concat1s = nn.functional.interpolate(concat1, scale_factor=0.25, mode='bilinear', align_corners=False)
        concat1s = self.conv_concat1s(concat1s)
        concat2 = torch.cat((x15, seg_mask2), dim=1)#65
        concat2 = self.conv_concat2(concat2)
        concat_12 = torch.cat((concat1,concat2), dim=1)#130
        concat_12 = self.conv_concat3(concat_12)
        concat_12 = nn.functional.interpolate(concat_12, scale_factor=0.5, mode='bilinear', align_corners=False)
        concat_12s = nn.functional.interpolate(concat_12, scale_factor=0.25, mode='bilinear', align_corners=False)
        concat_12s = self.conv_concat3s(concat_12s)
        concat3 = torch.cat((x13, seg_mask1), dim=1)
        concat3 = self.conv_concat4(concat3)        #1025
        concat_123 = torch.cat((concat_12, concat3), dim=1)#1155
        concat_123 = self.conv_concat5(concat_123)
        concat_123s = nn.functional.interpolate(concat_123, scale_factor=0.125, mode='bilinear', align_corners=False)
        concat_123s = self.conv_concat5s(concat_123s)
        # x19 = nn.functional.interpolate(x19, scale_factor=0.125, mode='bilinear', align_corners=False)
        # cat = torch.cat([x17, seg_mask3], dim=1)
        # cat = self.volume_lr_multiplier_layer(cat, self.volume_lr_multiplier_mask)
        cat = self.volume_lr_multiplier_layer(concat_123, self.volume_lr_multiplier_mask)

        features = self.extractor(cat)
        # x1s = self.maxpool1(cat)
        # x2s = self.convs1(x1s)
        # x3s = self.spatial_attention(inputs1=x2s, inputs2=concat1s)
        # x4s = torch.cat((x2s,  x3s), dim=1)
        # x5s = self.maxpool2(x4s)
        # x6s = self.convs2(x5s)
        # x7s = self.spatial_attention(inputs1=x6s, inputs2=concat_12s)
        # x8s = torch.cat((x7s, x6s), dim=1)
        # x9s= self.maxpool3(x8s)
        # x10s = self.convs3(x9s)
        # x11s = self.spatial_attention(inputs1=x10s, inputs2=concat_123s)
        # features = torch.cat((x11s, x10s), dim=1)
        # features = self.convs(features)
        # features = self.convss(features)
        # features = self.convsss(features)

        global_max_feat = torch.max(torch.max(features, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]

        global_avg_feat = torch.mean(features, dim=(-1, -2), keepdim=True)
        global_max_seg1 = torch.max(torch.max(seg_mask1, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
        global_avg_seg1 = torch.mean(seg_mask1, dim=(-1, -2), keepdim=True)
        global_max_seg2 = torch.max(torch.max(seg_mask2, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
        global_avg_seg2 = torch.mean(seg_mask2, dim=(-1, -2), keepdim=True)
        global_max_seg3 = torch.max(torch.max(seg_mask3, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
        global_avg_seg3 = torch.mean(seg_mask3, dim=(-1, -2), keepdim=True)

        global_max_feat = global_max_feat.reshape(global_max_feat.size(0), -1)

        global_avg_feat = global_avg_feat.reshape(global_avg_feat.size(0), -1)

        global_max_seg1 = global_max_seg1.reshape(global_max_seg1.size(0), -1)
        global_max_seg1 = self.glob_max_lr_multiplier_layer(global_max_seg1, self.glob_max_lr_multiplier_mask)
        global_max_seg2 = global_max_seg2.reshape(global_max_seg2.size(0), -1)
        global_max_seg2 = self.glob_max_lr_multiplier_layer(global_max_seg2, self.glob_max_lr_multiplier_mask)
        global_max_seg3 = global_max_seg3.reshape(global_max_seg3.size(0), -1)
        global_max_seg3 = self.glob_max_lr_multiplier_layer(global_max_seg3, self.glob_max_lr_multiplier_mask)
        global_avg_seg1 = global_avg_seg1.reshape(global_avg_seg1.size(0), -1)
        global_avg_seg1 = self.glob_avg_lr_multiplier_layer(global_avg_seg1, self.glob_avg_lr_multiplier_mask)
        global_avg_seg2 = global_avg_seg2.reshape(global_avg_seg2.size(0), -1)
        global_avg_seg2 = self.glob_avg_lr_multiplier_layer(global_avg_seg2, self.glob_avg_lr_multiplier_mask)
        global_avg_seg3 = global_avg_seg3.reshape(global_avg_seg3.size(0), -1)
        global_avg_seg3 = self.glob_avg_lr_multiplier_layer(global_avg_seg3, self.glob_avg_lr_multiplier_mask)

        fc_in = torch.cat([global_max_feat, global_avg_feat, global_max_seg1, global_avg_seg1,global_max_seg2,global_avg_seg2,global_max_seg3,global_avg_seg3], dim=1)

        fc_in = fc_in.reshape(fc_in.size(0), -1)
        prediction = self.fc(fc_in)
        # return prediction, seg_mask

        return prediction, seg_mask1, seg_mask2, seg_mask3
    # def forward(self, input):
    #     volume = self.volume(input)
    #     seg_mask = self.seg_mask(volume)
    #
    #     cat = torch.cat([volume, seg_mask], dim=1)
    #
    #     cat = self.volume_lr_multiplier_layer(cat, self.volume_lr_multiplier_mask)
    #
    #     features = self.extractor(cat)
    #     global_max_feat = torch.max(torch.max(features, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
    #     global_avg_feat = torch.mean(features, dim=(-1, -2), keepdim=True)
    #     global_max_seg = torch.max(torch.max(seg_mask, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
    #     global_avg_seg = torch.mean(seg_mask, dim=(-1, -2), keepdim=True)
    #
    #     global_max_feat = global_max_feat.reshape(global_max_feat.size(0), -1)
    #     global_avg_feat = global_avg_feat.reshape(global_avg_feat.size(0), -1)
    #
    #     global_max_seg = global_max_seg.reshape(global_max_seg.size(0), -1)
    #     global_max_seg = self.glob_max_lr_multiplier_layer(global_max_seg, self.glob_max_lr_multiplier_mask)
    #     global_avg_seg = global_avg_seg.reshape(global_avg_seg.size(0), -1)
    #     global_avg_seg = self.glob_avg_lr_multiplier_layer(global_avg_seg, self.glob_avg_lr_multiplier_mask)
    #
    #     fc_in = torch.cat([global_max_feat, global_avg_feat, global_max_seg, global_avg_seg], dim=1)
    #     fc_in = fc_in.reshape(fc_in.size(0), -1)
    #     prediction = self.fc(fc_in)
    #     return prediction, seg_mask


class GradientMultiplyLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mask_bw):
        ctx.save_for_backward(mask_bw)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        mask_bw, = ctx.saved_tensors
        return grad_output.mul(mask_bw), None
