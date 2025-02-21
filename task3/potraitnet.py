import torch
import torch.nn as nn
import numpy as np


def make_bilinear_weights(size, num_channels):
    ''' Make a 2D bilinear kernel suitable for upsampling
    Stack the bilinear kernel for application to tensor '''
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

    # print filt
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, 1, size, size)
    for i in range(num_channels):
        w[i, 0] = filt
    return w    

# 1x1 Convolution
def conv_1x1(inp, oup):
    """1x1 convolution with padding"""
    return nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, stride=1, padding=0, bias=False)

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(num_features=oup, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(inplace=True)
    )

def conv_bn(inp, oup, kernel, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=kernel, stride=stride, padding=(kernel-1)//2, bias=False),
        nn.BatchNorm2d(num_features=oup, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(inplace=True)
    )

def conv_dw(inp, oup, kernel, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel, stride, (kernel-1)//2, groups=inp, bias=False),
        nn.BatchNorm2d(num_features=inp, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(num_features=oup, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(inplace=True),
    )

# Inverted Residual is used for building Encoder Block
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, dilation=1):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(num_features=inp * expand_ratio, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 
                      kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
                      groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(num_features=inp * expand_ratio, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(num_features=oup, eps=1e-05, momentum=0.1, affine=True),
        )
        
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
        

# Residual Block is used for building PotraitNet Block
class ResidualBlock(nn.Module):
    def __init__(self, inp, oup, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            conv_dw(inp, oup, 3, stride=stride),
            nn.Conv2d(in_channels=oup, out_channels=oup, kernel_size=3, stride=1, padding=1, groups=oup, bias=False),
            nn.BatchNorm2d(num_features=oup, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=oup, out_channels=oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=oup, eps=1e-05, momentum=0.1, affine=True),
        )
        if inp == oup:
            self.residual = None
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=oup, eps=1e-05, momentum=0.1, affine=True),
            )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        
        out = self.block(x)
        if self.residual is not None:
            residual = self.residual(x)
            
        out += residual
        out = self.relu(out)
        return out
    
        
class PotraitNet(nn.Module):
    def __init__(self, n_class=2, useUpsample=False, useDeconvGroup=False, addEdge=False, 
                 channelRatio=1.0, minChannel=16, weightInit=True, video=False):
        
        super(PotraitNet, self).__init__()
        self.addEdge = addEdge
        self.channelRatio = channelRatio
        self.minChannel = minChannel
        self.useDeconvGroup = useDeconvGroup

        if video == True:
            self.stage0 = conv_bn(4, self.channels(32), 3, 2) # 1 -> 1/2
        else:
            self.stage0 = conv_bn(3, self.channels(32), 3, 2) # 1 -> 1/2

        # Encoder Structure
        self.stage1 = nn.Sequential( # 1 -> 1/2
            self.stage0 , # 1 -> 1/2
            InvertedResidual(self.channels(32), self.channels(16), 1, 1) # 1/2 -> 1/2
        )
        
        self.stage2 = nn.Sequential( # 1/2 -> 1/4
            InvertedResidual(self.channels(16), self.channels(24), 2, 6), # 1/2 -> 1/4
            InvertedResidual(self.channels(24), self.channels(24), 1, 6), # 1/4 -> 1/4
        )
        
        self.stage3 = nn.Sequential( # 1/4 -> 1/8
            InvertedResidual(self.channels(24), self.channels(32), 2, 6), # 1/4 -> 1/8
            InvertedResidual(self.channels(32), self.channels(32), 1, 6), # 1/8 -> 1/8
            InvertedResidual(self.channels(32), self.channels(32), 1, 6), # 1/8 -> 1/8
        )
        
        self.stage4 = nn.Sequential( # 1/8 -> 1/16
            InvertedResidual(self.channels(32), self.channels(64), 2, 6), # 1/8 -> 1/16
            InvertedResidual(self.channels(64), self.channels(64), 1, 6), # 1/16 -> 1/16
            InvertedResidual(self.channels(64), self.channels(64), 1, 6), # 1/16 -> 1/16
            InvertedResidual(self.channels(64), self.channels(64), 1, 6), # 1/16 -> 1/16
            InvertedResidual(self.channels(64), self.channels(96), 1, 6), # 1/16 -> 1/16
            InvertedResidual(self.channels(96), self.channels(96), 1, 6), # 1/16 -> 1/16
            InvertedResidual(self.channels(96), self.channels(96), 1, 6), # 1/16 -> 1/16
        )
        
        self.stage5 = nn.Sequential( # 1/16 -> 1/32
            InvertedResidual(self.channels(96), self.channels(160), 2, 6), # 1/16 -> 1/32
            InvertedResidual(self.channels(160), self.channels(160), 1, 6), # 1/32 -> 1/32
            InvertedResidual(self.channels(160), self.channels(160), 1, 6), # 1/32 -> 1/32
            InvertedResidual(self.channels(160), self.channels(320), 1, 6) # 1/32 -> 1/32
        )

        # Decoder Structure
        if useUpsample == True:
            self.deconv1 = nn.Upsample(scale_factor=2, mode='bilinear')
            self.deconv2 = nn.Upsample(scale_factor=2, mode='bilinear')
            self.deconv3 = nn.Upsample(scale_factor=2, mode='bilinear')    
            self.deconv4 = nn.Upsample(scale_factor=2, mode='bilinear')
            self.deconv5 = nn.Upsample(scale_factor=2, mode='bilinear')
        else:
            self.deconv1 = nn.ConvTranspose2d(self.channels(96), self.channels(96), groups = self.deconvGroup(96), 
                                                kernel_size=4, stride=2, padding=1, bias=False)
            self.deconv2 = nn.ConvTranspose2d(self.channels(32), self.channels(32), groups = self.deconvGroup(32), 
                                                kernel_size=4, stride=2, padding=1, bias=False)
            self.deconv3 = nn.ConvTranspose2d(self.channels(24), self.channels(24), groups = self.deconvGroup(24), 
                                                kernel_size=4, stride=2, padding=1, bias=False)
            self.deconv4 = nn.ConvTranspose2d(self.channels(16), self.channels(16), groups = self.deconvGroup(16), 
                                                kernel_size=4, stride=2, padding=1, bias=False)
            self.deconv5 = nn.ConvTranspose2d(self.channels(8),  self.channels(8),  groups = self.deconvGroup(8),  
                                                kernel_size=4, stride=2, padding=1, bias=False)
        
        self.transit1 = ResidualBlock(self.channels(320), self.channels(96))
        self.transit2 = ResidualBlock(self.channels(96),  self.channels(32))
        self.transit3 = ResidualBlock(self.channels(32),  self.channels(24))
        self.transit4 = ResidualBlock(self.channels(24),  self.channels(16))
        self.transit5 = ResidualBlock(self.channels(16),  self.channels(8))

        self.pred = nn.Conv2d(self.channels(8), n_class, 3, 1, 1, bias=False)
        if self.addEdge == True:
            self.edge = nn.Conv2d(self.channels(8), n_class, 3, 1, 1, bias=False)
        
        if weightInit == True:
            self._initialize_weights()
    
    def forward(self, x):
        # Encoder Pass
        feature_1_2  = self.stage1(x)
        feature_1_4  = self.stage2(feature_1_2)
        feature_1_8  = self.stage3(feature_1_4)
        feature_1_16 = self.stage4(feature_1_8)
        feature_1_32 = self.stage5(feature_1_16)

        # Decoder Pass
        up_1_16 = self.deconv1(self.transit1(feature_1_32))
        up_1_8  = self.deconv2(self.transit2(feature_1_16 + up_1_16))
        up_1_4  = self.deconv3(self.transit3(feature_1_8 + up_1_8))
        up_1_2  = self.deconv4(self.transit4(feature_1_4 + up_1_4))
        up_1_1  = self.deconv5(self.transit5(up_1_2))
        
        pred = self.pred(up_1_1)
        if self.addEdge == True:
            edge = self.edge(up_1_1)
            return pred, edge
        else:
            return pred
    
    def channels(self, channels):
        min_channel = min(channels, self.minChannel)
        return max(min_channel, int(channels*self.channelRatio))
    
    def deconvGroup(self, groups):
        if self.useDeconvGroup == True:
            return self.channels(groups)
        else:
            return 1
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = make_bilinear_weights(m.kernel_size[0], m.out_channels) # same as caffe
                m.weight.data.copy_(initial_weight)
                if self.useDeconvGroup == True:
                    m.requires_grad = False # use deconvolution as bilinear upsample
                    print ("freeze deconv")
        pass