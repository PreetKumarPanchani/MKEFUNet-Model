import torch
import torch.nn as nn
#from resnet import resnet50

class Conv2D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, bias=False, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_c, out_c,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=bias
            ),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x

class residual_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.network = nn.Sequential(
            Conv2D(in_c, out_c),
            Conv2D(out_c, out_c, kernel_size=1, padding=0, act=False)

        )
        self.shortcut = Conv2D(in_c, out_c, kernel_size=1, padding=0, act=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_init):
        x = self.network(x_init)
        s = self.shortcut(x_init)
        x = self.relu(x+s)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class ClassifierWithLatentVector(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1_1 = nn.Sequential(
            nn.Linear(in_c, in_c//8, bias=False), nn.ReLU(),)

        self.fc1_2 = nn.Sequential(    
            nn.Linear(in_c//8, out_c[0], bias=False)
        )
        


        
    ## Taking Avg Pooling to reduce Feature Size to 1x1 
    ## Then Using Linear Layers ==> To predict the Information like "chagas or non_chagas"
    def forward(self, feats):
        pool = self.avg_pool(feats).view(feats.shape[0], feats.shape[1])
        num_chagas_latent_vector = self.fc1_1(pool)
        num_chagas = self.fc1_2(num_chagas_latent_vector)

        return num_chagas, num_chagas_latent_vector




class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

from torchvision.models import densenet





class Encoder_DenseNet121(nn.Module):
    def __init__(self, ch):
        super().__init__()

        """ DenseNet210 """
        backbone = densenet.densenet121(pretrained=True)
        self.features = backbone.features

        """ Reduce feature channels """
        self.c1 = Conv2D(64, ch)
        self.c2 = Conv2D(256, ch)
        self.c3 = Conv2D(512, ch)
        self.c4 = Conv2D(1024, ch)  # DenseNet210 has 1920 output channels

    def forward(self, x):
        """ Backbone: DenseNet210 """
        x0 = self.features.conv0(x)
        x0 = self.features.norm0(x0)
        x0 = self.features.relu0(x0)
        #print(x0.shape)             # torch.Size([1, 64, 128, 128])
        x1 = self.features.pool0(x0)
        x1 = self.features.denseblock1(x1)
        #print(x1.shape)             # torch.Size([1, 256, 64, 64])
        x2 = self.features.transition1(x1)
        x2 = self.features.denseblock2(x2)
        #print(x2.shape)             # torch.Size([1, 512, 32, 32])
        x3 = self.features.transition2(x2)
        x3 = self.features.denseblock3(x3)
        #print(x3.shape)             # torch.Size([1, 1792, 16, 16])
        x4 = self.features.transition3(x3)
        x4 = self.features.denseblock4(x4)
        x4 = self.features.norm5(x4)
        #print(x4.shape)             # torch.Size([1, 1920, 8, 8])

        c1 = self.c1(x0)
        c2 = self.c2(x1)
        c3 = self.c3(x2)
        c4 = self.c4(x3)

        return c1, c2, c3, c4


class MultiKernelDilatedConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.c1 = Conv2D(in_c, out_c, kernel_size=1, padding=0)
        self.c2 = Conv2D(in_c, out_c, kernel_size=3, padding=1)
        self.c3 = Conv2D(in_c, out_c, kernel_size=7, padding=3)
        self.c4 = Conv2D(in_c, out_c, kernel_size=11, padding=5)
        self.s1 = Conv2D(out_c*4, out_c, kernel_size=1, padding=0)

        self.d1 = Conv2D(out_c, out_c, kernel_size=3, padding=1, dilation=1)
        self.d2 = Conv2D(out_c, out_c, kernel_size=3, padding=3, dilation=3)
        self.d3 = Conv2D(out_c, out_c, kernel_size=3, padding=7, dilation=7)
        self.d4 = Conv2D(out_c, out_c, kernel_size=3, padding=11, dilation=11)
        self.s2 = Conv2D(out_c*4, out_c, kernel_size=1, padding=0, act=False)
        self.s3 = Conv2D(in_c, out_c, kernel_size=1, padding=0, act=False)

        self.ca = ChannelAttention(out_c)
        self.sa = SpatialAttention()

#
    def forward(self, x):
        x0 = x
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)
        x = torch.cat([x1, x2, x3, x4], axis=1)
        x = self.s1(x)

        x1 = self.d1(x)
        x2 = self.d2(x)
        x3 = self.d3(x)
        x4 = self.d4(x)
        x = torch.cat([x1, x2, x3, x4], axis=1)
        x = self.s2(x)
        s = self.c3(x0)

        x = self.relu(x+s)
        x = x * self.ca(x)
        x = x * self.sa(x)

        return x


class MultiscaleFeatureFusion(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up_2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.c1 = Conv2D(in_c, out_c)
        self.c2 = Conv2D(out_c+in_c, out_c)
        self.c3 = Conv2D(in_c+ out_c, out_c)

        self.ca = ChannelAttention(out_c)
        self.sa = SpatialAttention()

    def forward(self, f1, f2, f3):
        x1 = self.up_2(f1)
        x1 = self.c1(x1)
        x1 = torch.cat([x1, f2], axis=1)

        x1 = self.up_2(x1)
        x1 = self.c2(x1)
        
        x1 = torch.cat([x1, f3], axis=1)
        x1 = self.up_2(x1)
        x1 = self.c3(x1)

        x1 = x1 * self.ca(x1)
        x1 = x1 * self.sa(x1)
        return x1

class DecoderBlock_WithEmbeddings(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.r1 = residual_block(in_c[0]+in_c[1], out_c)
        self.r2 = residual_block(out_c, out_c)

    def forward(self, x, s, embedded_num_pixel):
        
        x = self.up(x )
        emb = embedded_num_pixel[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        x = x + emb
        x = torch.cat([x, s], axis=1)
        
        x = self.r1(x )
        x = self.r2(x)
        
        return x 

import torch.nn.functional as F


class MKEF_Unet(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.encoder = Encoder_DenseNet121(96)
        

        """ Number of Chagas Instances and their Latent Vectors"""
        self.classifier = ClassifierWithLatentVector(96, [3, 1, 1])


        """ Embedding Vector """
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                96//8,
                96
            ),
        )
        """ MultiKernelDilation Conv  """
        self.c1 = MultiKernelDilatedConv(96, 96)
        self.c2 = MultiKernelDilatedConv(96, 96)
        self.c3 = MultiKernelDilatedConv(96, 96)
        self.c4 = MultiKernelDilatedConv(96, 96)

        """ Decoder """
        self.d1 = DecoderBlock_WithEmbeddings([96, 96], 96)
        self.d2 = DecoderBlock_WithEmbeddings([96, 96], 96)
        self.d3 = DecoderBlock_WithEmbeddings([96, 96], 96)


        """ Multiscale Feature Fusion """
        self.msf_seg = MultiscaleFeatureFusion(96, 96)        


        """ Segmentation Output """
        self.seg_out = nn.Conv2d(96, 1, kernel_size=1, padding=0)



    def forward(self, image, label):
        s0 = image

        """ Encoder """
        s1, s2, s3, s4 = self.encoder(image)
        
        """ Classifier  """
        num_chagas, num_chagas_latent_vector   = self.classifier(s4)
        
        """ Embedding Vector """
        embedded_vector_segmentation = self.emb_layer(num_chagas_latent_vector )

        """ MultiKernel Conv + Dilation """
        x1 = self.c1(s1)
        x2 = self.c2(s2 )
        x3 = self.c3(s3)
        x4 = self.c4(s4)


        """ Segmentation Decoder  """

        d1 = self.d1( x4 , x3 , embedded_vector_segmentation )
        
        d2 = self.d2( d1 , x2 , embedded_vector_segmentation )
        
        d3 = self.d3( d2 , x1 , embedded_vector_segmentation )

        """ Multiscale Feature Fusion """
        x = self.msf_seg(d1, d2, d3)

        """ Segmentation Last Layer """
        segmentation_output = self.seg_out(x)

        return segmentation_output ,  num_chagas 


if __name__ == "__main__":
    inputs = torch.randn((2, 3, 512, 512))
    model = MKEF_Unet()
    y = model(inputs, inputs[: , 0 , : , :] )
    
