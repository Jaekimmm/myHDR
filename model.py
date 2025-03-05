import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from torch.autograd import Variable

        
class LIGHTFUSE(nn.Module):
    def __init__(self, args):
        super(LIGHTFUSE, self).__init__()
        nChannel = args.nChannel
        nFeat = args.nFeat
        self.args = args
        print(f"[INFO] nChannel : {nChannel}, nFeat : {nFeat}")
        
        # DetailNet
        self.pointwise1 = nn.Conv2d(nChannel, nFeat, kernel_size=1, stride=1, padding=0, bias=True)
        self.pointwise2 = nn.Conv2d(nFeat, nFeat, kernel_size=1, stride=1, padding=0, bias=True)
        self.pointwise3 = nn.Conv2d(nFeat, 3, kernel_size=1, stride=1, padding=0, bias=True)

        # GlobalNet : 
        # Depthwise & Separable Convolution Layers
        self.depthwise1 = nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel)
        self.depthwise2 = nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel)
        self.separable_conv = nn.Sequential(
          nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel),
          nn.Conv2d(nChannel,        3, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # Upsampling
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.upsample3 = nn.Upsample(scale_factor=2)

        # Final Output
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x1, x2, x3):
        DEBUG_FLAG = 0
        
        if DEBUG_FLAG == 2: print(f"[INFO] x1: {x1.min():.6f} ~ {x1.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x2: {x2.min():.6f} ~ {x2.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x3: {x3.min():.6f} ~ {x3.max():.6f}")
        
        x = torch.cat((x1, x3), dim=1)
        
        # DetailNet
        if DEBUG_FLAG == 1: print(f"[INFO] x : {x.shape}")
        dnet_1 = self.relu(self.pointwise1(x))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_1 : {dnet_1.shape}")
        dnet_2 = self.relu(self.pointwise2(dnet_1))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_2 : {dnet_2.shape}")
        dnet_out = self.relu(self.pointwise3(dnet_2))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_out : {dnet_out.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] dnet_out: {dnet_out.min()} ~ {dnet_out.max()}")
        
        # GlobalNet 
        gnet_dconv_1 = self.depthwise1(x)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_dconv_1 : {gnet_dconv_1.shape}")
        gnet_dconv_2 = self.depthwise2(gnet_dconv_1)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_dconv_2 : {gnet_dconv_2.shape}")
        gnet_dconv_3 = self.separable_conv(gnet_dconv_2)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_dconv_3 : {gnet_dconv_3.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] gnet_dconv_3: {gnet_dconv_3.min()} ~ {gnet_dconv_3.max()}")

        # Upsampling
        gnet_up_1 = self.upsample1(gnet_dconv_3)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_up_1 : {gnet_up_1.shape}")
        gnet_up_2 = self.upsample2(gnet_up_1)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_up_2 : {gnet_up_2.shape}")
        gnet_out = self.upsample3(gnet_up_2)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_up_3 : {gnet_out.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] gnet_out: {gnet_out.min()} ~ {gnet_out.max()}")

        # Element-wise Addition
        out = dnet_out + gnet_out
        if DEBUG_FLAG == 2: print(f"[INFO] add_out: {out.min()} ~ {out.max()}")
        #out = torch.sigmoid(out)
        out = self.tanh(out)
        if DEBUG_FLAG == 2: print(f"[INFO] tanh_out: {out.min()} ~ {out.max()}")
        return out
 
class LIGHTFUSE_sigmoid(nn.Module):
    def __init__(self, args):
        super(LIGHTFUSE_sigmoid, self).__init__()
        nChannel = args.nChannel
        nFeat = args.nFeat
        self.args = args
        print(f"[INFO] nChannel : {nChannel}, nFeat : {nFeat}")
        
        # DetailNet
        self.pointwise1 = nn.Conv2d(nChannel, nFeat, kernel_size=1, stride=1, padding=0, bias=True)
        self.pointwise2 = nn.Conv2d(nFeat, nFeat, kernel_size=1, stride=1, padding=0, bias=True)
        self.pointwise3 = nn.Conv2d(nFeat, 3, kernel_size=1, stride=1, padding=0, bias=True)

        # GlobalNet : 
        # Depthwise & Separable Convolution Layers
        self.depthwise1 = nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel)
        self.depthwise2 = nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel)
        self.separable_conv = nn.Sequential(
          nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel),
          nn.Conv2d(nChannel,        3, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # Upsampling (ConvTranspose2d)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.upsample3 = nn.Upsample(scale_factor=2)

        # Final Output
        self.relu = nn.ReLU()

    def forward(self, x1, x2, x3):
        DEBUG_FLAG = 0
        
        if DEBUG_FLAG == 2: print(f"[INFO] x1: {x1.min():.6f} ~ {x1.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x2: {x2.min():.6f} ~ {x2.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x3: {x3.min():.6f} ~ {x3.max():.6f}")
        
        x = torch.cat((x1, x3), dim=1)
        
        # DetailNet
        if DEBUG_FLAG == 1: print(f"[INFO] x : {x.shape}")
        dnet_1 = self.relu(self.pointwise1(x))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_1 : {dnet_1.shape}")
        dnet_2 = self.relu(self.pointwise2(dnet_1))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_2 : {dnet_2.shape}")
        dnet_out = self.relu(self.pointwise3(dnet_2))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_out : {dnet_out.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] dnet_out: {dnet_out.min()} ~ {dnet_out.max()}")
        
        # GlobalNet 
        gnet_dconv_1 = self.depthwise1(x)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_dconv_1 : {gnet_dconv_1.shape}")
        gnet_dconv_2 = self.depthwise2(gnet_dconv_1)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_dconv_2 : {gnet_dconv_2.shape}")
        gnet_dconv_3 = self.separable_conv(gnet_dconv_2)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_dconv_3 : {gnet_dconv_3.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] gnet_dconv_3: {gnet_dconv_3.min()} ~ {gnet_dconv_3.max()}")

        # Upsampling
        gnet_up_1 = self.upsample1(gnet_dconv_3)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_up_1 : {gnet_up_1.shape}")
        gnet_up_2 = self.upsample2(gnet_up_1)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_up_2 : {gnet_up_2.shape}")
        gnet_out = self.upsample3(gnet_up_2)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_up_3 : {gnet_out.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] gnet_out: {gnet_out.min()} ~ {gnet_out.max()}")

        # Element-wise Addition
        out = dnet_out + gnet_out
        if DEBUG_FLAG == 2: print(f"[INFO] add_out: {out.min()} ~ {out.max()}")
        out = torch.sigmoid(out)
        #out = self.tanh(out)
        if DEBUG_FLAG == 2: print(f"[INFO] sigmoid_out: {out.min()} ~ {out.max()}")
        return out
 
class LIGHTFUSE_bilinear_upscale(nn.Module):
    def __init__(self, args):
        super(LIGHTFUSE_bilinear_upscale, self).__init__()
        nChannel = args.nChannel
        nFeat = args.nFeat
        self.args = args
        print(f"[INFO] nChannel : {nChannel}, nFeat : {nFeat}")
        
        # DetailNet
        self.pointwise1 = nn.Conv2d(nChannel, nFeat, kernel_size=1, stride=1, padding=0, bias=True)
        self.pointwise2 = nn.Conv2d(nFeat, nFeat, kernel_size=1, stride=1, padding=0, bias=True)
        self.pointwise3 = nn.Conv2d(nFeat, 3, kernel_size=1, stride=1, padding=0, bias=True)

        # GlobalNet : 
        # Depthwise & Separable Convolution Layers
        self.depthwise1 = nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel)
        self.depthwise2 = nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel)
        self.separable_conv = nn.Sequential(
          nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel),
          nn.Conv2d(nChannel,        3, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # Upsampling (ConvTranspose2d)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear')

        # Final Output
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x1, x2, x3):
        DEBUG_FLAG = 0
        
        if DEBUG_FLAG == 2: print(f"[INFO] x1: {x1.min():.6f} ~ {x1.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x2: {x2.min():.6f} ~ {x2.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x3: {x3.min():.6f} ~ {x3.max():.6f}")
        
        x = torch.cat((x1, x3), dim=1)
        
        # DetailNet
        if DEBUG_FLAG == 1: print(f"[INFO] x : {x.shape}")
        dnet_1 = self.relu(self.pointwise1(x))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_1 : {dnet_1.shape}")
        dnet_2 = self.relu(self.pointwise2(dnet_1))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_2 : {dnet_2.shape}")
        dnet_out = self.relu(self.pointwise3(dnet_2))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_out : {dnet_out.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] dnet_out: {dnet_out.min()} ~ {dnet_out.max()}")
        
        # GlobalNet 
        gnet_dconv_1 = self.depthwise1(x)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_dconv_1 : {gnet_dconv_1.shape}")
        gnet_dconv_2 = self.depthwise2(gnet_dconv_1)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_dconv_2 : {gnet_dconv_2.shape}")
        gnet_dconv_3 = self.separable_conv(gnet_dconv_2)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_dconv_3 : {gnet_dconv_3.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] gnet_dconv_3: {gnet_dconv_3.min()} ~ {gnet_dconv_3.max()}")

        # Upsampling
        gnet_up_1 = self.upsample1(gnet_dconv_3)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_up_1 : {gnet_up_1.shape}")
        gnet_up_2 = self.upsample2(gnet_up_1)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_up_2 : {gnet_up_2.shape}")
        gnet_out = self.upsample3(gnet_up_2)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_up_3 : {gnet_out.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] gnet_out: {gnet_out.min()} ~ {gnet_out.max()}")

        # Element-wise Addition
        out = dnet_out + gnet_out
        if DEBUG_FLAG == 2: print(f"[INFO] add_out: {out.min()} ~ {out.max()}")
        out = torch.sigmoid(out)
        #out = self.tanh(out)
        if DEBUG_FLAG == 2: print(f"[INFO] sigmoid_out: {out.min()} ~ {out.max()}")
        return out

class LIGHTFUSE_sigmoid_skip_long(nn.Module):
    def __init__(self, args):
        super(LIGHTFUSE_sigmoid_skip_long, self).__init__()
        nChannel = args.nChannel
        nFeat = args.nFeat
        self.args = args
        print(f"[INFO] nChannel : {nChannel}, nFeat : {nFeat}")
        
        # DetailNet
        self.pointwise1 = nn.Conv2d(nChannel, nFeat, kernel_size=1, stride=1, padding=0, bias=True)
        self.pointwise2 = nn.Conv2d(nFeat, nFeat, kernel_size=1, stride=1, padding=0, bias=True)
        self.pointwise3 = nn.Conv2d(nFeat, 3, kernel_size=1, stride=1, padding=0, bias=True)

        # GlobalNet : 
        # Depthwise & Separable Convolution Layers
        self.depthwise1 = nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel)
        self.depthwise2 = nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel)
        self.separable_conv = nn.Sequential(
          nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel),
          nn.Conv2d(nChannel,        3, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # Upsampling (ConvTranspose2d)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.upsample3 = nn.Upsample(scale_factor=2)

        # Final Output
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x1, x2, x3):
        DEBUG_FLAG = 0
        
        if DEBUG_FLAG == 2: print(f"[INFO] x1: {x1.min():.6f} ~ {x1.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x2: {x2.min():.6f} ~ {x2.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x3: {x3.min():.6f} ~ {x3.max():.6f}")
        
        x = torch.cat((x1, x3), dim=1)
        
        # DetailNet
        if DEBUG_FLAG == 1: print(f"[INFO] x : {x.shape}")
        dnet_1 = self.relu(self.pointwise1(x))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_1 : {dnet_1.shape}")
        dnet_2 = self.relu(self.pointwise2(dnet_1))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_2 : {dnet_2.shape}")
        dnet_out = self.relu(self.pointwise3(dnet_2))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_out : {dnet_out.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] dnet_out: {dnet_out.min()} ~ {dnet_out.max()}")
        
        # GlobalNet 
        gnet_dconv_1 = self.depthwise1(x)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_dconv_1 : {gnet_dconv_1.shape}")
        gnet_dconv_2 = self.depthwise2(gnet_dconv_1)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_dconv_2 : {gnet_dconv_2.shape}")
        gnet_dconv_3 = self.separable_conv(gnet_dconv_2)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_dconv_3 : {gnet_dconv_3.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] gnet_dconv_3: {gnet_dconv_3.min()} ~ {gnet_dconv_3.max()}")

        # Upsampling
        gnet_up_1 = self.upsample1(gnet_dconv_3)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_up_1 : {gnet_up_1.shape}")
        gnet_up_2 = self.upsample2(gnet_up_1)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_up_2 : {gnet_up_2.shape}")
        gnet_out = self.upsample3(gnet_up_2)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_up_3 : {gnet_out.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] gnet_out: {gnet_out.min()} ~ {gnet_out.max()}")

        # Element-wise Addition
        out = dnet_out + gnet_out + x3  # skip long exposure
        if DEBUG_FLAG == 2: print(f"[INFO] add_out: {out.min()} ~ {out.max()}")
        out = torch.sigmoid(out)
        #out = self.tanh(out)
        if DEBUG_FLAG == 2: print(f"[INFO] sigmoid_out: {out.min()} ~ {out.max()}")
        return out
 
class LIGHTFUSE_sigmoid_skip_short(nn.Module):
    def __init__(self, args):
        super(LIGHTFUSE_sigmoid_skip_short, self).__init__()
        nChannel = args.nChannel
        nFeat = args.nFeat
        self.args = args
        print(f"[INFO] nChannel : {nChannel}, nFeat : {nFeat}")
        
        # DetailNet
        self.pointwise1 = nn.Conv2d(nChannel, nFeat, kernel_size=1, stride=1, padding=0, bias=True)
        self.pointwise2 = nn.Conv2d(nFeat, nFeat, kernel_size=1, stride=1, padding=0, bias=True)
        self.pointwise3 = nn.Conv2d(nFeat, 3, kernel_size=1, stride=1, padding=0, bias=True)

        # GlobalNet : 
        # Depthwise & Separable Convolution Layers
        self.depthwise1 = nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel)
        self.depthwise2 = nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel)
        self.separable_conv = nn.Sequential(
          nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel),
          nn.Conv2d(nChannel,        3, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # Upsampling (ConvTranspose2d)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.upsample3 = nn.Upsample(scale_factor=2)

        # Final Output
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x1, x2, x3):
        DEBUG_FLAG = 0
        
        if DEBUG_FLAG == 2: print(f"[INFO] x1: {x1.min():.6f} ~ {x1.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x2: {x2.min():.6f} ~ {x2.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x3: {x3.min():.6f} ~ {x3.max():.6f}")
        
        x = torch.cat((x1, x3), dim=1)
        
        # DetailNet
        if DEBUG_FLAG == 1: print(f"[INFO] x : {x.shape}")
        dnet_1 = self.relu(self.pointwise1(x))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_1 : {dnet_1.shape}")
        dnet_2 = self.relu(self.pointwise2(dnet_1))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_2 : {dnet_2.shape}")
        dnet_out = self.relu(self.pointwise3(dnet_2))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_out : {dnet_out.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] dnet_out: {dnet_out.min()} ~ {dnet_out.max()}")
        
        # GlobalNet 
        gnet_dconv_1 = self.depthwise1(x)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_dconv_1 : {gnet_dconv_1.shape}")
        gnet_dconv_2 = self.depthwise2(gnet_dconv_1)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_dconv_2 : {gnet_dconv_2.shape}")
        gnet_dconv_3 = self.separable_conv(gnet_dconv_2)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_dconv_3 : {gnet_dconv_3.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] gnet_dconv_3: {gnet_dconv_3.min()} ~ {gnet_dconv_3.max()}")

        # Upsampling
        gnet_up_1 = self.upsample1(gnet_dconv_3)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_up_1 : {gnet_up_1.shape}")
        gnet_up_2 = self.upsample2(gnet_up_1)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_up_2 : {gnet_up_2.shape}")
        gnet_out = self.upsample3(gnet_up_2)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_up_3 : {gnet_out.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] gnet_out: {gnet_out.min()} ~ {gnet_out.max()}")

        # Element-wise Addition
        out = dnet_out + gnet_out + x1  # skip short exposure
        if DEBUG_FLAG == 2: print(f"[INFO] add_out: {out.min()} ~ {out.max()}")
        out = torch.sigmoid(out)
        #out = self.tanh(out)
        if DEBUG_FLAG == 2: print(f"[INFO] sigmoid_out: {out.min()} ~ {out.max()}")
        return out
 
class LIGHTFUSE_detail_only(nn.Module):
    def __init__(self, args):
        super(LIGHTFUSE_detail_only, self).__init__()
        nChannel = args.nChannel
        nFeat = args.nFeat
        self.args = args
        print(f"[INFO] nChannel : {nChannel}, nFeat : {nFeat}")
        
        # DetailNet
        self.pointwise1 = nn.Conv2d(nChannel, nFeat, kernel_size=1, stride=1, padding=0, bias=True)
        self.pointwise2 = nn.Conv2d(nFeat, nFeat, kernel_size=1, stride=1, padding=0, bias=True)
        self.pointwise3 = nn.Conv2d(nFeat, 3, kernel_size=1, stride=1, padding=0, bias=True)

        ## GlobalNet : 
        ## Depthwise & Separable Convolution Layers
        #self.depthwise1 = nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel)
        #self.depthwise2 = nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel)
        #self.separable_conv = nn.Sequential(
        #  nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel),
        #  nn.Conv2d(nChannel,        3, kernel_size=1, stride=1, padding=0, bias=True)
        #)

        ## Upsampling
        #self.upsample1 = nn.Upsample(scale_factor=2)
        #self.upsample2 = nn.Upsample(scale_factor=2)
        #self.upsample3 = nn.Upsample(scale_factor=2)

        # Final Output
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x1, x2, x3):
        DEBUG_FLAG = 0
        
        if DEBUG_FLAG == 2: print(f"[INFO] x1: {x1.min():.6f} ~ {x1.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x2: {x2.min():.6f} ~ {x2.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x3: {x3.min():.6f} ~ {x3.max():.6f}")
        
        x = torch.cat((x1, x3), dim=1)
        
        # DetailNet
        if DEBUG_FLAG == 1: print(f"[INFO] x : {x.shape}")
        dnet_1 = self.relu(self.pointwise1(x))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_1 : {dnet_1.shape}")
        dnet_2 = self.relu(self.pointwise2(dnet_1))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_2 : {dnet_2.shape}")
        dnet_out = self.relu(self.pointwise3(dnet_2))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_out : {dnet_out.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] dnet_out: {dnet_out.min()} ~ {dnet_out.max()}")
        
        # GlobalNet 
        #gnet_dconv_1 = self.depthwise1(x)
        #if DEBUG_FLAG == 1: print(f"[INFO] gnet_dconv_1 : {gnet_dconv_1.shape}")
        #gnet_dconv_2 = self.depthwise2(gnet_dconv_1)
        #if DEBUG_FLAG == 1: print(f"[INFO] gnet_dconv_2 : {gnet_dconv_2.shape}")
        #gnet_dconv_3 = self.separable_conv(gnet_dconv_2)
        #if DEBUG_FLAG == 1: print(f"[INFO] gnet_dconv_3 : {gnet_dconv_3.shape}")
        #if DEBUG_FLAG == 2: print(f"[INFO] gnet_dconv_3: {gnet_dconv_3.min()} ~ {gnet_dconv_3.max()}")

        ## Upsampling
        #gnet_up_1 = self.upsample1(gnet_dconv_3)
        #if DEBUG_FLAG == 1: print(f"[INFO] gnet_up_1 : {gnet_up_1.shape}")
        #gnet_up_2 = self.upsample2(gnet_up_1)
        #if DEBUG_FLAG == 1: print(f"[INFO] gnet_up_2 : {gnet_up_2.shape}")
        #gnet_out = self.upsample3(gnet_up_2)
        #if DEBUG_FLAG == 1: print(f"[INFO] gnet_up_3 : {gnet_out.shape}")
        #if DEBUG_FLAG == 2: print(f"[INFO] gnet_out: {gnet_out.min()} ~ {gnet_out.max()}")
        
        # Element-wise Addition
        out = dnet_out # + gnet_out
        if DEBUG_FLAG == 2: print(f"[INFO] add_out: {out.min()} ~ {out.max()}")
        out = torch.sigmoid(out)
        #out = self.tanh(out)
        if DEBUG_FLAG == 2: print(f"[INFO] sigmoid_out: {out.min()} ~ {out.max()}")
        return out

class LIGHTFUSE_global_only(nn.Module):
    def __init__(self, args):
        super(LIGHTFUSE_global_only, self).__init__()
        nChannel = args.nChannel
        nFeat = args.nFeat
        self.args = args
        print(f"[INFO] nChannel : {nChannel}, nFeat : {nFeat}")
        
        # DetailNet
        #self.pointwise1 = nn.Conv2d(nChannel, nFeat, kernel_size=1, stride=1, padding=0, bias=True)
        #self.pointwise2 = nn.Conv2d(nFeat, nFeat, kernel_size=1, stride=1, padding=0, bias=True)
        #self.pointwise3 = nn.Conv2d(nFeat, 3, kernel_size=1, stride=1, padding=0, bias=True)

        # GlobalNet : 
        # Depthwise & Separable Convolution Layers
        self.depthwise1 = nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel)
        self.depthwise2 = nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel)
        self.separable_conv = nn.Sequential(
          nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel),
          nn.Conv2d(nChannel,        3, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # Upsampling
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.upsample3 = nn.Upsample(scale_factor=2)

        # Final Output
        self.relu = nn.ReLU()

    def forward(self, x1, x2, x3):
        DEBUG_FLAG = 0
        
        if DEBUG_FLAG == 2: print(f"[INFO] x1: {x1.min():.6f} ~ {x1.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x2: {x2.min():.6f} ~ {x2.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x3: {x3.min():.6f} ~ {x3.max():.6f}")
        
        x = torch.cat((x1, x3), dim=1)
        
        ## DetailNet
        #if DEBUG_FLAG == 1: print(f"[INFO] x : {x.shape}")
        #dnet_1 = self.relu(self.pointwise1(x))
        #if DEBUG_FLAG == 1: print(f"[INFO] dnet_1 : {dnet_1.shape}")
        #dnet_2 = self.relu(self.pointwise2(dnet_1))
        #if DEBUG_FLAG == 1: print(f"[INFO] dnet_2 : {dnet_2.shape}")
        #dnet_out = self.relu(self.pointwise3(dnet_2))
        #if DEBUG_FLAG == 1: print(f"[INFO] dnet_out : {dnet_out.shape}")
        #if DEBUG_FLAG == 2: print(f"[INFO] dnet_out: {dnet_out.min()} ~ {dnet_out.max()}")
        
        # GlobalNet 
        gnet_dconv_1 = self.depthwise1(x)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_dconv_1 : {gnet_dconv_1.shape}")
        gnet_dconv_2 = self.depthwise2(gnet_dconv_1)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_dconv_2 : {gnet_dconv_2.shape}")
        gnet_dconv_3 = self.separable_conv(gnet_dconv_2)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_dconv_3 : {gnet_dconv_3.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] gnet_dconv_3: {gnet_dconv_3.min()} ~ {gnet_dconv_3.max()}")

        # Upsampling
        gnet_up_1 = self.upsample1(gnet_dconv_3)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_up_1 : {gnet_up_1.shape}")
        gnet_up_2 = self.upsample2(gnet_up_1)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_up_2 : {gnet_up_2.shape}")
        gnet_out = self.upsample3(gnet_up_2)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_up_3 : {gnet_out.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] gnet_out: {gnet_out.min()} ~ {gnet_out.max()}")
        
        # Element-wise Addition
        out = gnet_out # + gnet_out
        if DEBUG_FLAG == 2: print(f"[INFO] add_out: {out.min()} ~ {out.max()}")
        out = torch.sigmoid(out)
        #out = self.tanh(out)
        if DEBUG_FLAG == 2: print(f"[INFO] sigmoid_out: {out.min()} ~ {out.max()}")
        return out

class LIGHTFUSE_6ch_upscale(nn.Module):
    def __init__(self, args):
        super(LIGHTFUSE_6ch_upscale, self).__init__()
        nChannel = args.nChannel
        nFeat = args.nFeat
        self.args = args
        print(f"[INFO] nChannel : {nChannel}, nFeat : {nFeat}")
        
        # DetailNet
        self.pointwise1 = nn.Conv2d(nChannel, nFeat, kernel_size=1, stride=1, padding=0, bias=True)
        self.pointwise2 = nn.Conv2d(nFeat, nFeat, kernel_size=1, stride=1, padding=0, bias=True)
        self.pointwise3 = nn.Conv2d(nFeat, 3, kernel_size=1, stride=1, padding=0, bias=True)

        # GlobalNet : 
        # Depthwise & Separable Convolution Layers
        self.depthwise1 = nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel)
        self.depthwise2 = nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel)
        self.separable_conv = nn.Sequential(
          nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel),
          #nn.Conv2d(nChannel,        3, kernel_size=1, stride=1, padding=0, bias=True)
          nn.Conv2d(nChannel, nChannel, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # Upsampling (ConvTranspose2d)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.upsample3 = nn.Upsample(scale_factor=2)
        self.chmerge = nn.Conv2d(nChannel, 3, kernel_size=1, stride=1, padding=0, bias=True)

        # Final Output
        self.relu = nn.ReLU()

    def forward(self, x1, x2, x3):
        DEBUG_FLAG = 0
        
        if DEBUG_FLAG == 2: print(f"[INFO] x1: {x1.min():.6f} ~ {x1.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x2: {x2.min():.6f} ~ {x2.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x3: {x3.min():.6f} ~ {x3.max():.6f}")
        
        x = torch.cat((x1, x3), dim=1)
        
        # DetailNet
        if DEBUG_FLAG == 1: print(f"[INFO] x : {x.shape}")
        dnet_1 = self.relu(self.pointwise1(x))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_1 : {dnet_1.shape}")
        dnet_2 = self.relu(self.pointwise2(dnet_1))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_2 : {dnet_2.shape}")
        dnet_out = self.relu(self.pointwise3(dnet_2))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_out : {dnet_out.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] dnet_out: {dnet_out.min()} ~ {dnet_out.max()}")
        
        # GlobalNet 
        gnet_dconv_1 = self.depthwise1(x)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_dconv_1 : {gnet_dconv_1.shape}")
        gnet_dconv_2 = self.depthwise2(gnet_dconv_1)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_dconv_2 : {gnet_dconv_2.shape}")
        gnet_dconv_3 = self.separable_conv(gnet_dconv_2)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_dconv_3 : {gnet_dconv_3.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] gnet_dconv_3: {gnet_dconv_3.min()} ~ {gnet_dconv_3.max()}")

        # Upsampling
        gnet_up_1 = self.upsample1(gnet_dconv_3)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_up_1 : {gnet_up_1.shape}")
        gnet_up_2 = self.upsample2(gnet_up_1)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_up_2 : {gnet_up_2.shape}")
        gnet_up_3 = self.upsample3(gnet_up_2)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_up_3 : {gnet_up_3.shape}")
        gnet_out = self.chmerge(gnet_up_3)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_out : {gnet_out.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] gnet_out: {gnet_out.min()} ~ {gnet_out.max()}")

        # Element-wise Addition
        out = dnet_out + gnet_out
        if DEBUG_FLAG == 2: print(f"[INFO] add_out: {out.min()} ~ {out.max()}")
        out = torch.sigmoid(out)
        #out = self.tanh(out)
        if DEBUG_FLAG == 2: print(f"[INFO] sigmoid_out: {out.min()} ~ {out.max()}")
        return out