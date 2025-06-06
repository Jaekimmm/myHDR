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
 
class LIGHTFUSE_sigmoid_save_inter(nn.Module):
    def __init__(self, args):
        super(LIGHTFUSE_sigmoid_save_inter, self).__init__()
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
        return out, dnet_out, gnet_out
 
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

class LIGHTFUSE_skip(nn.Module):
    def __init__(self, args):
        super(LIGHTFUSE_skip, self).__init__()
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
        out = dnet_out + gnet_out + x2  # skip mid
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

class LIGHTFUSE_detail_3x3_ref_only(nn.Module):
    def __init__(self, args):
        super(LIGHTFUSE_detail_3x3_ref_only, self).__init__()
        nChannel = args.nChannel
        nFeat = args.nFeat
        self.args = args
        print(f"[INFO] nChannel : {nChannel}, nFeat : {nFeat}")
        
        # DetailNet
        #self.pointwise1 = nn.Conv2d(nChannel, nFeat, kernel_size=1, stride=1, padding=0, bias=True)
        #self.pointwise2 = nn.Conv2d(nFeat, nFeat, kernel_size=1, stride=1, padding=0, bias=True)
        #self.pointwise3 = nn.Conv2d(nFeat, 3, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(nFeat, 3, kernel_size=3, stride=1, padding=1, bias=True)

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
        
        #x = torch.cat((x1, x3), dim=1)
        x = x2
        
        # DetailNet
        if DEBUG_FLAG == 1: print(f"[INFO] x : {x.shape}")
        dnet_1 = self.relu(self.conv1(x))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_1 : {dnet_1.shape}")
        dnet_2 = self.relu(self.conv2(dnet_1))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_2 : {dnet_2.shape}")
        dnet_out = self.relu(self.conv3(dnet_2))
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
class LIGHTFUSE_detail_3x3_ref_only_skip(nn.Module):
    def __init__(self, args):
        super(LIGHTFUSE_detail_3x3_ref_only_skip, self).__init__()
        nChannel = args.nChannel
        nFeat = args.nFeat
        self.args = args
        print(f"[INFO] nChannel : {nChannel}, nFeat : {nFeat}")
        
        # DetailNet
        #self.pointwise1 = nn.Conv2d(nChannel, nFeat, kernel_size=1, stride=1, padding=0, bias=True)
        #self.pointwise2 = nn.Conv2d(nFeat, nFeat, kernel_size=1, stride=1, padding=0, bias=True)
        #self.pointwise3 = nn.Conv2d(nFeat, 3, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(nFeat, 3, kernel_size=3, stride=1, padding=1, bias=True)

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
        
        #x = torch.cat((x1, x3), dim=1)
        x = x2
        
        # DetailNet
        if DEBUG_FLAG == 1: print(f"[INFO] x : {x.shape}")
        dnet_1 = self.relu(self.conv1(x))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_1 : {dnet_1.shape}")
        dnet_2 = self.relu(self.conv2(dnet_1))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_2 : {dnet_2.shape}")
        dnet_out = self.relu(self.conv3(dnet_2))
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
        out = dnet_out + x2 # + gnet_out
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

class LIGHTFUSE_6ch_upscale_skip(nn.Module):
    def __init__(self, args):
        super(LIGHTFUSE_6ch_upscale_skip, self).__init__()
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
        out = dnet_out + gnet_out + x2
        if DEBUG_FLAG == 2: print(f"[INFO] add_out: {out.min()} ~ {out.max()}")
        out = torch.sigmoid(out)
        #out = self.tanh(out)
        if DEBUG_FLAG == 2: print(f"[INFO] sigmoid_out: {out.min()} ~ {out.max()}")
        return out

class LIGHTFUSE_3exp_9ch_upscale(nn.Module):
    def __init__(self, args):
        super(LIGHTFUSE_3exp_9ch_upscale, self).__init__()
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
        
        x = torch.cat((x1, x2, x3), dim=1)
        
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
class LIGHTFUSE_3exp_9ch_upscale_skip(nn.Module):
    def __init__(self, args):
        super(LIGHTFUSE_3exp_9ch_upscale_skip, self).__init__()
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
        
        x = torch.cat((x1, x2, x3), dim=1)
        
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
        out = dnet_out + gnet_out + x2
        if DEBUG_FLAG == 2: print(f"[INFO] add_out: {out.min()} ~ {out.max()}")
        out = torch.sigmoid(out)
        #out = self.tanh(out)
        if DEBUG_FLAG == 2: print(f"[INFO] sigmoid_out: {out.min()} ~ {out.max()}")
        return out

class LIGHTFUSE_3exp(nn.Module):
    def __init__(self, args):
        super(LIGHTFUSE_3exp, self).__init__()
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
        
        x = torch.cat((x1, x2, x3), dim=1)
        
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

class LIGHTFUSE_3exp_save_inter(nn.Module):
    def __init__(self, args):
        super(LIGHTFUSE_3exp_save_inter, self).__init__()
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
        
        x = torch.cat((x1, x2, x3), dim=1)
        
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
        return out, dnet_out, gnet_out
 
class LIGHTFUSE_3exp_global_only(nn.Module):
    def __init__(self, args):
        super(LIGHTFUSE_3exp_global_only, self).__init__()
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
        
        x = torch.cat((x1, x2, x3), dim=1)
        
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
        out = gnet_out #dnet_out + gnet_out
        if DEBUG_FLAG == 2: print(f"[INFO] add_out: {out.min()} ~ {out.max()}")
        out = torch.sigmoid(out)
        #out = self.tanh(out)
        if DEBUG_FLAG == 2: print(f"[INFO] sigmoid_out: {out.min()} ~ {out.max()}")
        return out
 
class LIGHTFUSE_3exp_detail_only(nn.Module):
    def __init__(self, args):
        super(LIGHTFUSE_3exp_detail_only, self).__init__()
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

        ## Upsampling (ConvTranspose2d)
        #self.upsample1 = nn.Upsample(scale_factor=2)
        #self.upsample2 = nn.Upsample(scale_factor=2)
        #self.upsample3 = nn.Upsample(scale_factor=2)

        # Final Output
        self.relu = nn.ReLU()

    def forward(self, x1, x2, x3):
        DEBUG_FLAG = 0
        
        if DEBUG_FLAG == 2: print(f"[INFO] x1: {x1.min():.6f} ~ {x1.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x2: {x2.min():.6f} ~ {x2.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x3: {x3.min():.6f} ~ {x3.max():.6f}")
        
        x = torch.cat((x1, x2, x3), dim=1)
        
        # DetailNet
        if DEBUG_FLAG == 1: print(f"[INFO] x : {x.shape}")
        dnet_1 = self.relu(self.pointwise1(x))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_1 : {dnet_1.shape}")
        dnet_2 = self.relu(self.pointwise2(dnet_1))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_2 : {dnet_2.shape}")
        dnet_out = self.relu(self.pointwise3(dnet_2))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_out : {dnet_out.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] dnet_out: {dnet_out.min()} ~ {dnet_out.max()}")
        
        ## GlobalNet 
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
        out = dnet_out #+ gnet_out
        if DEBUG_FLAG == 2: print(f"[INFO] add_out: {out.min()} ~ {out.max()}")
        out = torch.sigmoid(out)
        #out = self.tanh(out)
        if DEBUG_FLAG == 2: print(f"[INFO] sigmoid_out: {out.min()} ~ {out.max()}")
        return out
 
class LIGHTFUSE_3exp_detail_ref_only(nn.Module):
    def __init__(self, args):
        super(LIGHTFUSE_3exp_detail_ref_only, self).__init__()
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

        ## Upsampling (ConvTranspose2d)
        #self.upsample1 = nn.Upsample(scale_factor=2)
        #self.upsample2 = nn.Upsample(scale_factor=2)
        #self.upsample3 = nn.Upsample(scale_factor=2)

        # Final Output
        self.relu = nn.ReLU()

    def forward(self, x1, x2, x3):
        DEBUG_FLAG = 0
        
        if DEBUG_FLAG == 2: print(f"[INFO] x1: {x1.min():.6f} ~ {x1.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x2: {x2.min():.6f} ~ {x2.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x3: {x3.min():.6f} ~ {x3.max():.6f}")
        
        #x = torch.cat((x1, x2, x3), dim=1)
        x = x2
        
        # DetailNet
        if DEBUG_FLAG == 1: print(f"[INFO] x : {x.shape}")
        dnet_1 = self.relu(self.pointwise1(x))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_1 : {dnet_1.shape}")
        dnet_2 = self.relu(self.pointwise2(dnet_1))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_2 : {dnet_2.shape}")
        dnet_out = self.relu(self.pointwise3(dnet_2))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_out : {dnet_out.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] dnet_out: {dnet_out.min()} ~ {dnet_out.max()}")
        
        ## GlobalNet 
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
        out = dnet_out #+ gnet_out
        if DEBUG_FLAG == 2: print(f"[INFO] add_out: {out.min()} ~ {out.max()}")
        out = torch.sigmoid(out)
        #out = self.tanh(out)
        if DEBUG_FLAG == 2: print(f"[INFO] sigmoid_out: {out.min()} ~ {out.max()}")
        return out
 
class LIGHTFUSE_3exp_detail_ref_only_ccm(nn.Module):
    def __init__(self, args):
        super(LIGHTFUSE_3exp_detail_ref_only_ccm, self).__init__()
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

        ## Upsampling (ConvTranspose2d)
        #self.upsample1 = nn.Upsample(scale_factor=2)
        #self.upsample2 = nn.Upsample(scale_factor=2)
        #self.upsample3 = nn.Upsample(scale_factor=2)

        # Final Output
        self.relu = nn.ReLU()

    def forward(self, x1, x2, x3):
        DEBUG_FLAG = 0
        
        if DEBUG_FLAG == 2: print(f"[INFO] x1: {x1.min():.6f} ~ {x1.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x2: {x2.min():.6f} ~ {x2.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x3: {x3.min():.6f} ~ {x3.max():.6f}")
        
        #x = torch.cat((x1, x2, x3), dim=1)
        x = x2
        
        # DetailNet
        if DEBUG_FLAG == 1: print(f"[INFO] x : {x.shape}")
        dnet_1 = self.relu(self.pointwise1(x))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_1 : {dnet_1.shape}")
        dnet_2 = self.relu(self.pointwise2(dnet_1))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_2 : {dnet_2.shape}")
        dnet_out = self.relu(self.pointwise3(dnet_2))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_out : {dnet_out.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] dnet_out: {dnet_out.min()} ~ {dnet_out.max()}")
        
        ## GlobalNet 
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
        out = dnet_out #+ gnet_out
        if DEBUG_FLAG == 2: print(f"[INFO] add_out: {out.min()} ~ {out.max()}")
        out = torch.sigmoid(out)
        #out = self.tanh(out)
        if DEBUG_FLAG == 2: print(f"[INFO] sigmoid_out: {out.min()} ~ {out.max()}")
        return out
 
class LIGHTFUSE_3exp_detail_ref_only_skip_add_last(nn.Module):
    def __init__(self, args):
        super(LIGHTFUSE_3exp_detail_ref_only_skip_add_last, self).__init__()
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

        ## Upsampling (ConvTranspose2d)
        #self.upsample1 = nn.Upsample(scale_factor=2)
        #self.upsample2 = nn.Upsample(scale_factor=2)
        #self.upsample3 = nn.Upsample(scale_factor=2)
        
        self.depthwise_out = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1) #, groups=3)

        # Final Output
        self.relu = nn.ReLU()

    def forward(self, x1, x2, x3):
        DEBUG_FLAG = 0
        
        if DEBUG_FLAG == 2: print(f"[INFO] x1: {x1.min():.6f} ~ {x1.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x2: {x2.min():.6f} ~ {x2.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x3: {x3.min():.6f} ~ {x3.max():.6f}")
        
        #x = torch.cat((x1, x2, x3), dim=1)
        x = x2
        
        # DetailNet
        if DEBUG_FLAG == 1: print(f"[INFO] x : {x.shape}")
        dnet_1 = self.relu(self.pointwise1(x))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_1 : {dnet_1.shape}")
        dnet_2 = self.relu(self.pointwise2(dnet_1))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_2 : {dnet_2.shape}")
        dnet_out = self.relu(self.pointwise3(dnet_2))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_out : {dnet_out.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] dnet_out: {dnet_out.min()} ~ {dnet_out.max()}")
        
        ## GlobalNet 
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
        out = dnet_out + x2 #+ gnet_out
        if DEBUG_FLAG == 2: print(f"[INFO] add_out: {out.min()} ~ {out.max()}")
        out = self.depthwise_out(out)
        out = torch.sigmoid(out)
        #out = self.tanh(out)
        if DEBUG_FLAG == 2: print(f"[INFO] sigmoid_out: {out.min()} ~ {out.max()}")
        return out
 
class LIGHTFUSE_3exp_skip(nn.Module):
    def __init__(self, args):
        super(LIGHTFUSE_3exp_skip, self).__init__()
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
        
        x = torch.cat((x1, x2, x3), dim=1)
        
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
        out = dnet_out + gnet_out + x2
        if DEBUG_FLAG == 2: print(f"[INFO] add_out: {out.min()} ~ {out.max()}")
        out = torch.sigmoid(out)
        #out = self.tanh(out)
        if DEBUG_FLAG == 2: print(f"[INFO] sigmoid_out: {out.min()} ~ {out.max()}")
        return out
class LIGHTFUSE_3exp_skip_save_inter(nn.Module):
    def __init__(self, args):
        super(LIGHTFUSE_3exp_skip_save_inter, self).__init__()
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
        
        x = torch.cat((x1, x2, x3), dim=1)
        
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
        out = dnet_out + gnet_out + x2
        if DEBUG_FLAG == 2: print(f"[INFO] add_out: {out.min()} ~ {out.max()}")
        out = torch.sigmoid(out)
        #out = self.tanh(out)
        if DEBUG_FLAG == 2: print(f"[INFO] sigmoid_out: {out.min()} ~ {out.max()}")
        return out, dnet_out, gnet_out

class LIGHTFUSE_3exp_skip_add_last(nn.Module):
    def __init__(self, args):
        super(LIGHTFUSE_3exp_skip_add_last, self).__init__()
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

        self.depthwise_out = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3)
        
        # Final Output
        self.relu = nn.ReLU()

    def forward(self, x1, x2, x3):
        DEBUG_FLAG = 0
        
        if DEBUG_FLAG == 2: print(f"[INFO] x1: {x1.min():.6f} ~ {x1.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x2: {x2.min():.6f} ~ {x2.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x3: {x3.min():.6f} ~ {x3.max():.6f}")
        
        x = torch.cat((x1, x2, x3), dim=1)
        
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
        out = dnet_out + gnet_out + x2
        if DEBUG_FLAG == 2: print(f"[INFO] add_out: {out.min()} ~ {out.max()}")
        out = self.depthwise_out(out)
        out = torch.sigmoid(out)
        #out = self.tanh(out)
        if DEBUG_FLAG == 2: print(f"[INFO] sigmoid_out: {out.min()} ~ {out.max()}")
        return out

class LIGHTFUSE_3exp_skip_add_last_conv(nn.Module):
    def __init__(self, args):
        super(LIGHTFUSE_3exp_skip_add_last_conv, self).__init__()
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

        self.conv_out = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        
        # Final Output
        self.relu = nn.ReLU()

    def forward(self, x1, x2, x3):
        DEBUG_FLAG = 0
        
        if DEBUG_FLAG == 2: print(f"[INFO] x1: {x1.min():.6f} ~ {x1.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x2: {x2.min():.6f} ~ {x2.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x3: {x3.min():.6f} ~ {x3.max():.6f}")
        
        x = torch.cat((x1, x2, x3), dim=1)
        
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
        out = dnet_out + gnet_out + x2
        if DEBUG_FLAG == 2: print(f"[INFO] add_out: {out.min()} ~ {out.max()}")
        out = self.conv_out(out)
        out = torch.sigmoid(out)
        #out = self.tanh(out)
        if DEBUG_FLAG == 2: print(f"[INFO] sigmoid_out: {out.min()} ~ {out.max()}")
        return out

class LIGHTFUSE_3exp_add_last(nn.Module):
    def __init__(self, args):
        super(LIGHTFUSE_3exp_add_last, self).__init__()
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

        self.depthwise_out = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3)
        
        # Final Output
        self.relu = nn.ReLU()

    def forward(self, x1, x2, x3):
        DEBUG_FLAG = 0
        
        if DEBUG_FLAG == 2: print(f"[INFO] x1: {x1.min():.6f} ~ {x1.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x2: {x2.min():.6f} ~ {x2.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x3: {x3.min():.6f} ~ {x3.max():.6f}")
        
        x = torch.cat((x1, x2, x3), dim=1)
        
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
        out = self.depthwise_out(out)
        out = torch.sigmoid(out)
        #out = self.tanh(out)
        if DEBUG_FLAG == 2: print(f"[INFO] sigmoid_out: {out.min()} ~ {out.max()}")
        return out
    
class LIGHTFUSE_3exp_add_last_skip(nn.Module):
    def __init__(self, args):
        super(LIGHTFUSE_3exp_add_last_skip, self).__init__()
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

        self.depthwise_out = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3)
        
        # Final Output
        self.relu = nn.ReLU()

    def forward(self, x1, x2, x3):
        DEBUG_FLAG = 0
        
        if DEBUG_FLAG == 2: print(f"[INFO] x1: {x1.min():.6f} ~ {x1.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x2: {x2.min():.6f} ~ {x2.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x3: {x3.min():.6f} ~ {x3.max():.6f}")
        
        x = torch.cat((x1, x2, x3), dim=1)
        
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
        out = self.depthwise_out(out)
        out = out + x2
        out = torch.sigmoid(out)
        #out = self.tanh(out)
        if DEBUG_FLAG == 2: print(f"[INFO] sigmoid_out: {out.min()} ~ {out.max()}")
        return out

class LIGHTFUSE_3exp_add_last_conv_skip(nn.Module):
    def __init__(self, args):
        super(LIGHTFUSE_3exp_add_last_conv_skip, self).__init__()
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

        self.conv_out = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        
        # Final Output
        self.relu = nn.ReLU()

    def forward(self, x1, x2, x3):
        DEBUG_FLAG = 0
        
        if DEBUG_FLAG == 2: print(f"[INFO] x1: {x1.min():.6f} ~ {x1.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x2: {x2.min():.6f} ~ {x2.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x3: {x3.min():.6f} ~ {x3.max():.6f}")
        
        x = torch.cat((x1, x2, x3), dim=1)
        
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
        out = self.conv_out(out)
        out = out + x2
        out = torch.sigmoid(out)
        #out = self.tanh(out)
        if DEBUG_FLAG == 2: print(f"[INFO] sigmoid_out: {out.min()} ~ {out.max()}")
        return out

class LIGHTFUSE_add_last_depthwise(nn.Module):
    def __init__(self, args):
        super(LIGHTFUSE_add_last_depthwise, self).__init__()
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

        self.depthwise_out = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3)
        
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
        out = self.depthwise_out(out)
        out = torch.sigmoid(out)
        #out = self.tanh(out)
        if DEBUG_FLAG == 2: print(f"[INFO] sigmoid_out: {out.min()} ~ {out.max()}")
        return out

class LIGHTFUSE_add_last_depthwise_skip(nn.Module):
    def __init__(self, args):
        super(LIGHTFUSE_add_last_depthwise_skip, self).__init__()
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

        self.depthwise_out = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3)
        
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
        out = dnet_out + gnet_out + x2
        if DEBUG_FLAG == 2: print(f"[INFO] add_out: {out.min()} ~ {out.max()}")
        out = self.depthwise_out(out)
        out = torch.sigmoid(out)
        #out = self.tanh(out)
        if DEBUG_FLAG == 2: print(f"[INFO] sigmoid_out: {out.min()} ~ {out.max()}")
        return out

class LIGHTFUSE_add_last_conv(nn.Module):
    def __init__(self, args):
        super(LIGHTFUSE_add_last_conv, self).__init__()
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

        self.conv_out = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        
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
        out = self.conv_out(out)
        out = torch.sigmoid(out)
        #out = self.tanh(out)
        if DEBUG_FLAG == 2: print(f"[INFO] sigmoid_out: {out.min()} ~ {out.max()}")
        return out

class DETAILNET(nn.Module):
    def __init__(self, args):
        super(DETAILNET, self).__init__()
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
        self.conv1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(nFeat)
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(nFeat)
        self.conv3 = nn.Conv2d(nFeat, nChannel, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(nChannel)
       
        self.conv_out = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        
        # Final Output
        self.relu = nn.ReLU()

    def forward(self, x1, x2, x3):
        DEBUG_FLAG = 0
        
        if DEBUG_FLAG == 2: print(f"[INFO] x1: {x1.min():.6f} ~ {x1.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x2: {x2.min():.6f} ~ {x2.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x3: {x3.min():.6f} ~ {x3.max():.6f}")
        
        x = x2
        
        # DetailNet
        if DEBUG_FLAG == 1: print(f"[INFO] x : {x.shape}")
        dnet_1 = self.relu(self.bn1(self.conv1(x)))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_1 : {dnet_1.shape}")
        dnet_2 = self.relu(self.bn2(self.conv2(dnet_1)))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_2 : {dnet_2.shape}")
        dnet_out = self.relu(self.bn3(self.conv3(dnet_2)))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_out : {dnet_out.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] dnet_out: {dnet_out.min()} ~ {dnet_out.max()}")
        
        # Element-wise Addition
        out = dnet_out + x2
        if DEBUG_FLAG == 2: print(f"[INFO] add_out: {out.min()} ~ {out.max()}")
        out = self.conv_out(out)
        out = torch.sigmoid(out)
        #out = self.tanh(out)
        if DEBUG_FLAG == 2: print(f"[INFO] sigmoid_out: {out.min()} ~ {out.max()}")
        return out

class DETAILNET_nobn(nn.Module):
    def __init__(self, args):
        super(DETAILNET_nobn, self).__init__()
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
        self.conv1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, stride=1, padding=1)
        #self.bn1 = nn.BatchNorm2d(nFeat)
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1)
        #self.bn2 = nn.BatchNorm2d(nFeat)
        self.conv3 = nn.Conv2d(nFeat, nChannel, kernel_size=3, stride=1, padding=1)
        #self.bn3 = nn.BatchNorm2d(nChannel)
       
        self.conv_out = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        
        # Final Output
        self.relu = nn.ReLU()

    def forward(self, x1, x2, x3):
        DEBUG_FLAG = 0
        
        if DEBUG_FLAG == 2: print(f"[INFO] x1: {x1.min():.6f} ~ {x1.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x2: {x2.min():.6f} ~ {x2.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x3: {x3.min():.6f} ~ {x3.max():.6f}")
        
        x = x2
        
        # DetailNet
        if DEBUG_FLAG == 1: print(f"[INFO] x : {x.shape}")
        dnet_1 = self.relu(self.conv1(x))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_1 : {dnet_1.shape}")
        dnet_2 = self.relu(self.conv2(dnet_1))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_2 : {dnet_2.shape}")
        dnet_out = self.relu(self.conv3(dnet_2))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_out : {dnet_out.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] dnet_out: {dnet_out.min()} ~ {dnet_out.max()}")
        
        # Element-wise Addition
        out = dnet_out + x2
        if DEBUG_FLAG == 2: print(f"[INFO] add_out: {out.min()} ~ {out.max()}")
        out = self.conv_out(out)
        out = torch.sigmoid(out)
        #out = self.tanh(out)
        if DEBUG_FLAG == 2: print(f"[INFO] sigmoid_out: {out.min()} ~ {out.max()}")
        return out

class PLAIN_CONV_TEST(nn.Module):
    def __init__(self, args):
        super(PLAIN_CONV_TEST, self).__init__()
        nChannel = args.nChannel
        nFeat = args.nFeat
        self.args = args
        print(f"[INFO] nChannel : {nChannel}, nFeat : {nFeat}")
        
        self.conv1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(nFeat, nChannel, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x1, x2, x3):
        x = x2
        dnet_1 = self.relu(self.conv1(x))
        dnet_2 = self.relu(self.conv2(dnet_1))
        dnet_out = self.relu(self.conv3(dnet_2))
        out = dnet_out
        out = torch.sigmoid(out)
        
        return out

class PLAIN_CONV_RES(nn.Module):
    def __init__(self, args):
        super(PLAIN_CONV_RES, self).__init__()
        nChannel = args.nChannel
        nFeat = args.nFeat
        self.args = args
        print(f"[INFO] nChannel : {nChannel}, nFeat : {nFeat}")
        
        self.conv1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(nFeat, nChannel, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x1, x2, x3):
        x = x2
        dnet_1 = self.relu(self.conv1(x))
        dnet_2 = self.relu(self.conv2(dnet_1))
        dnet_out = self.relu(self.conv3(dnet_2))
        out = dnet_out + x2
        out = torch.sigmoid(out)
        
        return out