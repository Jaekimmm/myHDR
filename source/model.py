import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from torch.autograd import Variable

class myHDR_new2_conv(nn.Module):
    def __init__(self, args):
        super(myHDR_new2_conv, self).__init__()
        nChannel = args.nChannel
        nFeat = args.nFeat
        self.args = args
        print(f"[INFO] nChannel : {nChannel}, nFeat : {nFeat}")
        
        # DetailNet
        self.pointwise1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, stride=1, padding=1, bias=True)
        self.pointwise2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1, bias=True)
        self.pointwise3 = nn.Conv2d(nFeat, 3, kernel_size=3, stride=1, padding=1, bias=True)

        # GlobalNet : 
        # Depthwise & Separable Convolution Layers
        self.gconv1 = nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1)
        self.gconv2 = nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1)
        self.gconv3 = nn.Conv2d(nChannel, 3, kernel_size=3, stride=2, padding=1)

        # Upsampling (ConvTranspose2d)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.upsample3 = nn.Upsample(scale_factor=2)

        self.depthwise_out = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3)
        
        # Final Output
        self.relu = nn.ReLU()

    def forward(self, x1, x2, x3):
        x = torch.cat((x1, x2, x3), dim=1)
        
        # DetailNet
        dnet_1 = self.relu(self.pointwise1(x))
        dnet_2 = self.relu(self.pointwise2(dnet_1))
        dnet_out = self.pointwise3(dnet_2)
        
        # GlobalNet 
        gnet_dconv_1 = self.relu(self.gconv1(x))
        gnet_dconv_2 = self.relu(self.gconv2(gnet_dconv_1))
        gnet_dconv_3 = self.gconv3(gnet_dconv_2)

        # Upsampling
        gnet_up_1 = self.upsample1(gnet_dconv_3)
        gnet_up_2 = self.upsample2(gnet_up_1)
        gnet_out = self.upsample3(gnet_up_2)

        # Element-wise Addition
        out = dnet_out + gnet_out
        out = self.depthwise_out(out)
        out = out + x2
        out = torch.sigmoid(out)
        return out
    
class myHDR_new_conv(nn.Module):
    def __init__(self, args):
        super(myHDR_new_conv, self).__init__()
        nChannel = args.nChannel
        nFeat = args.nFeat
        self.args = args
        print(f"[INFO] nChannel : {nChannel}, nFeat : {nFeat}")
        
        # DetailNet
        self.pointwise1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, stride=1, padding=1, bias=True)
        self.pointwise2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1, bias=True)
        self.pointwise3 = nn.Conv2d(nFeat, 3, kernel_size=3, stride=1, padding=1, bias=True)

        # GlobalNet : 
        # Depthwise & Separable Convolution Layers
        self.gconv1 = nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1)
        self.gconv2 = nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1)
        self.gconv3 = nn.Conv2d(nChannel, 3, kernel_size=3, stride=2, padding=1)

        # Upsampling (ConvTranspose2d)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.upsample3 = nn.Upsample(scale_factor=2)

        self.depthwise_out = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3)
        
        # Final Output
        self.relu = nn.ReLU()

    def forward(self, x1, x2, x3):
        x = torch.cat((x1, x2, x3), dim=1)
        
        # DetailNet
        dnet_1 = self.relu(self.pointwise1(x))
        dnet_2 = self.relu(self.pointwise2(dnet_1))
        dnet_out = self.pointwise3(dnet_2)
        
        # GlobalNet 
        gnet_dconv_1 = self.gconv1(x)
        gnet_dconv_2 = self.gconv2(gnet_dconv_1)
        gnet_dconv_3 = self.gconv3(gnet_dconv_2)

        # Upsampling
        gnet_up_1 = self.upsample1(gnet_dconv_3)
        gnet_up_2 = self.upsample2(gnet_up_1)
        gnet_out = self.upsample3(gnet_up_2)

        # Element-wise Addition
        out = dnet_out + gnet_out
        out = self.depthwise_out(out)
        out = out + x2
        out = torch.sigmoid(out)
        return out

class myHDR_new2(nn.Module):
    def __init__(self, args):
        super(myHDR_new2, self).__init__()
        nChannel = args.nChannel
        nFeat = args.nFeat
        self.args = args
        print(f"[INFO] nChannel : {nChannel}, nFeat : {nFeat}")
        
        # DetailNet
        self.pointwise1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, stride=1, padding=1, bias=True)
        self.pointwise2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1, bias=True)
        self.pointwise3 = nn.Conv2d(nFeat, 3, kernel_size=3, stride=1, padding=1, bias=True)

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
        x = torch.cat((x1, x2, x3), dim=1)
        
        # DetailNet
        dnet_1 = self.relu(self.pointwise1(x))
        dnet_2 = self.relu(self.pointwise2(dnet_1))
        dnet_out = self.pointwise3(dnet_2)
        
        # GlobalNet 
        gnet_dconv_1 = self.relu(self.depthwise1(x))
        gnet_dconv_2 = self.relu(self.depthwise2(gnet_dconv_1))
        gnet_dconv_3 = self.relu(self.separable_conv(gnet_dconv_2))

        # Upsampling
        gnet_up_1 = self.upsample1(gnet_dconv_3)
        gnet_up_2 = self.upsample2(gnet_up_1)
        gnet_out = self.upsample3(gnet_up_2)

        # Element-wise Addition
        out = dnet_out + gnet_out
        out = self.depthwise_out(out)
        out = out + x2
        out = torch.sigmoid(out)
        return out
    
class myHDR_new(nn.Module):
    def __init__(self, args):
        super(myHDR_new, self).__init__()
        nChannel = args.nChannel
        nFeat = args.nFeat
        self.args = args
        print(f"[INFO] nChannel : {nChannel}, nFeat : {nFeat}")
        
        # DetailNet
        self.pointwise1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, stride=1, padding=1, bias=True)
        self.pointwise2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1, bias=True)
        self.pointwise3 = nn.Conv2d(nFeat, 3, kernel_size=3, stride=1, padding=1, bias=True)

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
        x = torch.cat((x1, x2, x3), dim=1)
        
        # DetailNet
        dnet_1 = self.relu(self.pointwise1(x))
        dnet_2 = self.relu(self.pointwise2(dnet_1))
        dnet_out = self.pointwise3(dnet_2)
        
        # GlobalNet 
        gnet_dconv_1 = self.depthwise1(x)
        gnet_dconv_2 = self.depthwise2(gnet_dconv_1)
        gnet_dconv_3 = self.separable_conv(gnet_dconv_2)

        # Upsampling
        gnet_up_1 = self.upsample1(gnet_dconv_3)
        gnet_up_2 = self.upsample2(gnet_up_1)
        gnet_out = self.upsample3(gnet_up_2)

        # Element-wise Addition
        out = dnet_out + gnet_out
        out = self.depthwise_out(out)
        out = out + x2
        out = torch.sigmoid(out)
        return out

class myHDR_detail_only_dconv333(nn.Module):
    def __init__(self, args):
        super(myHDR_detail_only_dconv333, self).__init__()
        nChannel = args.nChannel
        nFeat = args.nFeat
        self.args = args
        print(f"[INFO] nChannel : {nChannel}, nFeat : {nFeat}")
        
        # DetailNet
        self.pointwise1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, stride=1, padding=1, bias=True, groups=3)
        self.pointwise2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1, bias=True, groups=3)
        self.pointwise3 = nn.Conv2d(nFeat, 3, kernel_size=3, stride=1, padding=1, bias=True, groups=3)
        self.depthwise_out = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3)
        
        # Final Output
        self.relu = nn.ReLU()

    def forward(self, x1, x2, x3):
        x = x2
        
        # DetailNet
        dnet_1 = self.relu(self.pointwise1(x))
        dnet_2 = self.relu(self.pointwise2(dnet_1))
        dnet_out = self.pointwise3(dnet_2)

        # Element-wise Addition
        out = dnet_out
        out = self.depthwise_out(out)
        out = out + x2
        out = torch.sigmoid(out)
        return out

class myHDR_detail_only_conv333_fixed(nn.Module):
    def __init__(self):
        super(myHDR_detail_only_conv333_fixed, self).__init__()
        nChannel = 3
        nFeat = 16
        print(f"[INFO] nChannel : {nChannel}, nFeat : {nFeat}")
        
        # DetailNet
        self.pointwise1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, stride=1, padding=1, bias=True)
        self.pointwise2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1, bias=True)
        self.pointwise3 = nn.Conv2d(nFeat, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.depthwise_out = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3)
        
        # Final Output
        self.relu = nn.ReLU()

    def forward(self, x1, x2, x3):
        x = x2
        
        # DetailNet
        dnet_1 = self.relu(self.pointwise1(x))
        dnet_2 = self.relu(self.pointwise2(dnet_1))
        dnet_out = self.pointwise3(dnet_2)

        # Element-wise Addition
        out = dnet_out
        #out = self.depthwise_out(out)
        out = out + x2
        out = torch.sigmoid(out)
        return out

class myHDR_detail_only_conv333_old(nn.Module):
    def __init__(self, args):
        super(myHDR_detail_only_conv333_old, self).__init__()
        nChannel = args.nChannel
        nFeat = args.nFeat
        self.args = args
        print(f"[INFO] nChannel : {nChannel}, nFeat : {nFeat}")
        
        # DetailNet
        self.pointwise1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, stride=1, padding=1, bias=True)
        self.pointwise2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1, bias=True)
        self.pointwise3 = nn.Conv2d(nFeat, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.depthwise_out = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3)
        
        # Final Output
        self.relu = nn.ReLU()

    def forward(self, x1, x2, x3):
        x = x2
        
        # DetailNet
        dnet_1 = self.relu(self.pointwise1(x))
        dnet_2 = self.relu(self.pointwise2(dnet_1))
        dnet_out = self.relu(self.pointwise3(dnet_2))

        # Element-wise Addition
        out = dnet_out
        out = out + 2
        out = torch.sigmoid(out)
        return out

class myHDR_detail_only_conv333_new(nn.Module):
    def __init__(self, args):
        super(myHDR_detail_only_conv333_new, self).__init__()
        nChannel = args.nChannel
        nFeat = args.nFeat
        self.args = args
        print(f"[INFO] nChannel : {nChannel}, nFeat : {nFeat}")
        
        # DetailNet
        self.pointwise1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, stride=1, padding=1, bias=True)
        self.pointwise2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1, bias=True)
        self.pointwise3 = nn.Conv2d(nFeat, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.depthwise_out = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3)
        
        # Final Output
        self.relu = nn.ReLU()

    def forward(self, x1, x2, x3):
        x = x2
        
        # DetailNet
        dnet_1 = self.relu(self.pointwise1(x))
        dnet_2 = self.relu(self.pointwise2(dnet_1))
        dnet_out = self.pointwise3(dnet_2)

        # Element-wise Addition
        out = dnet_out
        out = out + x2
        out = torch.sigmoid(out)
        return out

class myHDR_detail_only_conv333(nn.Module):
    def __init__(self, args):
        super(myHDR_detail_only_conv333, self).__init__()
        nChannel = args.nChannel
        nFeat = args.nFeat
        self.args = args
        print(f"[INFO] nChannel : {nChannel}, nFeat : {nFeat}")
        
        # DetailNet
        self.pointwise1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, stride=1, padding=1, bias=True)
        self.pointwise2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1, bias=True)
        self.pointwise3 = nn.Conv2d(nFeat, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.depthwise_out = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3)
        
        # Final Output
        self.relu = nn.ReLU()

    def forward(self, x1, x2, x3):
        x = x2
        
        # DetailNet
        dnet_1 = self.relu(self.pointwise1(x))
        dnet_2 = self.relu(self.pointwise2(dnet_1))
        dnet_out = self.pointwise3(dnet_2)

        # Element-wise Addition
        out = dnet_out
        out = self.depthwise_out(out)
        out = out + x2
        out = torch.sigmoid(out)
        return out

class myHDR_detail_only(nn.Module):
    def __init__(self, args):
        super(myHDR_detail_only, self).__init__()
        nChannel = args.nChannel
        nFeat = args.nFeat
        self.args = args
        print(f"[INFO] nChannel : {nChannel}, nFeat : {nFeat}")
        
        # DetailNet
        self.pointwise1 = nn.Conv2d(nChannel, nFeat, kernel_size=1, stride=1, padding=0, bias=True)
        self.pointwise2 = nn.Conv2d(nFeat, nFeat, kernel_size=1, stride=1, padding=0, bias=True)
        self.pointwise3 = nn.Conv2d(nFeat, 3, kernel_size=1, stride=1, padding=0, bias=True)
        self.depthwise_out = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3)
        
        # Final Output
        self.relu = nn.ReLU()

    def forward(self, x1, x2, x3):
        x = x2
        
        # DetailNet
        dnet_1 = self.relu(self.pointwise1(x))
        dnet_2 = self.relu(self.pointwise2(dnet_1))
        dnet_out = self.pointwise3(dnet_2)

        # Element-wise Addition
        out = dnet_out
        out = self.depthwise_out(out)
        out = out + x2
        out = torch.sigmoid(out)
        return out

class myHDR_global_only_addrelu(nn.Module):
    def __init__(self, args):
        super(myHDR_global_only_addrelu, self).__init__()
        nChannel = args.nChannel
        nFeat = args.nFeat
        self.args = args
        print(f"[INFO] nChannel : {nChannel}, nFeat : {nFeat}")

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
        x = torch.cat((x1, x2, x3), dim=1)
        
        # GlobalNet 
        gnet_dconv_1 = self.relu(self.depthwise1(x))
        gnet_dconv_2 = self.relu(self.depthwise2(gnet_dconv_1))
        gnet_dconv_3 = self.relu(self.separable_conv(gnet_dconv_2))

        # Upsampling
        gnet_up_1 = self.upsample1(gnet_dconv_3)
        gnet_up_2 = self.upsample2(gnet_up_1)
        gnet_out = self.relu(self.upsample3(gnet_up_2))

        # Element-wise Addition
        out = gnet_out
        out = self.depthwise_out(out)
        out = out + x2
        out = torch.sigmoid(out)
        return out

class myHDR_global_only(nn.Module):
    def __init__(self, args):
        super(myHDR_global_only, self).__init__()
        nChannel = args.nChannel
        nFeat = args.nFeat
        self.args = args
        print(f"[INFO] nChannel : {nChannel}, nFeat : {nFeat}")

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
        x = torch.cat((x1, x2, x3), dim=1)
        
        # GlobalNet 
        gnet_dconv_1 = self.depthwise1(x)
        gnet_dconv_2 = self.depthwise2(gnet_dconv_1)
        gnet_dconv_3 = self.separable_conv(gnet_dconv_2)

        # Upsampling
        gnet_up_1 = self.upsample1(gnet_dconv_3)
        gnet_up_2 = self.upsample2(gnet_up_1)
        gnet_out = self.upsample3(gnet_up_2)

        # Element-wise Addition
        out = gnet_out
        out = self.depthwise_out(out)
        out = out + x2
        out = torch.sigmoid(out)
        return out


class myHDR_relu(nn.Module):
    def __init__(self, args):
        super(myHDR_relu, self).__init__()
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
        x = torch.cat((x1, x2, x3), dim=1)
        
        # DetailNet
        dnet_1 = self.relu(self.pointwise1(x))
        dnet_2 = self.relu(self.pointwise2(dnet_1))
        dnet_out = self.pointwise3(dnet_2)
        
        # GlobalNet 
        gnet_dconv_1 = self.relu(self.depthwise1(x))
        gnet_dconv_2 = self.depthwise2(gnet_dconv_1)
        gnet_dconv_3 = self.separable_conv(gnet_dconv_2)

        # Upsampling
        gnet_up_1 = self.upsample1(gnet_dconv_3)
        gnet_up_2 = self.upsample2(gnet_up_1)
        gnet_out = self.upsample3(gnet_up_2)

        # Element-wise Addition
        out = dnet_out + gnet_out
        out = self.depthwise_out(out)
        out = out + x2
        out = torch.sigmoid(out)
        return out

class myHDR_rmlastrelu(nn.Module):
    def __init__(self, args):
        super(myHDR_rmlastrelu, self).__init__()
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
        x = torch.cat((x1, x2, x3), dim=1)
        
        # DetailNet
        dnet_1 = self.relu(self.pointwise1(x))
        dnet_2 = self.relu(self.pointwise2(dnet_1))
        dnet_out = self.pointwise3(dnet_2)
        
        # GlobalNet 
        gnet_dconv_1 = self.depthwise1(x)
        gnet_dconv_2 = self.depthwise2(gnet_dconv_1)
        gnet_dconv_3 = self.separable_conv(gnet_dconv_2)

        # Upsampling
        gnet_up_1 = self.upsample1(gnet_dconv_3)
        gnet_up_2 = self.upsample2(gnet_up_1)
        gnet_out = self.upsample3(gnet_up_2)

        # Element-wise Addition
        out = dnet_out + gnet_out
        out = self.depthwise_out(out)
        out = out + x2
        out = torch.sigmoid(out)
        return out

class myHDR(nn.Module):
    def __init__(self, args):
        super(myHDR, self).__init__()
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
        x = torch.cat((x1, x2, x3), dim=1)
        
        # DetailNet
        dnet_1 = self.relu(self.pointwise1(x))
        dnet_2 = self.relu(self.pointwise2(dnet_1))
        dnet_out = self.relu(self.pointwise3(dnet_2))
        
        # GlobalNet 
        gnet_dconv_1 = self.depthwise1(x)
        gnet_dconv_2 = self.depthwise2(gnet_dconv_1)
        gnet_dconv_3 = self.separable_conv(gnet_dconv_2)

        # Upsampling
        gnet_up_1 = self.upsample1(gnet_dconv_3)
        gnet_up_2 = self.upsample2(gnet_up_1)
        gnet_out = self.upsample3(gnet_up_2)

        # Element-wise Addition
        out = dnet_out + gnet_out
        out = self.depthwise_out(out)
        out = out + x2
        out = torch.sigmoid(out)
        return out

class LF_3exp_skip_tanh_rmlastrelu(nn.Module):
    def __init__(self, args):
        super(LF_3exp_skip_tanh_rmlastrelu, self).__init__()
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
          nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel, bias=False),
          nn.Conv2d(nChannel,        3, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # Upsampling
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.upsample3 = nn.Upsample(scale_factor=2)

        # Final Output
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x1, x2, x3):
        x = torch.cat((x1, x2, x3), dim=1)
        
        # DetailNet
        dnet_1 = self.relu(self.pointwise1(x))
        dnet_2 = self.relu(self.pointwise2(dnet_1))
        dnet_out = self.pointwise3(dnet_2)
        
        # GlobalNet 
        gnet_dconv_1 = self.depthwise1(x)
        gnet_dconv_2 = self.depthwise2(gnet_dconv_1)
        gnet_dconv_3 = self.separable_conv(gnet_dconv_2)

        # Upsampling
        gnet_up_1 = self.upsample1(gnet_dconv_3)
        gnet_up_2 = self.upsample2(gnet_up_1)
        gnet_out = self.upsample3(gnet_up_2)

        # Element-wise Addition
        out = dnet_out + gnet_out + x2
        out = self.tanh(out)
        return out
        
class LF_3exp_skip_tanh(nn.Module):
    def __init__(self, args):
        super(LF_3exp_skip_tanh, self).__init__()
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
          nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel, bias=False),
          nn.Conv2d(nChannel,        3, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # Upsampling
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.upsample3 = nn.Upsample(scale_factor=2)

        # Final Output
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x1, x2, x3):
        x = torch.cat((x1, x2, x3), dim=1)
        
        # DetailNet
        dnet_1 = self.relu(self.pointwise1(x))
        dnet_2 = self.relu(self.pointwise2(dnet_1))
        dnet_out = self.relu(self.pointwise3(dnet_2))
        
        # GlobalNet 
        gnet_dconv_1 = self.depthwise1(x)
        gnet_dconv_2 = self.depthwise2(gnet_dconv_1)
        gnet_dconv_3 = self.separable_conv(gnet_dconv_2)

        # Upsampling
        gnet_up_1 = self.upsample1(gnet_dconv_3)
        gnet_up_2 = self.upsample2(gnet_up_1)
        gnet_out = self.upsample3(gnet_up_2)

        # Element-wise Addition
        out = dnet_out + gnet_out + x2
        out = self.tanh(out)
        return out
        
class LF_3exp_skip_rmlastrelu(nn.Module):
    def __init__(self, args):
        super(LF_3exp_skip_rmlastrelu, self).__init__()
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
          nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel, bias=False),
          nn.Conv2d(nChannel,        3, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # Upsampling
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.upsample3 = nn.Upsample(scale_factor=2)

        # Final Output
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, x3):
        x = torch.cat((x1, x2, x3), dim=1)
        
        # DetailNet
        dnet_1 = self.relu(self.pointwise1(x))
        dnet_2 = self.relu(self.pointwise2(dnet_1))
        dnet_out = self.pointwise3(dnet_2)
        
        # GlobalNet 
        gnet_dconv_1 = self.depthwise1(x)
        gnet_dconv_2 = self.depthwise2(gnet_dconv_1)
        gnet_dconv_3 = self.separable_conv(gnet_dconv_2)

        # Upsampling
        gnet_up_1 = self.upsample1(gnet_dconv_3)
        gnet_up_2 = self.upsample2(gnet_up_1)
        gnet_out = self.upsample3(gnet_up_2)

        # Element-wise Addition
        out = dnet_out + gnet_out + x2
        out = self.sigmoid(out)
        return out
        
class LF_3exp_skip(nn.Module):
    def __init__(self, args):
        super(LF_3exp_skip, self).__init__()
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
          nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel, bias=False),
          nn.Conv2d(nChannel,        3, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # Upsampling
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.upsample3 = nn.Upsample(scale_factor=2)

        # Final Output
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, x3):
        x = torch.cat((x1, x2, x3), dim=1)
        
        # DetailNet
        dnet_1 = self.relu(self.pointwise1(x))
        dnet_2 = self.relu(self.pointwise2(dnet_1))
        dnet_out = self.relu(self.pointwise3(dnet_2))
        
        # GlobalNet 
        gnet_dconv_1 = self.depthwise1(x)
        gnet_dconv_2 = self.depthwise2(gnet_dconv_1)
        gnet_dconv_3 = self.separable_conv(gnet_dconv_2)

        # Upsampling
        gnet_up_1 = self.upsample1(gnet_dconv_3)
        gnet_up_2 = self.upsample2(gnet_up_1)
        gnet_out = self.upsample3(gnet_up_2)

        # Element-wise Addition
        out = dnet_out + gnet_out + x2
        out = self.sigmoid(out)
        return out
        
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
          nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel, bias=False),
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
        x = torch.cat((x1, x3), dim=1)
        
        # DetailNet
        dnet_1 = self.relu(self.pointwise1(x))
        dnet_2 = self.relu(self.pointwise2(dnet_1))
        dnet_out = self.relu(self.pointwise3(dnet_2))
        
        # GlobalNet 
        gnet_dconv_1 = self.depthwise1(x)
        gnet_dconv_2 = self.depthwise2(gnet_dconv_1)
        gnet_dconv_3 = self.separable_conv(gnet_dconv_2)

        # Upsampling
        gnet_up_1 = self.upsample1(gnet_dconv_3)
        gnet_up_2 = self.upsample2(gnet_up_1)
        gnet_out = self.upsample3(gnet_up_2)

        # Element-wise Addition
        out = dnet_out + gnet_out
        out = self.tanh(out)
        return out

class LIGHTFUSE_sig(nn.Module):
    def __init__(self, args):
        super(LIGHTFUSE_sig, self).__init__()
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
          nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel, bias=False),
          nn.Conv2d(nChannel,        3, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # Upsampling
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.upsample3 = nn.Upsample(scale_factor=2)

        # Final Output
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, x3):
        x = torch.cat((x1, x3), dim=1)
        
        # DetailNet
        dnet_1 = self.relu(self.pointwise1(x))
        dnet_2 = self.relu(self.pointwise2(dnet_1))
        dnet_out = self.relu(self.pointwise3(dnet_2))
        
        # GlobalNet 
        gnet_dconv_1 = self.depthwise1(x)
        gnet_dconv_2 = self.depthwise2(gnet_dconv_1)
        gnet_dconv_3 = self.separable_conv(gnet_dconv_2)

        # Upsampling
        gnet_up_1 = self.upsample1(gnet_dconv_3)
        gnet_up_2 = self.upsample2(gnet_up_1)
        gnet_out = self.upsample3(gnet_up_2)

        # Element-wise Addition
        out = dnet_out + gnet_out
        out = self.sigmoid(out)
        return out

class myHDR4map(nn.Module):
    def __init__(self, args):
        super(myHDR4map, self).__init__()
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

        self.depthwise_out = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1)
        
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
        #out = out + x2
        out = torch.sigmoid(out)
        #out = self.tanh(out)
        if DEBUG_FLAG == 2: print(f"[INFO] sigmoid_out: {out.min()} ~ {out.max()}")
        return out

class myHDR4noise(nn.Module):
    def __init__(self, args):
        super(myHDR4noise, self).__init__()
        nChannel = args.nChannel
        nFeat = args.nFeat
        self.args = args
        print(f"[INFO] nChannel : {nChannel}, nFeat : {nFeat}")
        
        self.conv1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(nFeat, nChannel, kernel_size=3, stride=1, padding=1)
        
        # Final Output
        self.relu = nn.ReLU()

    def forward(self, x1, x2, x3):
        DEBUG_FLAG = 0
        
        if DEBUG_FLAG == 2: print(f"[INFO] x1: {x1.min():.6f} ~ {x1.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x2: {x2.min():.6f} ~ {x2.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x3: {x3.min():.6f} ~ {x3.max():.6f}")
        
        x = x2
        
        if DEBUG_FLAG == 1: print(f"[INFO] x : {x.shape}")
        conv_in = self.relu(self.conv1(x))
        if DEBUG_FLAG == 1: print(f"[INFO] conv_in : {conv_in.shape}")
        conv_res = self.relu(self.conv2(conv_in))
        if DEBUG_FLAG == 1: print(f"[INFO] conv_res : {conv_res.shape}")
        conv_out = self.conv3(conv_res)
        if DEBUG_FLAG == 1: print(f"[INFO] conv_out : {conv_out.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] dnet_out: {conv_out.min()} ~ {conv_out.max()}")
        

        # Element-wise Addition
        out = conv_out
        if DEBUG_FLAG == 2: print(f"[INFO] out: {out.min()} ~ {out.max()}")
        out = torch.sigmoid(out)
        if DEBUG_FLAG == 2: print(f"[INFO] sigmoid_out: {out.min()} ~ {out.max()}")
        return out

class myHDR4noise_res(nn.Module):
    def __init__(self, args):
        super(myHDR4noise_res, self).__init__()
        nChannel = args.nChannel
        nFeat = args.nFeat
        self.args = args
        print(f"[INFO] nChannel : {nChannel}, nFeat : {nFeat}")
        
        self.conv1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(nFeat, nChannel, kernel_size=3, stride=1, padding=1)
        
        # Final Output
        self.relu = nn.ReLU()

    def forward(self, x1, x2, x3):
        DEBUG_FLAG = 0
        
        if DEBUG_FLAG == 2: print(f"[INFO] x1: {x1.min():.6f} ~ {x1.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x2: {x2.min():.6f} ~ {x2.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x3: {x3.min():.6f} ~ {x3.max():.6f}")
        
        x = x2
        
        if DEBUG_FLAG == 1: print(f"[INFO] x : {x.shape}")
        conv_in = self.relu(self.conv1(x))
        if DEBUG_FLAG == 1: print(f"[INFO] conv_in : {conv_in.shape}")
        conv_res = self.relu(self.conv2(conv_in))
        if DEBUG_FLAG == 1: print(f"[INFO] conv_res : {conv_res.shape}")
        conv_out = self.conv3(conv_res)
        if DEBUG_FLAG == 1: print(f"[INFO] conv_out : {conv_out.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] dnet_out: {conv_out.min()} ~ {conv_out.max()}")
        

        # Element-wise Addition
        out = conv_out + x
        if DEBUG_FLAG == 2: print(f"[INFO] out: {out.min()} ~ {out.max()}")
        out = torch.sigmoid(out)
        if DEBUG_FLAG == 2: print(f"[INFO] sigmoid_out: {out.min()} ~ {out.max()}")
        return out
    
class myHDR4noise_channelwise(nn.Module):
    def __init__(self, args):
        super(myHDR4noise_channelwise, self).__init__()
        nChannel = args.nChannel
        nFeat = args.nFeat
        self.args = args
        print(f"[INFO] nChannel : {nChannel}, nFeat : {nFeat}")
        
        self.dconv1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, stride=1, padding=1, groups=nChannel)
        self.dconv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1, groups=nFeat)
        self.dconv3 = nn.Conv2d(nFeat, nChannel, kernel_size=3, stride=1, padding=1, groups=nFeat)
        
        # Final Output
        self.relu = nn.ReLU()

    def forward(self, x1, x2, x3):
        DEBUG_FLAG = 0
        
        if DEBUG_FLAG == 2: print(f"[INFO] x1: {x1.min():.6f} ~ {x1.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x2: {x2.min():.6f} ~ {x2.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x3: {x3.min():.6f} ~ {x3.max():.6f}")
        
        x = x2
        
        if DEBUG_FLAG == 1: print(f"[INFO] x : {x.shape}")
        dconv_in = self.relu(self.dconv1(x))
        if DEBUG_FLAG == 1: print(f"[INFO] conv_in : {dconv_in.shape}")
        dconv_res = self.relu(self.dconv2(dconv_in))
        if DEBUG_FLAG == 1: print(f"[INFO] conv_res : {dconv_res.shape}")
        dconv_out = self.dconv3(dconv_res)
        if DEBUG_FLAG == 1: print(f"[INFO] conv_out : {dconv_out.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] dnet_out: {dconv_out.min()} ~ {dconv_out.max()}")
        

        # Element-wise Addition
        out = dconv_out
        if DEBUG_FLAG == 2: print(f"[INFO] out: {out.min()} ~ {out.max()}")
        out = torch.sigmoid(out)
        if DEBUG_FLAG == 2: print(f"[INFO] sigmoid_out: {out.min()} ~ {out.max()}")
        return out

class myHDR4noise_channelwise_res(nn.Module):
    def __init__(self, args):
        super(myHDR4noise_channelwise_res, self).__init__()
        nChannel = args.nChannel
        nFeat = args.nFeat
        self.args = args
        print(f"[INFO] nChannel : {nChannel}, nFeat : {nFeat}")
        
        self.dconv1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, stride=1, padding=1, groups=nChannel)
        self.dconv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1, groups=nFeat)
        self.dconv3 = nn.Conv2d(nFeat, nChannel, kernel_size=3, stride=1, padding=1, groups=nFeat)
        
        # Final Output
        self.relu = nn.ReLU()

    def forward(self, x1, x2, x3):
        DEBUG_FLAG = 0
        
        if DEBUG_FLAG == 2: print(f"[INFO] x1: {x1.min():.6f} ~ {x1.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x2: {x2.min():.6f} ~ {x2.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x3: {x3.min():.6f} ~ {x3.max():.6f}")
        
        x = x2
        
        if DEBUG_FLAG == 1: print(f"[INFO] x : {x.shape}")
        dconv_in = self.relu(self.dconv1(x))
        if DEBUG_FLAG == 1: print(f"[INFO] conv_in : {dconv_in.shape}")
        dconv_res = self.relu(self.dconv2(dconv_in))
        if DEBUG_FLAG == 1: print(f"[INFO] conv_res : {dconv_res.shape}")
        dconv_out = self.dconv3(dconv_res)
        if DEBUG_FLAG == 1: print(f"[INFO] conv_out : {dconv_out.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] dnet_out: {dconv_out.min()} ~ {dconv_out.max()}")
        

        # Element-wise Addition
        out = dconv_out + x
        if DEBUG_FLAG == 2: print(f"[INFO] out: {out.min()} ~ {out.max()}")
        out = torch.sigmoid(out)
        if DEBUG_FLAG == 2: print(f"[INFO] sigmoid_out: {out.min()} ~ {out.max()}")
        return out

class myHDR4noise_channelwise_unbias(nn.Module):
    def __init__(self, args):
        super(myHDR4noise_channelwise_unbias, self).__init__()
        nChannel = args.nChannel
        nFeat = args.nFeat
        self.args = args
        print(f"[INFO] nChannel : {nChannel}, nFeat : {nFeat}")
        
        self.dconv1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, stride=1, padding=1, bias=False ,groups=nChannel)
        self.dconv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1, bias=False, groups=nFeat)
        self.dconv3 = nn.Conv2d(nFeat, nChannel, kernel_size=3, stride=1, padding=1, bias=False, groups=nFeat)
        
        # Final Output
        self.relu = nn.ReLU()

    def forward(self, x1, x2, x3):
        DEBUG_FLAG = 0
        
        if DEBUG_FLAG == 2: print(f"[INFO] x1: {x1.min():.6f} ~ {x1.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x2: {x2.min():.6f} ~ {x2.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x3: {x3.min():.6f} ~ {x3.max():.6f}")
        
        x = x2
        
        if DEBUG_FLAG == 1: print(f"[INFO] x : {x.shape}")
        dconv_in = self.relu(self.dconv1(x))
        if DEBUG_FLAG == 1: print(f"[INFO] conv_in : {dconv_in.shape}")
        dconv_res = self.relu(self.dconv2(dconv_in))
        if DEBUG_FLAG == 1: print(f"[INFO] conv_res : {dconv_res.shape}")
        dconv_out = self.dconv3(dconv_res + dconv_in)
        if DEBUG_FLAG == 1: print(f"[INFO] conv_out : {dconv_out.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] dnet_out: {dconv_out.min()} ~ {dconv_out.max()}")
        

        # Element-wise Addition
        out = dconv_out + x
        if DEBUG_FLAG == 2: print(f"[INFO] out: {out.min()} ~ {out.max()}")
        out = torch.sigmoid(out)
        if DEBUG_FLAG == 2: print(f"[INFO] sigmoid_out: {out.min()} ~ {out.max()}")
        return out
