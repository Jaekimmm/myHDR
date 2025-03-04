import os
import torch
from torch.nn import init
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.distributed as dist
import torchvision.models as models

def mk_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        if classname.find('Conv2d') != -1:
            init.kaiming_normal_(m.weight.data)
        else:
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def model_load(model, trained_model_dir, model_file_name):
    model_path = os.path.join(trained_model_dir, model_file_name)
    # trained_model_dir + model_file_name    # '/modelParas.pkl'
    print(f"[INFO] Load model from {model_path}")
    model.load_state_dict(torch.load(model_path))
    return model

class EarlyStopping:    # copy from freeSoul
    def __init__(self, patience=3, delta=0.0, mode='min', verbose=True):
        """
        patience (int): loss or score가 개선된 후 기다리는 기간. default: 3
        delta  (float): 개선시 인정되는 최소 변화 수치. default: 0.0
        mode     (str): 개선시 최소/최대값 기준 선정('min' or 'max'). default: 'min'.
        verbose (bool): 메시지 출력. default: True
        """
        self.early_stop = False
        self.best_update = False
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        
        self.best_score = np.Inf if mode == 'min' else 0
        self.mode = mode
        self.delta = delta

    def __call__(self, score):

        if self.best_score is None:
            self.best_score = score
            self.counter = 0
        elif self.mode == 'min':
            if score < (self.best_score - self.delta):
                self.counter = 0
                self.best_update = True
                self.best_score = score
                if self.verbose:
                    print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f}')
            else:
                self.counter += 1
                self.best_update = False
                if self.verbose:
                    print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                          f'Best: {self.best_score:.5f}' \
                          f', Current: {score:.5f}, Delta: {np.abs(self.best_score.item() - score.item()):.5f}')
                    
                    
        elif self.mode == 'max':
            if score > (self.best_score + self.delta):
                self.counter = 0
                self.best_score = score
                if self.verbose:
                    print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f}')
            else:
                self.counter += 1
                if self.verbose:
                    print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                          f'Best: {self.best_score:.5f}' \
                          f', Current: {score:.5f}, Delta: {np.abs(self.best_score.item() - score.item()):.5f}')
                
            
        if self.counter >= self.patience:
            if self.verbose:
                print(f'[EarlyStop Triggered] Best Score: {self.best_score:.5f}')
            # Early Stop
            self.early_stop = True
        else:
            # Continue
            self.early_stop = False
  
            
# Evaluation Metrics (copy from freeSoul)
def psnr(im0, im1):
    """ This function computes the Peak Signal to Noise Ratio (PSNR) between two images whose ranges are [0-1].
        the mu-law tonemapped image.
        Args:
            im0 (np.ndarray): Image 0, should be of same shape and type as im1
            im1 (np.ndarray: Image 1,  should be of same shape and type as im0

        Returns:
            np.ndarray (): Returns the mean PSNR value for the complete image.

        """
    return -10*np.log10(np.mean(np.power(im0-im1, 2)))

def normalized_psnr(im0, im1, norm):
    """ This function computes the Peak Signal to Noise Ratio (PSNR) between two images that are normalized by the
    specified norm value.

        the mu-law tonemapped image.
        Args:
            im0 (np.ndarray): Image 0, should be of same shape and type as im1
            im1 (np.ndarray: Image 1,  should be of same shape and type as im0
            norm (float) : Normalization value for both images.

        Returns:
            np.ndarray (): Returns the mean PSNR value for the complete image.

        """
    return psnr(im0/norm, im1/norm)


# Tone-mapping functions 
def mu_tonemap(hdr_image, mu=5000):
    """ This function computes the mu-law tonemapped image of a given input linear HDR image.

    Args:
        hdr_image (np.ndarray): Linear HDR image with values in the range of [0-1]
        mu (float): Parameter controlling the compression performed during tone mapping.

    Returns:
        np.ndarray (): Returns the mu-law tonemapped image.

    """
    return np.log(1 + mu * hdr_image) / np.log(1 + mu)

def norm_mu_tonemap(hdr_image, norm_value, mu=5000):
    """ This function normalizes the input HDR linear image by the specified norm_value and then computes
    the mu-law tonemapped image.
    Args:
        hdr_image (np.ndarray): Linear HDR image with values in the range of [0-1]
        norm_value (float): Value for the normalization (i.e. hdr_image/norm_value)
        mu (float): Parameter controlling the compression performed during tone mapping.

    Returns:
        np.ndarray (): Returns the mu-law tonemapped image.

    """
    return mu_tonemap(hdr_image/norm_value, mu)

def psnr_norm_mu(im0,im1,norm):
    return psnr(norm_mu_tonemap(im0,norm),norm_mu_tonemap(im1,norm))

def tanh_norm_mu_tonemap(hdr_image, norm_value, mu=5000):
    """ This function normalizes the input HDR linear image by the specified norm_value, afterwards bounds the
    HDR image values by applying a tanh function and afterwards computes the mu-law tonemapped image.

        the mu-law tonemapped image.
        Args:
            hdr_image (np.ndarray): Linear HDR image with values in the range of [0-1]
            norm_value (float): Value for the normalization (i.e. hdr_image/norm_value)
            mu (float): Parameter controlling the compression performed during tone mapping.

        Returns:
            np.ndarray (): Returns the mu-law tonemapped image.

        """
    bounded_hdr = np.tanh(hdr_image / norm_value)
    return  mu_tonemap(bounded_hdr, mu)

def psnr_tanh_norm_mu_tonemap(hdr_nonlinear_ref, hdr_nonlinear_res, percentile=99, gamma=2.24):
    """ This function computes Peak Signal to Noise Ratio (PSNR) between the mu-law computed images from two non-linear
    HDR images.

            Args:
                hdr_nonlinear_ref (np.ndarray): HDR Reference Image after gamma correction, used for the percentile norm
                hdr_nonlinear_res (np.ndarray: HDR Estimated Image after gamma correction
                percentile (float): Percentile to to use for normalization
                gamma (float): Value used to linearized the non-linear images

            Returns:
                np.ndarray (): Returns the mean mu-law PSNR value for the complete image.

            """
    hdr_linear_ref = hdr_nonlinear_ref**gamma
    hdr_linear_res = hdr_nonlinear_res**gamma
    norm_perc = np.percentile(hdr_linear_ref, percentile)
    
    return psnr(tanh_norm_mu_tonemap(hdr_linear_ref, norm_perc), tanh_norm_mu_tonemap(hdr_linear_res, norm_perc))

def tonemap(img, option=None):
    if option == 'mu':
        return mu_tonemap(img)
    elif option == 'norm_mu':
        norm_perc = np.percentile(img, 99)
        return norm_mu_tonemap(img, norm_perc)
    elif option == 'tanh_norm_mu':
        norm_perc = np.percentile(img, 99)
        return tanh_norm_mu_tonemap(img, norm_perc)
    elif option == 'gamma':
        gamma = 2.24
        return img ** gamma
    elif option == 'degamma':
        gamma = 1/2.24
        return img ** gamma
    elif option == 'gamma_tanh':
        gamma = 2.24
        img = img ** gamma
        return tanh_norm_mu_tonemap(img, norm_perc)
    elif option == 'degamma_tanh':
        gamma = 1/2.24
        img = img ** gamma
        return tanh_norm_mu_tonemap(img, norm_perc)
    else:
        return img
    


def save_plot(trained_model_dir):
    # CSV 파일 읽기 (첫 번째 행이 헤더라면 skiprows=1)
    file_path = trained_model_dir + '/plot_data.txt' 
    data = np.loadtxt(file_path, delimiter=",", skiprows=0)

    # CSV에서 데이터 읽기 (0번째 열: epoch, 1번째 열: train_loss, 2번째 열: val_loss)
            # fplot.write(f'{epoch},{train_loss},{valid_loss},{psnr},{psnr_mu}\n')
    epochs = data[1:, 0]
    train_loss = data[1:, 1]
    val_loss = data[1:, 2]

    # 그래프 그리기
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Training Loss", color="blue", linewidth=2)
    plt.plot(epochs, val_loss, label="Validation Loss", color="red", linestyle="dashed", linewidth=2)

    # 그래프 꾸미기
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Curve")
    plt.legend()
    plt.grid(True)

    plt.savefig(f"{trained_model_dir}/curve_loss.png", dpi=300, bbox_inches="tight")  # 고해상도 저장


    # CSV에서 데이터 읽기 (0번째 열: epoch, 1번째 열: train_loss, 2번째 열: val_loss)
            # fplot.write(f'{epoch},{train_loss},{valid_loss},{psnr},{psnr_mu}\n')
    epochs = data[:, 0]
    psnr_l = data[:, 3]
    psnr_m = data[:, 4]

    # 그래프 그리기
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, psnr_l, label="PSNR", color="blue", linewidth=2)
    plt.plot(epochs, psnr_m, label="PSNR_mu", color="red",  linewidth=2)
    plt.axhline(y=43.7708, color="blue", linestyle="--", linewidth=2, label="PSNR (Baseline)")
    plt.axhline(y=47.1223, color="red",  linestyle="--", linewidth=2, label="PSNR_mu (Baseline)")

    # 그래프 꾸미기
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.title("PSNR Curve")
    plt.legend()
    plt.grid(True)

    plt.savefig(f"{trained_model_dir}/curve_psnr.png", dpi=300, bbox_inches="tight")  # 고해상도 저장

# Color conversion
def rgb_to_yuv(tensor, mode="444"):
    r, g, b = tensor[0, :, :], tensor[1, :, :], tensor[2, :, :]

    # Compute Y channel
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = (b - y) / 2.032
    v = (r - y) / 1.140
    
    u = u.clamp(min=-1.0, max=1.0)
    v = v.clamp(min=-1.0, max=1.0)
    
    if mode == "444":
        # No downsampling, keep full resolution
        yuv = torch.stack((y, u, v), dim=0)
    
    elif mode == "422":
        # Downsample U, V horizontally (W -> W/2)
        u = F.avg_pool2d(u.unsqueeze(1), kernel_size=(1, 2), stride=(1, 2)).squeeze(1)
        v = F.avg_pool2d(v.unsqueeze(1), kernel_size=(1, 2), stride=(1, 2)).squeeze(1)
        yuv = torch.stack((y, u, v), dim=1)
    
    elif mode == "420":
        # Downsample U, V both horizontally and vertically (H -> H/2, W -> W/2)
        u = F.avg_pool2d(u.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
        v = F.avg_pool2d(v.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
        yuv = torch.stack((y, u, v), dim=1)

    return yuv

def yuv_to_rgb(tensor, mode="444"):
    y, u, v = tensor[:, 0, :, :], tensor[:, 1, :, :], tensor[:, 2, :, :]

    r = y + 1.140 * v
    g = y - 0.395 * u - 0.581 * v
    b = y + 2.032 * u

    rgb = torch.stack((r, g, b), dim=1)
    rgb = torch.clamp(rgb, 0, 1)  # 0~1 범위로 클리핑

    return rgb

def yuv_to_rgb_torch_gt(tensor, mode="444"):
    y, u, v = tensor[:, 0, :], tensor[:, 1, :], tensor[:, 2, :]

    r = y + 1.140 * v
    g = y - 0.395 * u - 0.581 * v
    b = y + 2.032 * u

    rgb = torch.stack((r, g, b), dim=1)
    rgb = torch.clamp(rgb, 0, 1)  # 0~1 범위로 클리핑

    return rgb

def rgb_to_mono(tensor):
    r, g, b = tensor[:, 0, :, :], tensor[:, 1, :, :], tensor[:, 2, :, :]
    mono = 0.299 * r + 0.587 * g + 0.114 * b
    return mono.unsqueeze(1)
def rgb_to_mono_gt(tensor):
    r, g, b = tensor[:, 0, :], tensor[:, 1, :], tensor[:, 2, :]
    mono = 0.299 * r + 0.587 * g + 0.114 * b
    return mono.unsqueeze(1)

# VGG loss from LIGHTFUSE
# << Orginal kera code >>
# Patch_size = 256
# Ch         = 3
#
# Select_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
# Vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(patch_size, patch_size, ch))
# Vgg16.trainable = False
# For l in vgg16.layers:
#   l.trainable = False
# Select = [vgg16.get_layer(name).output for name in select_layers]
#
# Model_vgg = Model(inputs=vgg16.input, outputs=select)
class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        
        vgg = models.vgg16(weights='DEFAULT').features.eval()
        self.layers = nn.ModuleList(vgg[:25+1])
        for param in self.layers.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        
        return x
    
def get_activation(layer):
    """ Layer에서 Activation 함수 찾기 """
    if isinstance(layer, nn.ReLU):
        return "ReLU"
    elif isinstance(layer, nn.Sigmoid):
        return "Sigmoid"
    elif isinstance(layer, nn.Tanh):
        return "Tanh"
    return "None"

def print_model_info(model, input_size=(1, 3, 256, 256)):
    """ PyTorch 모델의 모든 Layer 정보를 출력 (ModuleList 지원) """
    
    print("--------------------------------------------------------------------------------")
    print(f"{'Layer Type':<30}{'Output Shape':<30}{'Activation':<20}")
    print("="*80)

    # 가상의 입력을 생성하여 Forward 수행
    x = torch.randn(*input_size).to(next(model.parameters()).device)

    for name, layer in model.named_children():
        # ✅ ModuleList인 경우 개별 Layer 순회
        if isinstance(layer, nn.ModuleList):
            for sub_idx, sub_layer in enumerate(layer):
                x = sub_layer(x)  # ✅ 개별 Layer 실행
                activation = get_activation(sub_layer)
                print(f"{f'{name}[{sub_idx}] ({sub_layer.__class__.__name__})':<30}"
                      f"{str(tuple(x.shape)):<30}"
                      f"{activation:<20}")
        else:
            x = layer(x)  # ✅ 일반 Layer 실행
            activation = get_activation(layer)
            print(f"{f'{name} ({layer.__class__.__name__})':<30}"
                  f"{str(tuple(x.shape)):<30}"
                  f"{activation:<20}")

    print("="*80)