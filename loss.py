from typing import Any
import torch
import torchvision
import torch.nn as nn
import kornia as K
from torch.autograd import Variable
from kornia.color import rgb_to_yuv, yuv_to_rgb, rgb_to_y
# https://github.com/winfried-ripken/deep-hist

class hist_retinex_loss(nn.Module):
    def __init__(self):
        super(hist_retinex_loss, self).__init__()
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(self, y, cbcr, gt):
        gt_yuv = rgb_to_yuv(gt)
        # img_rgb = yuv_to_rgb(torch.cat((y, cbcr), dim=1))

        l_y = self.smooth_l1(y, gt_yuv[:, :1])
        l_cbcr = self.smooth_l1(cbcr, gt_yuv[:, 1:])
        # l_rgb = self.smooth_l1(img_rgb, gt)
        
        return l_y, l_cbcr

class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self, device='cuda'):
        super(VGGLoss, self).__init__()
        # self.vgg = VGG19().to(device)
        index = 31
        vgg_model = torchvision.models.vgg16(pretrained=True).to(device)
        for param in vgg_model.parameters():
            param.requires_grad = False
        self.vgg = nn.Sequential(*list(vgg_model.features.children())[:index])
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        # self.criterion = nn.MSELoss()
        # self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(self.vgg_preprocess(x)), self.vgg(self.vgg_preprocess(y))
        # loss = 0
        # for i in range(len(x_vgg)):
        #     loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        loss = torch.mean((self.instancenorm(x_vgg) - self.instancenorm(y_vgg)) ** 2)
        return loss
    
    def vgg_preprocess(self, batch): 
        tensor_type = type(batch.data) 
        (r, g, b) = torch.chunk(batch, 3, dim=1) 
        batch = torch.cat((b, g, r), dim=1)  # convert RGB to BGR 
        batch = batch * 255  # * 0.5  [-1, 1] -> [0, 255] 
        mean = tensor_type(batch.data.size()).cuda() 
        mean[:, 0, :, :] = 103.939 
        mean[:, 1, :, :] = 116.779 
        mean[:, 2, :, :] = 123.680 
        batch = batch.sub(Variable(mean))  # subtract mean 
        return batch 
    
class HIST_loss(nn.Module):
    def __init__(self, sample_rate=0.1):
        super(HIST_loss, self).__init__()
        self.criterion = nn.CosineSimilarity(dim=0, eps=1e-6)
        self.blur = K.filters.blur_pool2d
        self.r=sample_rate
    def forward(self, x, y):
        shape = x.shape[-2] * x.shape[-1]
        x = self.blur(x, 7)
        y = self.blur(y, 7)

        total_loss = 0
        for b in range(x.shape[0]):
            for c in range(x.shape[1]):
                x_h = torch.histc(x[b:c], int(256*self.r), 0, 1) / shape
                y_h = torch.histc(y[b:c], int(256*self.r), 0, 1) / shape
                total_loss += self.criterion(x_h, y_h) 
        loss = total_loss/(x.shape[0]*x.shape[1])
        return loss

class HSVLoss(nn.Module):
    def __init__(self, h=0, s=1, v=0.7, eps=1e-7, threshold_h=0.03, threshold_sv=0.1):
        super(HSVLoss, self).__init__()
        self.hsv = [h, s, v]
        self.loss = nn.L1Loss(reduction='none')
        self.eps = eps

        # since Hue is a circle (where value 0 is equal to value 1 that are both "red"), 
        # we need a threshold to prevent the gradient explod effect
        # the smaller the threshold, the optimal hue can to more close to target hue
        self.threshold_h = threshold_h
        # since Hue and (Value and Satur) are conflict when generated image' hue is not the target Hue, 
        # we have to condition V to prevent it from interfering the Hue loss
        # the larger the threshold, the ealier to activate V loss
        self.threshold_sv = threshold_sv

    def get_hsv(self, im):
        img = im * 0.5 + 0.5
        hue = torch.Tensor(im.shape[0], im.shape[2], im.shape[3]).to(im.device)

        hue[ img[:,2]==img.max(1)[0] ] = 4.0 + ( (img[:,0]-img[:,1]) / ( img.max(1)[0] - img.min(1)[0] + self.eps) ) [ img[:,2]==img.max(1)[0] ]
        hue[ img[:,1]==img.max(1)[0] ] = 2.0 + ( (img[:,2]-img[:,0]) / ( img.max(1)[0] - img.min(1)[0] + self.eps) ) [ img[:,1]==img.max(1)[0] ]
        hue[ img[:,0]==img.max(1)[0] ] = (0.0 + ( (img[:,1]-img[:,2]) / ( img.max(1)[0] - img.min(1)[0] + self.eps) ) [ img[:,0]==img.max(1)[0] ]) % 6

        hue[img.min(1)[0]==img.max(1)[0]] = 0.0
        hue = hue/6

        saturation = ( img.max(1)[0] - img.min(1)[0] ) / ( img.max(1)[0] + self.eps )
        saturation[ img.max(1)[0]==0 ] = 0

        value = img.max(1)[0]
        return hue, saturation, value

    def get_rgb_from_hsv(self):
        C = self.hsv[2] * self.hsv[1]
        X = C * ( 1 - abs( (self.hsv[0]*6)%2 - 1 ) )
        m = self.hsv[2] - C

        if self.hsv[0] < 1/6:
            R_hat, G_hat, B_hat = C, X, 0
        elif self.hsv[0] < 2/6:
            R_hat, G_hat, B_hat = X, C, 0
        elif self.hsv[0] < 3/6:
            R_hat, G_hat, B_hat = 0, C, X
        elif self.hsv[0] < 4/6:
            R_hat, G_hat, B_hat = 0, X, C
        elif self.hsv[0] < 5/6:
            R_hat, G_hat, B_hat = X, 0, C
        elif self.hsv[0] <= 6/6:
            R_hat, G_hat, B_hat = C, 0, X

        R, G, B = (R_hat+m), (G_hat+m), (B_hat+m)
        
        return R, G, B
    
    
    def forward(self, input):
        h, s, v = self.get_hsv(input)

        target_h = torch.Tensor(h.shape).fill_(self.hsv[0]).to(input.device).type_as(h)
        target_s = torch.Tensor(s.shape).fill_(self.hsv[1]).to(input.device).type_as(s)
        target_v = torch.Tensor(v.shape).fill_(self.hsv[2]).to(input.device).type_as(v)

        loss_h = self.loss(h, target_h)
        loss_h[loss_h<self.threshold_h] = 0.0
        loss_h = loss_h.mean()

        if loss_h < self.threshold_h*3:
            loss_h = torch.Tensor([0]).to(input.device)
        
        loss_s = self.loss(s, target_s).mean()
        if loss_h.item() > self.threshold_sv:   
            loss_s = torch.Tensor([0]).to(input.device)

        loss_v = self.loss(v, target_v).mean()
        if loss_h.item() > self.threshold_sv:   
            loss_v = torch.Tensor([0]).to(input.device)

        return loss_h + 4e-1*loss_s + 4e-1*loss_v

class Perceptual_loss(nn.Module):
    # https://github.com/fengzhang427/HEP/blob/main/models/loss.py
    def __init__(self):
        super(Perceptual_loss, self).__init__()
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

        index = 31
        vgg_model = torchvision.models.vgg16(pretrained=True)
        for param in vgg_model.parameters():
            param.requires_grad = False
        self.vgg = nn.Sequential(*list(vgg_model.features.children())[:index])


    def forward(self, img, target):
        # img: LLIE output
        # target: original low light input image
        # img_he = self.Histogram_Equalize(img)
        img_fea = self.vgg(self.vgg_preprocess(img))
        target_he = self.Histogram_Equalize(target)
        target_fea = self.vgg(self.vgg_preprocess(target_he))
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)
    
    def Histogram_Equalize(self, img):
        y, u, v = torch.split(rgb_to_yuv(img), 1 ,dim=1)
        y = K.enhance.equalize(y)
        out = torch.cat((y, u, v), dim=1)
        out = yuv_to_rgb(out)
        return out
    
    def vgg_preprocess(self, batch): 
        tensor_type = type(batch.data) 
        (r, g, b) = torch.chunk(batch, 3, dim=1) 
        batch = torch.cat((b, g, r), dim=1)  # convert RGB to BGR 
        batch = batch * 255  # * 0.5  [-1, 1] -> [0, 255] 
        mean = tensor_type(batch.data.size()).cuda() 
        mean[:, 0, :, :] = 103.939 
        mean[:, 1, :, :] = 116.779 
        mean[:, 2, :, :] = 123.680 
        batch = batch.sub(Variable(mean))  # subtract mean 
        return batch 
        

class BDCPLoss(nn.Module):
    def __init__(self,):
        super(BDCPLoss, self).__init__()
        self.mse = nn.MSELoss()
    def __call__(self, rgb_img, rgb_gt, *args: Any, **kwds: Any):
        bcp_loss = self.mse(self.get_bcp(rgb_img), self.get_bcp(rgb_gt))
        dcp_loss = self.mse(self.get_dcp(rgb_img), self.get_dcp(rgb_gt))
        loss = bcp_loss+dcp_loss
        return loss
    def get_bcp(self, rgb):
        img_max, _ = torch.max(rgb, 1, keepdim=True)
        return img_max
    def get_dcp(self, rgb):
        img_min, _ = torch.min(rgb, 1, keepdim=True)
        return img_min




if __name__ == '__main__':
    x = torch.rand(13, 3, 245,245)
    y = torch.rand(13, 3, 245,245)
    his = HIST_loss()
    his(x, y)
    # hep = Perceptual_loss()
    # hep(x , y)