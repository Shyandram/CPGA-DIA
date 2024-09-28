import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.color import rgb_to_y
from guided_filter_pytorch.guided_filter import FastGuidedFilter

class LAL_BDP(nn.Module):
    def __init__(self, n_channels=8):
        super(LAL_BDP, self).__init__()
        ch_n = 3 # 3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=ch_n, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=ch_n, out_channels=ch_n, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=ch_n*2, out_channels=ch_n, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=ch_n*2, out_channels=ch_n, kernel_size=7, stride=1, padding=3)
        self.conv5 = nn.Conv2d(in_channels=ch_n*4, out_channels=1, kernel_size=1, stride=1, padding=0)

        # A_estimation
        self.conv1_A = nn.Conv2d(in_channels=3, out_channels=n_channels, kernel_size=1, stride=1, padding=0)
        self.conv2_A = ResBlock(conv=default_conv, n_feats=n_channels, kernel_size=3)
        self.conv3_A = ResBlock(conv=default_conv, n_feats=n_channels, kernel_size=3)
        self.conv4_A = nn.Conv2d(in_channels=n_channels, out_channels=3, kernel_size=1, stride=1, padding=0)
           

    def forward(self, x, dbc=None, get_all=False):
        dbc = self.get_dbc(x)
        # T estimation
        t = self.k_estimation(dbc)
        A = self.A_estimation(x)
        
        out = ((x-A)*t + A)
        out = torch.clamp(out, min=1e-9, max=1)

        output = out
        if get_all:
            return dbc, t, A, output
        
        return output, t
    def get_t(self, x):
        dbc = self.get_dbc(x)
        t = self.k_estimation(dbc)
        return t
    def r_estimation(self, x):
        x1 = F.relu(self.conv1_r(x))
        x2 = F.relu(self.conv2_r(x1))
        cat1 = torch.cat((x1, x2), 1)
        x3 = F.relu(self.conv3_r(cat1))
        cat2 = torch.cat((x2, x3), 1)
        x4 = F.relu(self.conv4_r(cat2))
        cat3 = torch.cat((x1, x2, x3, x4), 1)
        k = F.sigmoid(self.conv5_r(cat3))

        return k 
        
    def k_estimation(self, dbc):
        x1 = F.relu(self.conv1(dbc))
        x2 = F.relu(self.conv2(x1))
        cat1 = torch.cat((x1, x2), 1)
        x3 = F.relu(self.conv3(cat1))
        cat2 = torch.cat((x2, x3), 1)
        x4 = F.relu(self.conv4(cat2))
        cat3 = torch.cat((x1, x2, x3, x4), 1)
        k = F.relu(self.conv5(cat3))

        return k

    def A_estimation(self, x):
        y1 = self.conv1_A(x)
        y = self.conv2_A(y1)
        y = F.relu(y)
        y = self.conv3_A(y)
        y = self.conv4_A(y)

        return y
    
    def get_dbc(self, rgb):
        img_max, _ = torch.max(rgb, 1, keepdim=True)
        img_min, _ = torch.min(rgb, 1, keepdim=True)
        y = rgb_to_y(rgb)
        return torch.cat((img_max, img_min, y), dim=1)

class enhance_color(nn.Module):
    def __init__(self, n_channels=16, isdgf=False, *args, **kwargs) -> None:
        super(enhance_color, self).__init__(*args, **kwargs)

        # Gamma_estimation
        self.conv1_g = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv2_g = ResCBAM(16, 16)
        self.conv3_g = ResCBAM(16, 16)
        self.conv4_g = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.gap_g = nn.AdaptiveAvgPool2d(1)
        
        self.conv1_post_g = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv2_post_g = ResBlock(conv=default_conv, n_feats=16, kernel_size=3)
        self.conv3_post_g = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1, stride=1, padding=0)

        # LLIE
        self.ll = LAL_BDP(n_channels=n_channels)

        if isdgf:
            self.gf = FastGuidedFilter(r = 1)
        else:
            self.gf = isdgf
            
    def __call__(self, x, gamma_pre = 0, ext=False, isvid=False, get_all=False):
        if self.gf:
            xx = x
            x = F.interpolate(x, [x.shape[2]//2, x.shape[3]//2], mode='bicubic', align_corners=True)
        if get_all:
            dbc, t, A, llie = self.ll(x, x, get_all=get_all)
        else:
            llie, t = self.ll(x, x,)
        output = llie

        gamma = self.gamma_estimation(x)
      
        out_g = llie
        out_g = torch.pow(out_g, gamma)
        
        out = self.conv1_post_g(torch.cat((out_g, llie), dim=1))
        out = self.conv2_post_g(out)
        intersection = self.conv3_post_g(out)
        output = -intersection + out_g + llie
        
        output = torch.clamp(output, min=1e-9, max=1)

        if self.gf:
            output = self.gf(x, output, xx)            
            output = torch.clamp(output, min=1e-9, max=1)
            
        if get_all:
            return output, gamma, intersection,out_g, dbc, llie, t, A, 
        if not ext and not isvid:
            return output
        elif isvid:
            return output, gamma, llie
        else:
            return output, intersection
    
    def gamma_estimation(self, x):

        y = self.conv1_g(x)
        y = self.conv2_g(y)
        y = self.conv3_g(y)

        g = self.conv4_g(y)
        g = self.gap_g(g)
        g = torch.clamp(input=g, min=1e-8, max=1e8)

        return g
    
    def fusion_strategy(self, llie, out_g, wintsec=False):
        
        out = self.conv1_post_g(torch.cat((out_g, llie), dim=1))
        out = self.conv2_post_g(out)
        intersection = self.conv3_post_g(out)
        output = -intersection + out_g + llie
        
        output = torch.clamp(output, min=1e-9, max=1)
        if wintsec:
            return output, intersection
        return output
        

class CPGA_DN(nn.Module):
    def __init__(self, n_channels=16, isdgf=False, *args, **kwargs) -> None:
        super(CPGA_DN, self).__init__()
        # Gamma_estimation        
        self.gamma_est = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0),
            ResCBAM(16, 16),
            ResCBAM(16, 16),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d((1, 50)),
            nn.Linear(50, 2),
            nn.Tanh()
        )
        self.post_fusion = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=1, stride=1, padding=0),
            ResBlock(conv=default_conv, n_feats=16, kernel_size=3),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1, stride=1, padding=0)
        )
        
        self.ll_d =enhance_color()
        self.ll_n = enhance_color()
        if isdgf:
            self.gf = FastGuidedFilter(r = 1)
        else:
            self.gf = isdgf
            
    def __call__(self, x, gamma_pre=0, ext=False, isvid=False, get_all=False):
        
        if self.gf:
            xx = x
            x = F.interpolate(x, [x.shape[2]//2, x.shape[3]//2], mode='bicubic', align_corners=True)
            
        llie, _ = self.ll_d.ll(x,x)
        nlie, _ = self.ll_n.ll(x,x)
        
        g_l = self.ll_d.gamma_estimation(x)     
        g_n = self.ll_n.gamma_estimation(x)    
        
        g = self.gamma_estimation(x)  
        g = torch.reshape(g, [-1, 2])
        g_l = (g_l+(g[:,0].reshape_as(g_l)))
        g_n = (g_n+(g[:,1]).reshape_as(g_n))
        g_l = torch.clamp(g_l, min=1e-9, )
        g_n = torch.clamp(g_n, min=1e-9, )
        
        llie_g = torch.pow(llie, g_l)
        nlie_g = torch.pow(nlie, g_n)
        
        ll = self.ll_d.fusion_strategy(llie, llie_g)
        nl = self.ll_n.fusion_strategy(nlie, nlie_g)
        g_intersection = self.post_fusion(torch.cat((ll, nl), dim=1))
        output = llie + nlie - g_intersection
        
        if self.gf:
            output = self.gf(x, output, xx)
                
        output = torch.clamp(output, min=1e-9, max=1)
        if isvid:
            return output, g_l, g_n, g[:,0].reshape_as(g_l), g[:,1].reshape_as(g_n)
        return output
    
    def gamma_estimation(self, x):

        y = self.gamma_est(x)

        g = y

        return g
    
class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res
    
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    
class ResCBAM(nn.Module):
    def __init__(self, inplanes, planes, downsample=None):
        super(ResCBAM, self).__init__()
        self.conv1 = default_conv(inplanes, planes, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = default_conv(inplanes, planes, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


if __name__ == '__main__':
    x = torch.randn(8, 3, 600, 400)
    model = enhance_color()
    model(x)
    model = CPGA_DN()
    model(x)