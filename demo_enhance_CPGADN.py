import os
import torch
import torch.backends.cudnn
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torchvision import transforms
from tqdm import tqdm
from utils import weight_init #,logger
from config import get_config
from model import CPGA_DN
from data import LLIEDataset
from fvcore.nn import FlopCountAnalysis, parameter_count_table

from skimage import img_as_ubyte
import cv2, time

import warnings
warnings.filterwarnings("ignore")


def load_data_eval(cfg):
    data_transform = transforms.Compose([
        # transforms.Resize([400, 600]),
        # transforms.RandomCrop([256, 256]),
        # transforms.CenterCrop([400, 600]),
        transforms.ToTensor()
    ])
    val_haze_dataset = LLIEDataset(cfg.val_ori_data_path, cfg.val_haze_data_path, data_transform, dataset_type = cfg.dataset_type, isdemo=True)
    val_loader = torch.utils.data.DataLoader(val_haze_dataset, batch_size=cfg.val_batch_size, shuffle=False,
                                             num_workers=cfg.num_workers, drop_last=True, pin_memory=True)

    return val_loader, len(val_loader)

def load_network(device):
    net = CPGA_DN().to(device)
    net.apply(weight_init)
    return net

def load_pretrain_network(cfg, device):
    net = CPGA_DN().to(device)
    net.load_state_dict(torch.load(os.path.join(cfg.ckpt))['state_dict'])
    return net

def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def main(cfg):
    # -------------------------------------------------------------------
    # basic config
    print(cfg)
    if cfg.gpu > -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # -------------------------------------------------------------------
    # load data
    val_loader, val_number = load_data_eval(cfg)
    # -------------------------------------------------------------------
    #
    # -------------------------------------------------------------------
    # load network
    if cfg.ckpt:
        network = load_pretrain_network(cfg, device)
    else:
        network = load_network(cfg.upscale_factor, device)
    
    # -------------------------------------------------------------------
    out_dir = os.path.join('results', cfg.net_name)
    os.makedirs(out_dir, exist_ok=True)

    # start train
    print('Start demo')
    network.eval()
    with torch.no_grad():
        valloader = tqdm(val_loader)
        valloader.set_description_str('Demo')
        for step, (ori_image, haze_image, _, name) in enumerate(valloader):
            ori_image, haze_image = ori_image.to(device), haze_image.to(device)
            LLIE_image = network(haze_image)
            LLIE_image = torch.clamp(LLIE_image, 0, 1)
            LLIE_image = LLIE_image.permute(0, 2, 3, 1).cpu().detach().numpy()
            LLIE_image = img_as_ubyte(LLIE_image[0])
            save_img((os.path.join(out_dir, name[0])), LLIE_image)
            
            
    total_time = 0
    start_t = time.time_ns()
    itr_cnt = int(1e2)
    with torch.no_grad():
        for step in enumerate(range(itr_cnt)):
            haze_image = torch.rand((1,3,400, 600)).cuda()
            # start_time = time.time_ns()
            LLIE_image = network(haze_image)
            LLIE_image = torch.clamp(LLIE_image, 0, 1)
            # y = LLIE_image.permute(0, 2, 3, 1).cpu().detach().numpy()
            # y = img_as_ubyte(y[0])
            # end_time = time.time_ns()
            # temp = end_time - start_time
            # total_time += temp
    end_t = time.time_ns()
       
           
        
    print('finish demo')
        # # train finish
        # summary.close()
    # ---------------------------
    input = torch.randn(1, 3, 400, 600).to(device)
    # macs, params = profile(network, inputs=(input, ))
    flops = FlopCountAnalysis(network, (input, ))
    print('FLOPs = ' + str(flops.total()/1000**3) + 'G')
    # print('MACs = ' + str(macs/1000**3) + 'G')
    # print('Params = ' + str(params/1000**2) + 'M')
    print(sum(p.numel() for p in network.parameters() if p.requires_grad))
    
    # ---------------------------
    
    print("Average time taken by network is : %f ms"%(total_time/1e9*1e3/itr_cnt))
    print("Average time (total) taken by network is : %f ms"%((end_t-start_t)/1e9*1e3/itr_cnt))

if __name__ == '__main__':
    config_args, unparsed_args = get_config()
    main(config_args)
