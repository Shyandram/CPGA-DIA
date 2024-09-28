import os
import torch
import numpy as np
from PIL import Image
from os.path import join
import cv2
import glob
import random

from guided_filter_pytorch.guided_filter import GuidedFilter
import kornia as K

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.bmp', '.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

class LLIEDataset(torch.utils.data.Dataset):
    def __init__(self, ori_root, lowlight_root, transforms, istrain = False, isdemo = False, dataset_type = 'LOL-v1'):
        self.lowlight_root = lowlight_root
        self.ori_root = ori_root
        self.matching_dict = {}
        self.file_list = []
        self.istrain = istrain
        self.get_image_pair_list(dataset_type)
        self.transforms = transforms
        self.isdemo = isdemo
        print("Total data examples:", len(self.file_list))

    def __getitem__(self, item):
        """
        :param item:
        :return: haze_img, ori_img
        """
        ori_image_name, ll_image_name = self.file_list[item]
        ori_image = self.transforms(
            Image.open(ori_image_name).convert('RGB')
            )
        LL_image_PIL = Image.open(ll_image_name).convert('RGB')

        # LL_image_DC = self.transforms(
        #     estimatedarkchannel(LL_image_PIL)
        #     )
        # LL_image_BC = self.transforms(
        #     estimatebrightchannel(LL_image_PIL)
        #     )
        # LL_image_y = self.transforms(
        #     LL_image_PIL.convert('L')
        #     )
        # LL_image_DBC = torch.concat([LL_image_DC, LL_image_BC, LL_image_y], dim=0)
        
        LL_image = self.transforms(
            LL_image_PIL
            )
        # if self.istrain:        
            # LL_image = torchvision.transforms.GaussianBlur(kernel_size=(7, 9), sigma=(0.1, 5.))(LL_image)
            # LL_image = K.augmentation.RandomGamma((0.8,1.2), p=0.5, keepdim=True)(LL_image)
            # LL_image = K.augmentation.RandomGaussianNoise(p=0.2, keepdim=True)(LL_image)
            # LL_image = K.augmentation.RandomPosterize(keepdim=True)(LL_image)
        if self.isdemo:
            return ori_image, LL_image, LL_image, ori_image_name.split('/')[-1].split("\\")[-1]
        
        return ori_image, LL_image, LL_image

    def __len__(self):
        return len(self.file_list)
    
    def get_image_pair_list(self, dataset_type):

        if dataset_type == 'LOL-v1':
            image_name_list = [join(self.lowlight_root, x) for x in os.listdir(self.lowlight_root) if is_image_file(x)]
            for key in image_name_list:
                key = key.split("/")[-1]
                if os.name == 'nt':
                    key = key.split("\\")[-1]
                self.file_list.append([os.path.join(self.ori_root, key), 
                                    os.path.join(self.lowlight_root, key)])
        elif dataset_type == 'LOL-v2' or dataset_type == 'LOL-v2-real' or dataset_type == 'LOL-v2-Syn':
            if self.istrain:
                Real_Low_root = join(self.lowlight_root,'Real_captured', 'Train', "Low")
                Synthetic_Low_root = join(self.lowlight_root,'Synthetic', 'Train', "Low")
                Real_High_root = join(self.ori_root,'Real_captured', 'Train', "Normal")
                Synthetic_High_root = join(self.ori_root,'Synthetic', 'Train', "Normal")
            else:
                Real_Low_root = join(self.lowlight_root,'Real_captured', 'Test', "Low")
                Synthetic_Low_root = join(self.lowlight_root,'Synthetic', 'Test', "Low")
                Real_High_root = join(self.ori_root,'Real_captured', 'Test', "Normal")
                Synthetic_High_root = join(self.ori_root,'Synthetic', 'Test', "Normal")
            
            # For Real
            if dataset_type == 'LOL-v2-Syn':
                Real_name_list =[]
            else:
                Real_name_list = [join(Real_Low_root, x) for x in os.listdir(Real_Low_root) if is_image_file(x)]
            
            for key in Real_name_list:
                key = key.split("/")[-1]
                if os.name == 'nt':
                    key = key.split("\\")[-1]
                self.file_list.append([os.path.join(Real_High_root, 'normal'+key[3:]), 
                                    os.path.join(Real_Low_root, key)])
            
            # For Synthetic

            if dataset_type == 'LOL-v2-real':
                Synthetic_name_list =[]
            else:
                Synthetic_name_list = [join(Synthetic_Low_root, x) for x in os.listdir(Synthetic_Low_root) if is_image_file(x)]
            
            for key in Synthetic_name_list:
                key = key.split("/")[-1]
                if os.name == 'nt':
                    key = key.split("\\")[-1]
                self.file_list.append([os.path.join(Synthetic_High_root, key), 
                                    os.path.join(Synthetic_Low_root, key)])
        
        elif dataset_type == 'RESIDE':
            image_name_list = [x for x in os.listdir(self.lowlight_root) if is_image_file(x)]
            # if self.istrain:
            if os.path.isfile( os.path.join(self.ori_root, image_name_list[0].split('_')[0]+'.jpg')):
                FileE = '.jpg'
            else:
                FileE = '.png'
            for key in image_name_list:
                key = key.split("/")[-1]
                if os.name == 'nt':
                    key = key.split("\\")[-1]
                self.file_list.append([os.path.join(self.ori_root, key.split('_')[0]+FileE), 
                                    os.path.join(self.lowlight_root,key)])   
        elif dataset_type == 'expe':
            image_name_list = [x for x in os.listdir(self.lowlight_root) if is_image_file(x)]
            if os.path.isfile( os.path.join(self.ori_root, '_'.join(image_name_list[0].split('_')[:-1])+'.jpg')):
                FileE = '.jpg'
            else:
                FileE = '.png'
            for key in image_name_list:
                key = key.split("/")[-1]
                if os.name == 'nt':
                    key = key.split("\\")[-1]
                self.file_list.append([os.path.join(self.ori_root, '_'.join(key.split('_')[:-1])+FileE), 
                                    os.path.join(self.lowlight_root,key)])   
        if self.istrain or (dataset_type[:6] == 'LOL-v2'):
            random.shuffle(self.file_list)

    def add_dataset(self, ori_root, lowlight_root, dataset_type = 'LOL-v1',):
        self.lowlight_root = lowlight_root
        self.ori_root = ori_root
        self.get_image_pair_list(dataset_type)

class MEFDataset(torch.utils.data.Dataset):
    def __init__(self, image_root, transforms):
        self.lowlight_root = image_root
        self.matching_dict = {}
        self.file_list = []
        self.get_image_pair_list()
        self.transforms = transforms
        print("Total data examples:", len(self.file_list))

    def __getitem__(self, item):
        """
        :param item:
        """
        a_image_name, b_image_name, key = self.file_list[item]
        
        A_image = self.transforms(
            Image.open(a_image_name).convert('RGB')
            )
        B_image = self.transforms(
            Image.open(b_image_name).convert('RGB')
            )
        # B_image = GFF(A_image, B_image)
        
        return A_image, B_image, key

    def __len__(self):
        return len(self.file_list)
    
    def get_image_pair_list(self):

        name_list = os.listdir(self.lowlight_root)
        for key in name_list:
            path = os.path.join(self.lowlight_root, key)
            A_path = os.path.join(key+'_A')
            fe = '.tif'
            for x in os.listdir(path):
                if x[:len(A_path)] == A_path:
                    fe = x[len(A_path):] 
                    break
            
            self.file_list.append([
                                   os.path.join(self.lowlight_root, key, key+'_A'+fe), 
                                    os.path.join(self.lowlight_root, key, key+'_B'+fe),
                                    key
                                    ])

def GFF(im1, im2, r1=30, r2=15, eps1 = 0.3, eps2 = 1e-6):
    im1 = im1.unsqueeze(0)
    im2 = im2.unsqueeze(0)
    GF1 = GuidedFilter(r1, eps1)
    GF2 = GuidedFilter(r2, eps2)
    Blur = K.filters.GaussianBlur2d((31,31), (1, 1))
    Blur1 = K.filters.GaussianBlur2d((3,3), (1, 1))
    Lapl = K.filters.Laplacian(kernel_size=3)
    
    b1 = Blur(im1)
    d1 = im1 - b1
    b2 = Blur(im2)
    d2 = im2 - b2
    
    laplacian1 = torch.abs(Lapl(im1))
    s1 = Blur1(laplacian1)
    laplacian2 = torch.abs(Lapl(im2))
    s2 = Blur1(laplacian2)
    
    p1 = torch.zeros_like(im1, dtype=torch.uint8)
    p2 = torch.zeros_like(im2, dtype=torch.uint8)
    
    p1[s1 >= s2] = 1
    p2[s2 > s1] = 1
    
    w1 = GF1(im1, p1)
    w2 = GF2(im2, p2)
    
    bf = w1 * b1 + w2 * b2
    df = w1 * d1 + w2 * d2
    
    fused_im = bf+df
    fused_im = fused_im.squeeze(0)
    return fused_im
    
    
        
def estimatedarkchannel(im,sz=3):
    im = np.asarray(im)
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return Image.fromarray(dark)


def estimatebrightchannel(im,sz=3):
    im = np.asarray(im)
    b,g,r = cv2.split(im)
    bc = cv2.max(cv2.max(r,g),b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    bright = cv2.dilate(bc,kernel)
    return Image.fromarray(bright)

def he(img):
    img = np.asarray(img)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return Image.fromarray(img_output)

def clahe(img):
    img = np.asarray(img)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    
    return Image.fromarray(img_output)

class LLIE_Dataset(LLIEDataset):
    def __init__(self, ori_root, lowlight_root, transforms, istrain = True):
        self.lowlight_root = lowlight_root
        self.ori_root = ori_root
        self.image_name_list = glob.glob(os.path.join(self.lowlight_root, '*.png'))
        self.matching_dict = {}
        self.file_list = []
        self.istrain = istrain
        self.get_image_pair_list()
        self.transforms = transforms
        print("Total data examples:", len(self.file_list))

    def __getitem__(self, item):
        """
        :param item:
        :return: haze_img, ori_img
        """
        ori_image_name, ll_image_name = self.file_list[item]
        ori_image = self.transforms(
            Image.open(ori_image_name)
            )

        LL_image_PIL = Image.open(ll_image_name)
        LL_image = self.transforms(
            LL_image_PIL
            )
        
        return ori_image, LL_image

    def __len__(self):
        return len(self.file_list)