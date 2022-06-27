from __future__ import print_function
import argparse
import os
import glob
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import copy
from itertools import islice
import torch.utils.data as data
from torchvision import datasets, transforms
from PIL import Image
from torchvision import utils
from util.util import save_image
import networks

from torch.autograd import Variable


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.gif'
]

class ImagePair(data.Dataset):
    def __init__(self, impath1, impath2, mode='RGB', transform=None):
        self.impath1 = impath1
        self.impath2 = impath2
        self.mode = mode
        self.transform = transform

    def loader(self, path):
        return Image.open(path).convert(self.mode)

    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def get_pair(self):
        if self.is_image_file(self.impath1):
            img1 = self.loader(self.impath1)
        if self.is_image_file(self.impath2):
            img2 = self.loader(self.impath2)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2

    def get_source(self):
        if self.is_image_file(self.impath1):
            img1 = self.loader(self.impath1)
        if self.is_image_file(self.impath2):
            img2 = self.loader(self.impath2)
        return img1, img2


class ImageSequence(data.Dataset):
    def __init__(self, is_folder=False, mode='RGB', transform=None, *impaths):
        self.is_folder = is_folder
        self.mode = mode
        self.transform = transform
        self.impaths = impaths

    def loader(self, path):
        return Image.open(path).convert(self.mode)

    def get_imseq(self):
        if self.is_folder:
            folder_path = self.impaths[0]
            impaths = self.make_dataset(folder_path)
        else:
            impaths = self.impaths

        imseq = []
        for impath in impaths:
            if os.path.exists(impath):
                im = self.loader(impath)
                if self.transform is not None:
                    im = self.transform(im)
                imseq.append(im)
        return imseq

    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def make_dataset(self, img_root):
        images = []
        for root, _, fnames in sorted(os.walk(img_root)):
            for fname in fnames:
                if self.is_image_file(fname):
                    img_path = os.path.join(img_root, fname)
                    images.append(img_path)
        return images



def _crop(img,ow,oh):
    #ow, oh = raw_img.size 
    temp_arr = img[:,:,:oh,:ow]

    return temp_arr

def denorm(mean=[0, 0, 0], std=[1, 1, 1], tensor=None):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def __patch_instance_norm_state_dict(state_dict, module, keys, i=0):
    """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
    key = keys[i]
    if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
        if module.__class__.__name__.startswith('InstanceNorm') and \
                (key == 'running_mean' or key == 'running_var'):
            if getattr(module, key) is None:
                state_dict.pop('.'.join(keys))
        if module.__class__.__name__.startswith('InstanceNorm') and \
           (key == 'num_batches_tracked'):
            state_dict.pop('.'.join(keys))
    else:
        __patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

def load_(net,state_dict):
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata

    # patch InstanceNorm checkpoints prior to 0.4
    for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
        print(state_dict.keys())
        print(state_dict[key].size())
        __patch_instance_norm_state_dict(state_dict, net, key.split('.'))
    net.load_state_dict(state_dict)


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



device = torch.device("cuda:0")


net_G  = networks.define_En_ADW(1, 1, 64).cuda()


model_G_path = "models/300_net_G.pth"

state_dict_G = torch.load(model_G_path, map_location=str(device))
load_(net_G,state_dict_G)


net_G.eval()



for i in range(0, 10):



    path1 = os.path.join("./test_imgs/A/",'A'+ str(i + 1) + ".bmp")
    path2 = os.path.join("./test_imgs/B/", 'B'+str(i + 1) + ".bmp")



    img_A = Image.open(path1).convert('L')

    w, h = img_A.size

    print(w)
    print(h)

    pair_loader = ImagePair(impath1=path1, impath2=path2,
                            transform=transforms.Compose([
                                transforms.Grayscale(1),
                                #transforms.Resize((512,512)),
                                transforms.Pad((0, 0, 1024 - w, 1024 - h), padding_mode="reflect"),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5,), std=(0.5,))

                            ]))
    img1, img2 = pair_loader.get_pair()
    img1.unsqueeze_(0)
    img2.unsqueeze_(0)


    img1 = img1.cuda()
    img2 = img2.cuda()
    with torch.no_grad():

        fuse_img = net_G(img1,img2)


    fuse_img = _crop(fuse_img, w, h)
    image_path = os.path.join("./results/", str(i+1) + ".png")
    fold_path = os.path.join("./results/")
    if not os.path.exists(fold_path):
        os.makedirs(fold_path)

    utils.save_image(
        fuse_img, image_path, nrow=fuse_img.shape[0] + 1, normalize=True,
        range=(-1, 1)
    )




