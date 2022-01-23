import os.path as osp
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.datasets as datasets
import os
from PIL import Image
from PIL import ImageEnhance

transformtypedict=dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast, Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color)
class ImageJitter(object):
    def __init__(self, transformdict):
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]

    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha*(randtensor[i]*2.0 -1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')

        return out
class CUB(Dataset):

    def __init__(self, train=True):


        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.RandomRotation(45),
                transforms.RandomResizedCrop(224),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),    
        }


        data_dir = './data/cub/'
        if(train):
            dataset = datasets.ImageFolder(os.path.join(data_dir,'train'),data_transforms['train'])
        else:
            dataset = datasets.ImageFolder(os.path.join(data_dir,'test'),data_transforms['test'])           



        lb=[y for _,y in dataset]

        indx = [i for i,c in enumerate(lb) if c<=99]
        self.indx =indx
        self.dataset = dataset
        data = []
        label = []
        


        self.train = train


    def __len__(self):
        return len(self.indx)

    def __getitem__(self, i):
        image, label = self.dataset[self.indx[i]]
        return image, label

