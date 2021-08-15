from numpy.core.numeric import zeros_like
from torch._C import device
from torch.utils.data import Dataset
import torch
import os
import torch.nn.functional as F
# import io
from skimage import io
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
from torchvision import transforms as T
import matplotlib.pyplot as plt
from torch.utils import data

def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
        color_jitter = T.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = T.RandomApply([color_jitter], p=0.8)
        rnd_gray = T.RandomGrayscale(p=0.2)
        color_distort = T.Compose([
        rnd_color_jitter,
        rnd_gray])
        return color_distort
class XviewsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, img_dir, label_dir, transform1=None,transform2=None,transform3=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.label_dir = label_dir
        self.img_dir = img_dir
        self.images=[f for f in listdir(img_dir) if isfile(join(img_dir, f))]
        self.labels=[f for f in listdir(label_dir) if isfile(join(label_dir, f))]
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print(self.img_dir)
        # print(self.label_dir)
        # print(self.images[idx])
        img_name = os.path.join(self.img_dir,
                                self.images[idx])
        label_name=os.path.join(self.label_dir,
                                self.labels[idx])
        # print(img_name)
        # print(label_name)
        image1=Image.open(img_name)
        image2=Image.open(img_name)
        label=Image.open(label_name)
        if self.transform1:
            image1 = self.transform1(image1)
        if self.transform2:
            image2=self.transform2(image2)
        if self.transform3:
            label=self.transform3(label)
        # landmarks = self.landmarks_frame.iloc[idx, 1:]
        # landmarks = np.array([landmarks])
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image1': image1, 'image2': image2,'label':label}

        # if self.transform:
        #     sample = self.transform(sample)

        return sample
        # return image1,image2,label

transform = T.Compose([
                T.Resize([513,513]),
                T.CenterCrop([513,513]),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
transform2=T.Compose([
        get_color_distortion(),
        transform
        
    ])

transform3=T.Compose([
        T.PILToTensor(),
         T.Resize([65,65],Image.NEAREST)
    ])
train_dst=XviewsDataset('D:/downloads/rdata512/rdata512/train_pre/images',
                                    'D:/downloads/rdata512/rdata512/train_pre/targets',transform,transform2,transform3)

def my_collate(batch):
    print(len(batch))
    len_batch = len(batch) # original batch length

    batch = list(filter (lambda x:torch.count_nonzero(x['label'])!=0, batch)) # filter out all the Nones
    print(len(batch))
    if len_batch > len(batch): # source all the required samples from the original dataset at random
        diff = len_batch - len(batch)
        while diff != 0:
            a = train_dst[np.random.randint(0, len(train_dst))]
            if torch.count_nonzero(a['label'])==0:                
                continue
            batch.append(a)
            diff -= 1
        # for i in range(diff):
        #     batch.append(train_dst[np.random.randint(0, len(train_dst))])

    yy=torch.utils.data.dataloader.default_collate(batch)
    # print(yy)
    return yy
train_loader = data.DataLoader(
            train_dst, batch_size=16, shuffle=True,collate_fn=my_collate)
count1=0
for samples in train_loader:
    label=samples['label']
    image1=samples['image1']
    image2=samples['image2']
    print(image1.shape)
    print(image2.shape)
    print(label.shape)
    device=torch.device('cuda')
    for count in range(16):
                label1=label[count]
                # jj=(label1!=0)
                # print(torch.masked_select(label1, jj))
                label1 = label1.to(device, dtype=torch.long)
                one_hot = torch.nn.functional.one_hot(label1)
                # xx=F.normalize(label1, p=2, dim=1)
                print(one_hot.shape)
                print(label1.shape)
                print(one_hot)
                print(torch.count_nonzero(label1))
    count1+=1
    if count1>1:
        break
