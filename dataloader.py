import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import random

do_augmentation = True
print('AUGMENTATION: ', do_augmentation)

def augment(a):
    #Transpose  axes
    do_transpose = random.randint(0, 1)
    permutation = (0, 1, 3, 2)
    if do_transpose:
        a = np.transpose(a, permutation)
    #Flip axes
    flip_axes = np.random.randint(2, size=3)
    axes = np.where(flip_axes)[0]+1
    a = np.flip(a, axes)
    #Change color intensity
    color_intensity = random.uniform(0.8, 1.2)
    a = a * color_intensity
    #Roll
    shift = np.random.randint(-8, high=8, size=(3,), dtype=int)
    a = np.roll(a, shift=shift, axis=(1, 2, 3))
    return a

#np.roll
#-> csak tanítás időben
#random skalárral szorozni az intenzitásokat 0.8,  1.2 között
#pool a felére, nagyobb batch size 

class BratsDataset(Dataset):
    def __init__(self, datadir, extension='.npy', transform=None):
        if not os.path.isdir(datadir):
            raise Exception('Not existing data dir: ', datadir)
        self.datadir = datadir
        self.extension = extension
        self.transform = transform
        self.ids = [i[:-len(self.extension)] for i in os.listdir(datadir) if i.endswith(extension)]
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        patient_id = self.ids[idx]
        data_path = os.path.join(self.datadir, patient_id+self.extension)
        data = np.load(data_path)
        if self.transform != None:
            data = self.transform(data)
        image = data[:4]
        seg = data[4]
        seg = seg[None, ...]
        return image, seg, patient_id
    
datadir_train = '/home/csanyid/BraTS/data/preprocessed_train/'
if do_augmentation:
    train_ds = BratsDataset(datadir_train, transform = augment)
else:
    train_ds = BratsDataset(datadir_train, transform = None)
datadir_val = '/home/csanyid/BraTS/data/preprocessed_val/'
val_ds = BratsDataset(datadir_val)

batch_size = 8

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
