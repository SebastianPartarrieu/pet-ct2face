from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import nibabel as nib
import numpy as np
import torch

class PET_dataset(Dataset):
  def __init__(self, list_ct, list_pet, transform=None):
    self.ct_names = list_ct
    self.pet_names = list_pet
    self.transform = transform

  def __len__(self):
    return(len(self.ct_names))

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    ct = nib.load(self.ct_names[idx]).get_fdata()
    pet = nib.load(self.pet_names[idx]).get_fdata()
    sample = {"pet":pet, "ct":ct}

    if self.transform:
      sample = self.transform(sample)
    return sample

### implement custom preprocessing of the data

class Normalize(object):
  def __call__(self, sample, eps=1e-5):
    pet, ct = sample['pet'], sample['ct']
    
    # get top 3% intensity to normalize between -1 and 1
    high_intensity_pet = np.percentile(np.abs(pet), 97, axis=(0,1))
    high_intensity_pet[high_intensity_pet < eps] = 1

    high_intensity_ct = np.percentile(np.abs(ct), 97, axis=(0,1))
    high_intensity_ct[high_intensity_ct < eps] = 1

    pet = pet/high_intensity_pet
    ct = ct/high_intensity_ct  
    return {"pet": pet, "ct": ct}


class Rescale(object):
  def __init__(self, output_size=(400, 400, 650)):
    self.output_size = output_size
  
  def __call__(self, sample):
    pet, ct = sample['pet'], sample['ct']
    assert pet.shape == ct.shape

    x, y, z = pet.shape
    dx, dy, dz = (self.output_size[0] - x)//2, (self.output_size[1] - y)//2, (self.output_size[2] - z)//2
    new_pet = np.pad(pet, ((dx, dx), (dy, dy), (0, 0)))
    new_ct = np.pad(ct, ((dx, dx), (dy, dy), (0, 0)))
    return {'pet': new_pet, 'ct': new_ct}

class ToTensor(object):
  def __call__(self, sample):
    pet, ct = sample["pet"], sample["ct"]

    # transpose imgs because 
    # np img : HxWxC
    # torch img : CxHxW
    pet = pet.transpose(2, 0, 1)
    ct = ct.transpose(2, 0, 1)
    return {"pet": torch.Tensor(pet), "ct":torch.Tensor(ct)}