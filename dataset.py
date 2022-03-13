from datetime import datetime, time, timedelta
import os
import numpy as np
import pandas as pd
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch.utils.data as data
from torchvision.datasets.utils import download_url
import albumentations as A
from PIL import Image


class ClimateHackDataset(data.Dataset):
    """
    Args:
        root (string): Root directory of dataset where directory
            ``MiraBest-F.py` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an np array
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional):
    """
    def __init__(self, root='cleaned_data', split=None,
                 transform=None, target_transform=None,
                 csv='./cleaned_data/meta_data.csv',
                 target_time_steps=24
                ):
        self.root = os.path.expanduser(root)
        file_list = [file for file in os.listdir(self.root) if '.npy' in file]
        self.csv = pd.read_csv(csv)
        csv_length = self.csv.shape[0]
        self.target_time_steps = target_time_steps
        
        # Set test data is the last 20% of the data. Should be treated identically to the validation data?
        self.test_list  = [entry.replace('-','_')+'.npy' for entry in self.csv[-csv_length//5:].date_column.unique()]
        # Set train and validation as randomly selected days throughout the data set
        residual_list = [entry for entry in file_list if entry not in self.test_list]
        self.train_list = residual_list[:-20]
        self.val_list = residual_list[-20:] ### Make >20 after debugging complete
        
        # The random shuffle might be a good idea, but it would become more difficult to 
        # track the data using the csvs.
        #rng = np.random.default_rng(seed=42)
        #rng.random.shuffle(residual_list)
        
        #train_list = residual_list[:(len(residual_list) // 10) * 8]
        #val_list   = residual_list[(len(residual_list) // 10) * 8:]
        
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # training set, validation set or test set
        
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.')
        
        if self.split == 'train':
            used_list = self.train_list
            #self.csv  = self.csv.iloc[-csv_length//5:]
        elif self.split in ['val', 'validation']:
            used_list = self.val_list
        elif self.split == 'test':
            used_list = self.test_list
            #self.csv  = self.csv.iloc[:-csv_length//5-2]
        else:
            used_list = self.train_list + self.val_list + self.test_list
        self.csv  = self.csv[self.csv.date_column.str.replace('-','_').str.contains('|'.join([file[:-4] for file in used_list]))]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if index > self.csv[self.csv.valid_start].shape[0]:
            return
        # Read out valid array
        df = self.csv[self.csv.valid_start].reset_index()
        idx = df.idx.iloc[index]
        date = df.date_column.iloc[index]
        file_name = date.replace('-','_')+'.npy'
        file_path = os.path.join(self.root, file_name)        
        try:
            data = np.load(file_path, mmap_mode='c')
        except:
            print(file_path)
        df = df[df.date_column==date].reset_index()
        date_index = df[df.idx==idx].index[0]
        # Select starting index
        #t0 = self.csv[self.csv.valid_start].iloc[index].idx
        
        # Derive index within the data array for that date
        #valid_index = df[df.idx==t0].index[0]
        
        # Extract indices for desired images
        img    = data[date_index:date_index+12]
        target = data[date_index+12:date_index+12+self.target_time_steps]
        
        # Transform with the same same random crops
        if self.transform is not None:
            img = img.transpose(1,2,0)
            target = target.transpose(1,2,0)
            augmented = self.transform(image=img, target=target)
            img    = augmented['image'].transpose(2,0,1)
            target = augmented['target'].transpose(2,0,1)
        
        # Crop target to (64,64)
        #target = A.CenterCrop(64,64)(image=target.transpose(1,2,0))['image'].transpose(2,0,1)
        
        # output with dimensions (t,c,h,w)
        return img[:,np.newaxis,:,:].astype(np.float32), target[:,np.newaxis,:,:].astype(np.float32)
        #return img[:,np.newaxis,:,:].astype(np.float32), target[:,np.newaxis,:,:][:1].astype(np.float32)

    def __len__(self):
        return self.csv[self.csv.valid_start].shape[0]

    def _check_integrity(self):
        """Doesn't actually check anything other than whether or not the paths are files are there."""
        for filename in (self.train_list + self.test_list):
            fpath = os.path.join(self.root, filename)
            if not os.path.isfile(fpath):
                return False
        return True

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.split)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str