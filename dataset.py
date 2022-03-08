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

# Dataloader object: https://pytorch.org/docs/stable/data.html#module-torch.utils.data
#Dataloader(dataset, batch_size=1, shuffle=False, sampler=None,
#           batch_sampler=None, num_workers=0, collate_fn=None,
#           pin_memory=False, drop_last=False, timeout=0,
#           worker_init_fn=None, *, prefetch_factor=2,
#           persistent_workers=False)
# Composite transforms
#


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
                 csv='./cleaned_data/meta_data.csv'
                ):
        self.root = os.path.expanduser(root)
        file_list = [file for file in os.listdir(self.root) if '.npy' in file]
        self.csv = pd.read_csv(csv)
        csv_length = self.csv.shape[0]
        
        # Set test data is the last 20% of the data. Should be treated identically to the validation data?
        self.test_list  = [entry.replace('-','_')+'.npy' for entry in self.csv[-csv_length//5:].date_column.unique()]
        # Set train and validation as randomly selected days throughout the data set
        residual_list = [entry for entry in file_list if entry not in self.test_list]
        self.train_list = residual_list[:-2]
        self.val_list = residual_list[-2:] ### Make >20 after debugging complete
        
        # Theoretically the random shuffle is a good idea, but it would become more difficult to 
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
        
        # now load the picked numpy arrays
        self.data = []
        for file_name in used_list:
            file_path = os.path.join(self.root, file_name)
            try:
                entry = np.load(file_path, mmap_mode='c')
            except:
                print(file_path)
            self.data.append(entry)
        
        self.data = np.vstack(self.data).reshape(-1, 891, 1843)
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # Select starting index
        t0 = self.csv[self.csv.valid_start].iloc[index].idx
        #day_time = index within day which t0 refers to on a global scale. ### TODO
        day_time=0
        # Extract file name to load
        # Extract indices for desired images
        img    = self.data[day_time:day_time+12]
        target = self.data[day_time+12:day_time+36]
        
        # Transform with the same same random crops
        
        # Crop target more
        
        # Transformation
        
        # Notes:
            # To create the target, maybe treat it as a mask: https://albumentations.ai/docs/getting_started/mask_augmentation/
            # Multi-target augmentations: https://albumentations.ai/docs/examples/example_multi_target/
            # Might support C>3 data: https://github.com/albumentations-team/albumentations/issues/152

        if self.transform is not None:
            img = self.transform(image=img)['image']

        if self.target_transform is not None:
            target = self.target_transform(image=target)['image']

        return img, target
    
    
    
    def __len__(self):
        return len(self.data)

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