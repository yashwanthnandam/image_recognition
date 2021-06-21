from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class GestureDataset(Dataset):
    """Gesture dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.gesture_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.gesture_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.gesture_frame.iloc[idx, 0]+'.jpg')
        image = io.imread(img_name)
        label = self.gesture_frame.iloc[idx, 1]
        label = int(label)

        if self.transform:
            image = self.transform(image)

        return image, label