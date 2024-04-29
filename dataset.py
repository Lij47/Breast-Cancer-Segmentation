import torch
from torch.utils.data import Dataset
import os
import numpy as np
import glob
from PIL import Image
from torchvision.transforms import v2
from torchvision import transforms

class BreastCancerDataset(Dataset):
    def __init__(self, dataset_configs):

        self.image_path_list = sorted(glob.glob(pathname=dataset_configs["image_path"]))
        self.mask_path_list = sorted(glob.glob(pathname=dataset_configs["mask_path"]))
        for image_path, mask_path in zip(self.image_path_list, self.mask_path_list):
            assert(os.path.basename(image_path) == os.path.basename(mask_path))

        RESIZE = (224, 224)
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]

        self.image_transforms = transforms.Compose([transforms.Resize(RESIZE),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean = MEAN, std = STD)])

        self.mask_transforms = transforms.Compose([transforms.Resize(RESIZE), 
                                            transforms.PILToTensor()])

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_path_list[idx]).convert("RGB")
        mask = Image.open(self.mask_path_list[idx])
        unnormalized_image = self.mask_transforms(image)
        image = self.image_transforms(image)
        mask = self.mask_transforms(mask)
        mask = mask.squeeze(dim=0)
        mask = mask.type(torch.long)
        return image, mask, unnormalized_image.type(torch.uint8)