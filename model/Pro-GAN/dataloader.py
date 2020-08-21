import torch
import torchvision as torchv


class Dataset:
    def __init__(self, resolution=64, size=96000):
        self.resolution = resolution
        self.size = size

        self.HubbleXDF = Image.open("../../data/HubbleXDF_cropped.jpg")
        
        self.transforms = torchv.transforms.Compose([
            torchv.transforms.RandomCrop((512, 512)),
            torchv.transforms.Resize((self.resolution, self.resolution)),
            torchv.transforms.ToTensor()
        ])
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.transforms(self.HubbleXDF)
