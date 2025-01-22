import os
from glob import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class dataset_loader(Dataset):
    def __init__(self, mode, load_mode, input_path, transform=None):
        assert mode in ['train', 'validation','test'], "mode is 'train' or 'test'"
        assert load_mode in [0,1], "load_mode is 0 or 1"

        self.mode = mode
        self.load_mode = load_mode
        self.transform = transform
        self.input_path = os.listdir(input_path)
        print(input_path) 

        if mode == 'train':
            input_ = [os.path.join(input_path, f) for f in self.input_path]
            if load_mode == 0: # batch data load
                self.input_ = input_
            else: # all data load
                self.input_ = [np.load(f) for f in input_]
        else: # mode =='test'
            input_ = [f for f in input_path]
            if load_mode == 0:
                self.input_ = input_
            else:
                self.input_ = [np.load(f) for f in input_]

    def __len__(self):
        return len(self.input_)

    def __getitem__(self, idx):
        input_img_path = self.input_[idx]
        target = 1 if 'dog' in self.input_[idx] else 0

        #input_img = np.load(input_img_path)
        #input_img = Image.fromarray(input_img.astype('uint8'))
        input_img = Image.open(input_img_path)

        if self.transform:
            input_img = self.transform(input_img)

        return input_img, target



def get_loader(mode='train', load_mode=0, input_path=None, batch_size=32, num_workers=6):


    transform = transforms.Compose([
        transforms.Resize((360, 360)),
        transforms.ToTensor(),
    ])

    dataset_ = dataset_loader(mode, load_mode, input_path,  transform)
    data_loader = DataLoader(dataset=dataset_, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return data_loader
