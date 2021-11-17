from data_augmentation import *
from torch.utils.data import Dataset
import os
import torch
from PIL import Image
import re

"""
possibili migliorie: aggiungere controlli 
"""


class CustomDataset(Dataset):
    def __init__(self, path, used_for_train):
        self.path = path
        self.data = self.get_data()

        if used_for_train:
            self.preprocessing = build_transforms()
            # contained in DataAugmentation file. Return a Torch.Transform object
        else:
            self.preprocessing = build_transforms(is_train=False)

    def get_data(self):
        data = list()
        imgs_names = os.listdir(self.path)
        for name in imgs_names:
            if re.match("[0-9]{4}", str(name[0:4])) is not None:
                data.append([name, name[0:4], name[6]])
        return data

    def __getitem__(self, i):
        img_path = os.path.join(self.path, self.data[i][0])
        img = Image.open(img_path)
        id = int(self.data[i][1])
        camera_id = int(self.data[i][2])
        return self.preprocessing(img), torch.tensor(id), torch.tensor(camera_id)

    def __len__(self):
        return len(self.data)


# data_set = CustomDataset(path="FinalDataset/train_set_prova", used_for_train=True)
