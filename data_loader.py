from random import shuffle
import torch
from torch.utils import data
from torchvision import transforms as T
from PIL import Image

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, imList, petlist, labelList, mode = 'train'):
        self.imList = imList
        self.petlist = petlist
        self.labelList = labelList
        self.mode = mode
        self.shuffle = shuffle

    def __len__(self):
        return len(self.imList)

    def __getitem__(self, idx):

        image_name = self.imList[idx]
        pet_name = self.petlist[idx]
        label_name = self.labelList[idx]

        image = Image.open(image_name)
        pet = Image.open(pet_name)
        label = Image.open(label_name)

        Transform = []
        Transform.append(T.Resize((256, 256)))
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)          # 接受一个转换列表并返回一个按顺序应用这些转换的可调用对象

        image = Transform(image)
        pet = Transform(pet)
        label = Transform(label)
        image = image.float()
        pet = pet.float()
        label = label.float()

        Norm_ = T.Normalize(([0.5]), ([0.5]))      # 减去平均值并除以标准差来标准化图像张量
        # Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        image = Norm_(image)
        pet = Norm_(pet)
        return image, pet, label


def get_loader(imList, petlist, labelList, batch_size, num_workers=1, mode='train', drop_last=True):
    """Builds and returns Dataloader."""

    dataset = MyDataset(imList, petlist, labelList, mode=mode)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader