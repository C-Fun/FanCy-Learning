import os

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class Dataset(object):
    name = ""
    normalize = None
    num_classes = None
    ori_im_size = None
    TorchDataset = None

    def __init__(self, path, im_size=None):
        self.path = path
        if type(im_size) == int:
            im_size = (im_size, im_size)

        if im_size==None:
            self.im_size = self.ori_im_size
        else:
            self.im_size = im_size

        self.train_dataset = self.train_dataset()
        self.test_dataset = self.test_dataset()

    def train_dataset(self):
        return self.TorchDataset(root=self.path, train=True, download=True,
                                 transform=self.transform_train())

    def test_dataset(self):
        return self.TorchDataset(root=self.path, train=False, download=True,
                                 transform=self.transform_test())

    def transform_train(self):
        if self.im_size==self.ori_im_size:
            transform = transforms.Compose([transforms.RandomCrop(self.im_size),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       self.normalize])
        else:
            transform = transforms.Compose([transforms.Resize(self.im_size),
                                       transforms.RandomResizedCrop(self.im_size),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       self.normalize])
        return transform

    def transform_test(self):
        return transforms.Compose([transforms.Resize(self.im_size),
                                   transforms.CenterCrop(self.im_size),
                                   transforms.ToTensor(),
                                   self.normalize,
                                   ])

    def train_loader(self, batch_size, shuffle=True):
        return torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=shuffle)

    def test_loader(self, batch_size=1, shuffle=False):
        return torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=batch_size, shuffle=shuffle)


class CIFAR100(Dataset):
    name = "CIFAR100"
    normalize = transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))
    num_classes = 100
    ori_im_size = (32, 32)
    TorchDataset = datasets.CIFAR100

    def __init__(self, path='./', im_size=None):
        super(CIFAR100, self).__init__(path+'/data/cifar_data', im_size)


class CIFAR10(Dataset):
    name = "CIFAR10"
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    num_classes = 10
    ori_im_size = (32, 32)
    TorchDataset = datasets.CIFAR10

    def __init__(self, path='./', im_size=None):
        super(CIFAR10, self).__init__(path+'/data/cifar_data', im_size)


class STL10(Dataset):
    name = "STL10"
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    num_classes = 10
    ori_im_size = (96, 96)
    TorchDataset = datasets.STL10

    def __init__(self, path='./', im_size=None):
        super(STL10, self).__init__(path+'/data/stl10_data', im_size)

    def train_dataset(self):
        return self.TorchDataset(root=self.path, split="train", download=True,
                                 transform=self.transform_train())

    def test_dataset(self):
        return self.TorchDataset(root=self.path, split="test", download=True,
                                 transform=self.transform_test())

class ImageNet(Dataset):
    name = "ImageNet"
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    num_classes = 1000
    ori_im_size = (224, 224)
    TorchDataset = datasets.ImageFolder

    def __init__(self, path='/m2/data/imagenet', im_size=None, traindir=None, valdir=None):
        super(ImageNet, self).__init__(path, im_size)
        if traindir==None:
            self.traindir = os.path.join(path, 'train')
        else:
            self.traindir = traindir
        if valdir==None:
            self.valdir = os.path.join(path, 'val')
        else:
            self.valdir = traindir

    def train_dataset(self):
        return self.TorchDataset(self.traindir, self.transform_train())

    def test_dataset(self):
        return self.TorchDataset(self.valdir, self.transform_test())

def get_dataset(name, **kwargs):
    NAME = name.upper()
    if NAME=='CIFAR100':
        return CIFAR100(**kwargs)
    elif NAME=='CIFAR10':
        return CIFAR10(**kwargs)
    elif NAME=='STL10':
        return STL10(**kwargs)
    elif NAME=='IMAGENET':
        return ImageNet(**kwargs)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    dataset = STL10('../')
    print(dataset.train_loader(8))
