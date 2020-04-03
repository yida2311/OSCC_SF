import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
import random
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Rotate(object):
    """随机选择0， 90， 180， 270度"""
    def __call__(self, *args):
        angle = np.random.randint(0, 3)
        result = []
        for arg in args:
            result.append(F.rotate(arg, angle*90))
        return tuple(result)


class Hflip(object):
    """随机翻转"""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, *args):
        if random.random() < self.p:
            result = []
            for arg in args:
                result.append(F.hflip(arg))
            return tuple(result)
        return args


class Vflip(object):
    """随机翻转"""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, *args):
        if random.random() < self.p:
            result = []
            for arg in args:
                result.append(F.vflip(arg))
            return tuple(result)
        return args


class RandomCrop(object):
    def __init__(self, size, scale=8):
        if isinstance(size, tuple):
            self.size = size
        elif isinstance(size, int):
            self.size = (size, size)
        
        self.scale = scale
    
    @staticmethod
    def get_params(x, y, crop_size, scale):
        state = False # if true, need resize
        x_w, x_h = x.size
        y_w, y_h = y.size
        x_th, x_tw = crop_size
        y_th, y_tw = int(x_th/scale), int(x_tw/scale)
        
        if x_w == x_tw and x_h == x_th:
            return (0, 0, x_th, x_tw), (0, 0, y_w, y_h)
        
        x_i = random.randint(0, x_h-x_th)
        x_j = random.randint(0, x_w-x_tw)
        y_i = int(x_i / scale)
        y_j = int(x_j / scale)
        
        return (x_i, x_j, x_th, x_tw), (y_i, y_j, y_th, y_tw)
    
    def __call__(self, x, y):
        box_x, box_y = self.get_params(x, y, self.size, self.scale)
        crop_x = F.crop(x, box_x[0], box_x[1], box_x[2], box_x[3])
        crop_y = F.crop(y, box_y[0], box_y[1], box_y[2], box_y[3])

        return crop_x, crop_y

class Transformer(object):
    def __init__(self, size):
        self.crop = RandomCrop(size=size)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
        self.color_jitter = transforms.ColorJitter(brightness=64.0/255, contrast=0.75, saturation=0.25, hue=0.04)
        self.rotate = Rotate()
        self.hflip = Hflip()
        self.vflip = Vflip()
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img, target):
        img, target = self.crop(img, target)
        # print(img.size)
        img, target = self.rotate(img, target)
        img, target = self.hflip(img, target)
        img, target = self.vflip(img, target)
        img = self.color_jitter(img.convert('RGB'))
        img = self.to_tensor(img)
        img = self.normalize(img)
        img = transforms.ToPILImage()(img)
        target = self.to_tensor(target.convert('I'))

        return img, target


class ValTransformer(object):
    def __init__(self, size):
        self.crop = RandomCrop(size=size)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()
    
    def __call__(self, img, target):
        img, target = self.crop(img, target)
        img = self.to_tensor(img.convert('RGB'))
        img = self.normalize(img)
        img = transforms.ToPILImage()(img)
        target = self.to_tensor(target.convert('I'))
        
        return img, target


class TestTransformer(object):
    def __init__(self, size):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.resize = transforms.Resize(self.size)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img, target):
        img, target = self.to_tensor(img.convert('RGB')), self.to_tensor(target.convert('I'))
        img = self.normalize(img)
        img = transforms.ToPILImage()(img)

        return img, target