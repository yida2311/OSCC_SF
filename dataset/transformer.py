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
    def __init__(self, size):
        if isinstance(size, tuple):
            self.size = size
        elif isinstance(size, int):
            self.size = (size, size)
        
        self.resize = transforms.Resize(size=self.size)
    
    @staticmethod
    def get_params(x, crop_size):
        state = False # if true, need resize
        w, h = x.size
        th, tw = crop_size
        top, left, height,width = 0, 0, 0, 0

        if w <= tw:
            state = True
            left = 0
            width = w
        else:
            left = random.randint(0, w-tw)
            width = tw
        
        if h <= th:
            state = True
            top = 0
            height = h
        else:
            top = random.randint(0, h-th)
            height = th
        
        return (top, left, height,width), state
    
    def __call__(self, *args):
        box, state = self.get_params(args[0], self.size)
        results = []
        for arg in args:
            crop_arg = F.crop(arg, box[0], box[1], box[2], box[3])
            if state:
                crop_arg = self.resize(crop_arg)
            results.append(crop_arg)
        return tuple(results)


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

        return img, target