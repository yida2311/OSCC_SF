import os
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageFile
import random
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.autograd import Variable
import torch 
import cv2
import pandas as pd 


ImageFile.LOAD_TRUNCATED_IMAGES = True

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def RGB_mapping_to_calss(label):
    h, w = label.shape[0], label.shape[1]
    classmap = np.zeros(shape=(h, w))

    indices = np.where(np.all(label == (255, 0, 0), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 1
    indices = np.where(np.all(label == (0, 255, 0), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 2
    indices = np.where(np.all(label == (0, 0, 255), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 3
    indices = np.where(np.all(label == (0, 0, 0), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 0

    return classmap


def class_to_RGB(label):
    h, w = label.shape[0], label.shape[1]
    colmap = np.zeros(shape=(h, w, 3)).astype(np.float32)

    indices = np.where(label == 1)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [255, 0, 0]
    indices = np.where(label == 2)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 255, 0]
    indices = np.where(label == 3)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 0, 255]
    indices = np.where(label == 0)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 0, 0]
    transform = ToTensor()

    return transform(colmap)


def class_to_target(inputs, numClass):
    batchSize, h, w = inputs.shape[0], inputs.shape[1], inputs.shape[2]
    target = np.zeros(shape=(batchSize, h, w, numClass), dtype=np.float32)
    for index in range(numClass):
        indices = np.where(inputs == index)
        temp = np.zeros(shape=numClass, dtype=np.float32)
        temp[index] = 1
        target[indices[0].tolist(), indices[1].tolist(), indices[2].tolist(), :] = temp
    return target.transpose(0, 3, 1, 2) 


def label_bluring(inputs):
    batchSize, numClass, height, width = inputs.shape
    outputs = np.ones((batchSize, numClass, height, width), dtype=np.float)
    for batchCnt in range(batchSize):
        for index in range(numClass):
            outputs[batchCnt, index, ...] = cv2.GaussianBlur(inputs[batchCnt, index, ...].astype(np.float), (7, 7), 0)
    return outputs


def pil_loader(path):
    img = Image.open(path)
    return img


def resize(image, shape, label=False):
    '''
    resize PIL image
    shape: (w, h)
    '''
    if label:
        return image.resize(shape, Image.NEAREST)
    else:
        return image.resize(shape, Image.BILINEAR)


def masks_transform_form(masks, shape, label=True, numpy=False):
    '''
    masks: list of PIL images
    '''
    targets = []
    for m in masks:
        m = resize(m, shape, label=label)
        targets.append(_mask_transform(m))
    targets = np.array(targets)
    if numpy:
        return targets
    else:
        return torch.from_numpy(targets).long()


def _mask_transform(mask):
    target = np.array(mask).astype('int32')
    target[target == 255] = -1
    # target -= 1 # in DeepGlobe: make class 0 (should be ignored) as -1 (to be ignored in cross_entropy)
    return target


class DeepOSCC(data.Dataset):
    """
    input and label image dataset
    """
    
    def __init__(self, 
                root, 
                meta_file,
                args,
                mode=1, # 1:global; 2:local; 3:global&local
                val=False,
                data_suffix='train',
                target_suffix='target_train', 
                transform=None):
        super(DeepOSCC, self).__init__()
        """
        Args:
            root -->
            transform -->
        """
        self.root = root
        self.root_img = os.path.join(root, data_suffix)
        self.root_target = os.path.join(root, target_suffix)
        self.mode = mode
        self.val = val
        self.transform = transform
        self.classdict = {1: "normal", 2: "mucosa", 3: "tumor", 0: "background"}

        # self.to_tensor = transforms.ToTensor()
        samples = []
        df = pd.read_csv(meta_file)
        if self.val:
            for i in df.index:
                cnt = df.iloc[i]
                if int(cnt[1]) == 4:
                    img_path = os.path.join(self.root_img, cnt[0])
                    target_path = os.path.join(self.root_target, cnt[0])
                    samples.append((img_path, target_path))
        else:
            for i in df.index:
                cnt = df.iloc[i]
                if int(cnt[1]) != 4:
                    img_path = os.path.join(self.root_img, cnt[0])
                    target_path = os.path.join(self.root_target, cnt[0])
                    samples.append((img_path, target_path))
        self.samples = samples
    
    def __getitem__(self, index):
        img_path, target_path = self.samples[index]
        img, target = pil_loader(img_path), pil_loader(target_path)
        path_split = img_path.split('/')
        name = path_split[-1]
        # print(name)

        if self.transform is not None:
            img, target = self.transform(img, target)
        sample = self.sample_generator(img, target, self.mode)
        sample['id'] = name
        
        return sample
    
    def sample_generator(self, img, target, mode):
        w, h = img.shape
        sample['output_size'] = (h//8, w//8)
        w = w//4; h = h//4
        sample = {"label": target}

        if mode == 1:
            sample['image_g'] = resize(img, (w, h))
        elif mode == 2:
            sample['image_l'] = img
        elif mode == 3:
            sample['image_l'] = img
            sample['image_g'] = resize(img, (w, h))
        else:
            raise ValueError('Unmatched mode: {}'.format(mode))

        return sample

    def __len__(self):
        return len(self.samples)


class SlideOSCC(data.Dataset):
    """
    input and label image dataset
    """
    
    def __init__(self, 
                root, 
                meta_file,
                mode=1,
                label=False,
                data_suffix='subslide/val',
                target_suffix='subslide/target_val', 
                slide_suffix='slide/target_val_x8',
                transform=None):
        super(SlideOSCC, self).__init__()
        """
        Args:
            root -->
            transform -->
        """
        self.root = root
        self.args = args
        self.mode = mode
        self.label = label
        self.transform = transform
        self.classdict = {1: "normal", 2: "mucosa", 3: "tumor", 0: "background"}

        self.root_img = os.path.join(root, data_suffix)
        if label:
            self.root_target = os.path.join(root, target_suffix)
            self.root_slide = os.path.join(root, slide_suffix)

        with open(meta_file, 'r') as f:
            cnt = json.load(f)

        # self.to_tensor = transforms.ToTensor()
        samples = []
        slide_info = {}
        slide_num = {}
        slide_name = []


        for slide in os.listdir(self.root_img):
            slide_name.append(slide)
            slide_info[slide] = cnt[slide]
            slide_dir = os.path.join(self.root_img, slide)
            slide_num[slide] = len(os.listdir(slide_dir))
            for patch in os.listdir(slide_dir):
                samples.append(os.path.join(slide_dir, patch))
        
        self.samples = samples
        self.slide_num = slide_num
        self.slide_info = slide_info
        self.slide_name = slide_name
    
    def __getitem__(self, index):
        img_path = self.samples[index]
        img = pil_loader(img_path).convert('RGB')
        target = None
        path_split = img_path.split('/')
        name = path_split[-1]
        if self.label:
            target_path = os.path.join(os.path.join(self.root_target, path_split[-2]), name)
            target = pil_loader(target_path).convert('L')

        if self.transform is not None:
            img, target = self.transform(img, target)
        sample = self.sample_generator(img, target, self.mode)
        sample['id'] = name
        
        return sample
    
    def get_slide_target(self, slide):
        target_path = os.path.join(self.root_slide, slide+'.png')
        target = pil_loader(target_path).convert('L')
        return np.asarray(target, dtype='uint8')
    
    def get_slide_num(self, slide):
        return self.slide_num[slide]

    def get_slide_size(self, slide):
        return self.slide_info[slide]['size']
    
    def get_slide_tiles(self, slide):
        return self.slide_info[slide]['tiles']
    
    def get_slide_step(self, slide):
        return self.slide_info[slide]['step']

    def get_slide_name(self, index):
        return self.slide_name[index]
    
    def sample_generator(self, img, target, mode):
        w, h = img.shape
        sample['output_size'] = (h//8, w//8)
        w = w//4; h = h//4
        if self.label:
            sample = {"label": target}

        if mode == 1:
            sample['image_g'] = resize(img, (w, h))
        elif mode == 2:
            sample['image_l'] = img
        elif mode == 3:
            sample['image_l'] = img
            sample['image_g'] = resize(img, (w, h))
        else:
            raise ValueError('Unmatched mode: {}'.format(mode))
        
        return sample

    def __len__(self):
        return len(self.samples)
    
    def len_slide(self):
        return len(self.slide_name)
    
    def parse_name(self, slide):
        name = slide.split('_')
        ver = int(name[-3])
        col = int(name[-2])
        return (ver, col)

