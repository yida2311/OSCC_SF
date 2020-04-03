from __future__ import absolute_import, division, print_function

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms

from models.global_local_ensemble import FPN_GL, get_fpn_global, get_fpn_local
from utils.metrics import ConfusionMatrix, AverageMeter
from PIL import Image

# torch.cuda.synchronize()
# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

transformer = transforms.Compose([
        transforms.ToTensor(),
])

def _mask_transform(mask):
    target = np.array(mask).astype('int32')
    target[target == 255] = -1
    # target -= 1 # in DeepGlobe: make class 0 (should be ignored) as -1 (to be ignored in cross_entropy)
    return target

def masks_transform(masks, numpy=False):
    '''
    masks: list of PIL images
    '''
    targets = []
    for m in masks:
        targets.append(_mask_transform(m))
    targets = np.array(targets)
    if numpy:
        return targets
    else:
        return torch.from_numpy(targets).long().cuda()

def images_transform(images):
    '''
    images: list of PIL images
    '''
    inputs = []
    for img in images:
        inputs.append(transformer(img))
    inputs = torch.stack(inputs, dim=0).cuda()
    return inputs

def one_hot_gaussian_blur(index, classes):
    '''
    index: numpy array b, h, w
    classes: int
    '''
    mask = np.transpose((np.arange(classes) == index[..., None]).astype(float), (0, 3, 1, 2))
    b, c, _, _ = mask.shape
    for i in range(b):
        for j in range(c):
            mask[i][j] = cv2.GaussianBlur(mask[i][j], (0, 0), 8)

    return mask

def collate(batch):
    batch_dict = {}
    for key in batch[0].keys():
        batch_dict[key] = [b[key] for b in batch]
    for key in batch_dict.keys():
        if not (key=='output_size' or key=='id'):
            batch_dict[key] = torch.stack(batch_dict[key], dim=0)

    return batch_dict


def create_model_load_weights(n_class, mode=1, evaluation=False, path_g=None, path_l=None, path=None, upsample='SemanticFlow'):
    if mode == 1:
        model = get_fpn_global(n_class, upsample, pretrained=evaluation, path=path_g)
    elif mode == 2:
        model = get_fpn_local(n_class, upsample, pretrained=evaluation, path=path_l)   
    elif mode == 3:
        model = FPN_GL(n_class, path_g=path_g, path_l=path_l)
        if evaluation and path:
            state = model.state_dict()
            state.update(torch.load(path))
            model.load_state_dict(state)
    model = nn.DataParallel(model)
    model = model.cuda()

    return model


def get_optimizer(model, mode=1, learning_rate=2e-5):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    
    return optimizer


class Trainer(object):
    def __init__(self, criterion, optimizer, n_class, mode=1, fmreg=0.15):
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics_global = ConfusionMatrix(n_class)
        self.metrics_local = ConfusionMatrix(n_class)
        self.metrics = ConfusionMatrix(n_class)
        self.n_class = n_class
        self.mode = mode
        self.fmreg = fmreg # regulization item

    def get_scores(self):
        score_train = self.metrics.get_scores()
        score_train_local = self.metrics_local.get_scores()
        score_train_global = self.metrics_global.get_scores()
        
        return score_train, score_train_global, score_train_local

    def reset_metrics(self):
        self.metrics.reset()
        self.metrics_local.reset()
        self.metrics_global.reset()  

    def train(self, sample, model):
        model.train()
        labels = sample['label'].squeeze(1).long()
        labels_npy = np.array(labels)
        labels_torch = labels.cuda()
        h, w = sample['output_size'][0]
        # print(labels[0].size)
        if self.mode == 1:  # global
            img_g = sample['image_g'].cuda()
            outputs_g = model.forward(img_g)
            outputs_g = F.interpolate(outputs_g, size=(h, w), mode='bilinear')
            # print(outputs_g.size(), labels_torch.size())
            loss = self.criterion(outputs_g, labels_torch)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        if self.mode == 2:  # local
            img_l = sample['image_l'].cuda()
            outputs_l = model.forward(img_l)
            outputs_l = F.interpolate(outputs_l, size=(h, w), mode='bilinear')
            loss = self.criterion(outputs_l, labels_torch)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        if self.mode == 3:  # global&local
            img_g = sample['image_g'].cuda()
            img_l = sample['image_l'].cuda()
            outputs, outputs_g, outputus_l, mse = model.forward(img_g, img_l, target=labels_torch)
            loss = 2* self.criterion(outputs, labels_torch) + self.criterion(outputs_g, labels_torch) + self.criterion(outputus_l, labels_torch) + self.fmreg * mse
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        # predictions
        if self.mode == 1:
            outputs_g = outputs_g.cpu()
            predictions_global = [outputs_g[i:i+1].argmax(1).detach().numpy() for i in range(len(labels))]
            self.metrics_global.update(labels_npy, predictions_global)
        
        if self.mode == 2:
            outputs_l = outputs_l.cpu()
            predictions_local = [outputs_l[i:i+1].argmax(1).detach().numpy() for i in range(len(labels))]
            self.metrics_local.update(labels_npy, predictions_local)
        
        if self.mode == 3:
            outputs_g = outputs_g.cpu(); outputs_l = outputs_l.cpu(); outputs = outputs.cpu()
            predictions_global = [outputs_g[i:i+1].argmax(1).detach().numpy() for i in range(len(labels))]
            predictions_local = [outputs_l[i:i+1].argmax(1).detach().numpy() for i in range(len(labels))]
            predictions = [outputs[i:i+1].argmax(1).detach().numpy() for i in range(len(labels))]
            self.metrics_global.update(labels_npy, predictions_global)
            self.metrics_local.update(labels_npy, predictions_local)
            self.metrics.update(labels_npy, predictions)

        return loss


class Evaluator(object):
    def __init__(self, n_class, mode=1, test=False):
        self.metrics_global = ConfusionMatrix(n_class)
        self.metrics_local = ConfusionMatrix(n_class)
        self.metrics = ConfusionMatrix(n_class)
        self.n_class = n_class
        self.mode = mode
        self.test = test

    def get_scores(self):
        score_train = self.metrics.get_scores()
        score_train_local = self.metrics_local.get_scores()
        score_train_global = self.metrics_global.get_scores()
        
        return score_train, score_train_global, score_train_local

    def reset_metrics(self):
        self.metrics.reset()
        self.metrics_local.reset()
        self.metrics_global.reset()  

    def eval_test(self, sample, model):
        with torch.no_grad():
            ids = sample['id']
            h, w = sample['output_size'][0]
            if not self.test:
                labels = sample['label'].squeeze(1).long()
                labels_npy = np.array(labels)

            if self.mode == 1:  # global
                img_g = sample['image_g'].cuda()
                outputs_g = model.forward(img_g)
                outputs_g = F.interpolate(outputs_g, size=(h, w), mode='bilinear')
        
            if self.mode == 2:  # local
                img_l = sample['image_l'].cuda()
                outputs_l = model.forward(img_l)
                outputs_l = F.interpolate(outputs_l, size=(h, w), mode='bilinear')
        
            if self.mode == 3:  # global&local
                img_g = sample['image_g'].cuda()
                img_l = sample['image_l'].cuda()
                # no target
                outputs, outputs_g, outputus_l, mse = model.forward(img_g, img_l)
        
        # predictions
        if self.mode == 1:
            outputs_g = outputs_g.cpu()
            predictions_global = [outputs_g[i:i+1].argmax(1).detach().numpy() for i in range(len(labels))]
            if not self.test:
                self.metrics_global.update(labels_npy, predictions_global)
            
            return None, predictions_global, None
        
        if self.mode == 2:
            outputs_l = outputs_l.cpu()
            predictions_local = [outputs_l[i:i+1].argmax(1).detach().numpy() for i in range(len(labels))]
            if not self.test:
                self.metrics_local.update(labels_npy, predictions_local)
            
            return None, None, predictions_local
        
        if self.mode == 3:
            outputs_g = outputs_g.cpu(); outputs_l = outputs_l.cpu(); outputs = outputs.cpu()
            predictions_global = [outputs_g[i:i+1].argmax(1).detach().numpy() for i in range(len(labels))]
            predictions_local = [outputs_l[i:i+1].argmax(1).detach().numpy() for i in range(len(labels))]
            predictions = [outputs[i:i+1].argmax(1).detach().numpy() for i in range(len(labels))]
            if not self.test:
                self.metrics_global.update(labels_npy, predictions_global)
                self.metrics_local.update(labels_npy, predictions_local)
                self.metrics.update(labels_npy, predictions)
            
            return predictions, predictions_global, predictions_local











