from __future__ import absolute_import, division, print_function

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dataset.dataset_oscc import DeepOSCC, SlideOSCC, class_to_RGB, is_image_file
from dataset.transformer import Transformer, ValTransformer, TestTransformer
from utils.loss import CrossEntropyLoss2d, SoftCrossEntropyLoss2d, FocalLoss
from utils.lovasz_losses import lovasz_softmax
from utils.metrics import AverageMeter
from utils.lr_scheduler import LR_Scheduler
from helper import create_model_load_weights, get_optimizer, Trainer, Evaluator, collate
from option import Options


args = Options().parse()
n_class = args.n_class

# torch.cuda.synchronize()
# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

data_path = args.data_path
meta_path = args.meta_path

model_path = args.model_path
if not os.path.isdir(model_path): 
    os.makedirs(model_path)


log_path = args.log_path
if not os.path.isdir(log_path): 
    os.makedirs(log_path)

task_name = args.task_name
print(task_name)
###################################

mode = args.mode # 1: train global; 2: train local ; 3: train global & local
evaluation = args.evaluation
test = evaluation and False
print("mode:", mode, "evaluation:", evaluation, "test:", test)

###################################
print("preparing datasets and dataloaders......")
batch_size = args.batch_size

data_time = AverageMeter("DataTime", ':6.3f')
batch_time = AverageMeter("BatchTime", ':6.3f')

transformer_train = Transformer(args.size_crop)
dataset_train = DeepOSCC(data_path, meta_path, args, mode=mode, val=False, transform=transformer_train)
dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, num_workers=4, batch_size=batch_size, collate_fn=collate, shuffle=True, pin_memory=True)
transformer_val = ValTransformer(args.size_crop)
dataset_val = DeepOSCC(data_path, meta_path, args, mode=mode, val=True, transform=transformer_val)
dataloader_val = torch.utils.data.DataLoader(dataset=dataset_val, num_workers=4, batch_size=batch_size, collate_fn=collate, shuffle=False, pin_memory=True)

###################################
print("creating models......")

path_g = os.path.join(model_path, args.path_g)
path_l = os.path.join(model_path, args.path_l)
path_gl = os.path.join(model_path, args.path_gl)
model = create_model_load_weights(n_class, mode, evaluation, path_g=path_g, path_l=path_l, path=path_gl)

###################################
num_epochs = args.num_epochs
learning_rate = args.lr
fmreg = args.fmreg

optimizer = get_optimizer(model, mode, learning_rate=learning_rate)

scheduler = LR_Scheduler('poly', learning_rate, num_epochs, len(dataloader_train))
##################################

criterion1 = FocalLoss(gamma=3)
criterion2 = nn.CrossEntropyLoss()
criterion3 = lovasz_softmax
criterion = lambda x,y: criterion1(x, y)
# criterion = lambda x,y: 0.5*criterion1(x, y) + 0.5*criterion3(x, y)
mse = nn.MSELoss()

if not evaluation:
    writer = SummaryWriter(log_dir=log_path + task_name)
    f_log = open(log_path + task_name + ".log", 'w')

trainer = Trainer(criterion, optimizer, n_class, mode=mode, fmreg=fmreg)
evaluator = Evaluator(n_class, mode=mode, test=test)

best_pred = 0.0
print("start training......")
for epoch in range(num_epochs):
    optimizer.zero_grad()
    tbar = tqdm(dataloader_train)
    train_loss = 0

    start_time = time.time()
    for i_batch, sample_batched in enumerate(tbar):
        print(i_batch)
        data_time.update(time.time()-start_time)
        if evaluation:  # evaluation pattern: no training
            break
        scheduler(optimizer, i_batch, epoch, best_pred)
        loss = trainer.train(sample_batched, model)
        train_loss += loss.item()
        score_train, score_train_global, score_train_local = trainer.get_scores()

        batch_time.update(time.time()-start_time)
        start_time = time.time()
        if mode == 1: 
            tbar.set_description('Train loss: %.3f; global mIoU: %.3f; data time: %.3f; batch time: %.3f' % 
                (train_loss / (i_batch + 1), np.mean(np.nan_to_num(score_train_global["iou"][1:])), data_time.avg, batch_time.avg))
        elif mode == 2: 
            tbar.set_description('Train loss: %.3f; local mIoU: %.3f; data time: %.3f; batch time: %.3f' % 
                (train_loss / (i_batch + 1), np.mean(np.nan_to_num(score_train_local["iou"][1:])), data_time.avg, batch_time.avg))
        else:
            tbar.set_description('Train loss: %.3f; agg mIoU: %.3f; data time: %.3f; batch time: %.3f' % 
                (train_loss / (i_batch + 1), np.mean(np.nan_to_num(score_train["iou"][1:])), data_time.avg, batch_time.avg))

    writer.add_scalar('loss', train_loss/len(tbar), epoch)

    trainer.reset_metrics()
    data_time.reset()
    batch_time.reset()

    if epoch % 1 == 0:
        with torch.no_grad():
            model.eval()
            print("evaluating...")

            if test: 
                tbar = tqdm(dataloader_test)
            else: 
                tbar = tqdm(dataloader_val)
            
            start_time = time.time()
            for i_batch, sample_batched in enumerate(tbar):
                data_time.update(time.time()-start_time)
                predictions, predictions_global, predictions_local = evaluator.eval_test(sample_batched, model)
                score_val, score_val_global, score_val_local = evaluator.get_scores()
                # use [1:] since class0 is not considered in deep_globe metric
                batch_time.update(time.time()-start_time)

                if mode == 1: 
                    tbar.set_description('global mIoU: %.3f; data time: %.3f; batch time: %.3f' % 
                        (np.mean(np.nan_to_num(score_val_global["iou"])[1:]), data_time.avg, batch_time.avg))
                elif mode == 2: 
                    tbar.set_description('local mIoU: %.3f; data time: %.3f; batch time: %.3f' % 
                        (np.mean(np.nan_to_num(score_val_local["iou"])[1:]), data_time.avg, batch_time.avg))
                else: 
                    tbar.set_description('agg mIoU: %.3f; data time: %.3f; batch time: %.3f' % 
                    (np.mean(np.nan_to_num(score_val["iou"])[1:]), data_time.avg, batch_time.avg))
                
                if not test: # has label
                    labels = sample_batched['label'] # PIL images

                if test: # save predictions
                    if not os.path.isdir("./prediction/"): os.mkdir("./prediction/")
                    for i in range(len(sample_batched['id'])):
                        if mode == 1:
                            transforms.functional.to_pil_image(class_to_RGB(predictions_global[i]) * 255.).save("./prediction/" + sample_batched['id'][i] + "_mask.png")
                        elif mode == 2:
                            transforms.functional.to_pil_image(class_to_RGB(predictions_local[i]) * 255.).save("./prediction/" + sample_batched['id'][i] + "_mask.png")
                        else:
                            transforms.functional.to_pil_image(class_to_RGB(predictions[i]) * 255.).save("./prediction/" + sample_batched['id'][i] + "_mask.png")

                if not evaluation and not test: # train:val
                    if i_batch * batch_size + len(sample_batched['id']) > (epoch % len(dataloader_val)) and i_batch * batch_size <= (epoch % len(dataloader_val)):
                        # writer.add_image('image', transforms.ToTensor()(images[(epoch % len(dataloader_val)) - i_batch * batch_size]), epoch)
                        if not test:
                            writer.add_image('mask', class_to_RGB(np.array(labels[(epoch % len(dataloader_val)) - i_batch * batch_size])) * 255., epoch)
                        if mode == 1:
                            writer.add_image('prediction_global', class_to_RGB(predictions_global[(epoch % len(dataloader_val)) - i_batch * batch_size]) * 255., epoch)
                        elif mode == 2:
                            writer.add_image('prediction_local', class_to_RGB(predictions_local[(epoch % len(dataloader_val)) - i_batch * batch_size]) * 255., epoch)
                        else:
                            writer.add_image('prediction', class_to_RGB(predictions[(epoch % len(dataloader_val)) - i_batch * batch_size]) * 255., epoch)

                start_time = time.time()
            
            data_time.reset()
            batch_time.reset()

            if not (test or evaluation): 
                if epoch // 2:
                    torch.save(model.state_dict(), "./saved_models/" + task_name + ".epoch" + str(epoch) + ".pth")

            if test:  # one epoch
                break
            else: # val log results
                score_val, score_val_global, score_val_local = evaluator.get_scores()
                evaluator.reset_metrics()
                if mode == 1:
                    if np.mean(np.nan_to_num(score_val_global["iou"][1:])) > best_pred: 
                        best_pred = np.mean(np.nan_to_num(score_val_global["iou"][1:]))
                elif mode == 2:
                    if np.mean(np.nan_to_num(score_val_local["iou"][1:])) > best_pred: 
                        best_pred = np.mean(np.nan_to_num(score_val_local["iou"][1:]))
                else:
                    if np.mean(np.nan_to_num(score_val["iou"][1:])) > best_pred:
                         best_pred = np.mean(np.nan_to_num(score_val["iou"][1:]))
                log = ""
                if mode == 1:
                    log = log + 'epoch [{}/{}] Global -- IoU: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, np.mean(np.nan_to_num(score_train_global["iou"][1:])), np.mean(np.nan_to_num(score_val_global["iou"][1:]))) + "\n"
                    log = log + "Global train:" + str(score_train_global["iou"]) + "\n"
                    log = log + "Global val:" + str(score_val_global["iou"]) + "\n"
                elif mode == 2:
                    log = log + 'epoch [{}/{}] Local  -- IoU: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, np.mean(np.nan_to_num(score_train_local["iou"][1:])), np.mean(np.nan_to_num(score_val_local["iou"][1:]))) + "\n"
                    log = log + "Local train:" + str(score_train_local["iou"]) + "\n"
                    log = log + "Local val:" + str(score_val_local["iou"]) + "\n"
                else:
                    log = log + 'epoch [{}/{}] IoU: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, np.mean(np.nan_to_num(score_train["iou"][1:])), np.mean(np.nan_to_num(score_val["iou"][1:]))) + "\n"
                    log = log + 'epoch [{}/{}] Global -- IoU: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, np.mean(np.nan_to_num(score_train_global["iou"][1:])), np.mean(np.nan_to_num(score_val_global["iou"][1:]))) + "\n"
                    log = log + 'epoch [{}/{}] Local  -- IoU: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, np.mean(np.nan_to_num(score_train_local["iou"][1:])), np.mean(np.nan_to_num(score_val_local["iou"][1:]))) + "\n"
                    log = log + "train: " + str(score_train["iou"]) + "\n"
                    log = log + "val:" + str(score_val["iou"]) + "\n"
                    log = log + "Global train:" + str(score_train_global["iou"]) + "\n"
                    log = log + "Global val:" + str(score_val_global["iou"]) + "\n"
                    log = log + "Local train:" + str(score_train_local["iou"]) + "\n"
                    log = log + "Local val:" + str(score_val_local["iou"]) + "\n"
                
                log += "================================\n"
                print(log)
                if evaluation: break  # one peoch

                f_log.write(log)
                f_log.flush()
                if mode == 1:
                    writer.add_scalars('IoU', {'train iou': np.mean(np.nan_to_num(score_train_global["iou"][1:])), 'validation iou': np.mean(np.nan_to_num(score_val_global["iou"][1:]))}, epoch)
                else:
                    writer.add_scalars('IoU', {'train iou': np.mean(np.nan_to_num(score_train["iou"][1:])), 'validation iou': np.mean(np.nan_to_num(score_val["iou"][1:]))}, epoch)

if not evaluation: 
    f_log.close()












