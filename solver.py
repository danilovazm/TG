import os
import wandb
import time
import numpy as np
import matplotlib
import math
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim

from AlexNet import AlexNet
from VGG import VGG16

# Melhorar código de treino e rever parâmetros


class Solver(object):
    def __init__(self, args, data_loader):
        self.mode = args.mode
        self.load_mode = args.load_mode
        self.data_loader = data_loader
        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.save_path = args.save_path
        self.multi_gpu = args.multi_gpu

        self.num_epochs = args.num_epochs
        self.print_iters = args.print_iters
        self.save_iters = args.save_iters
        self.test_iters = args.test_iters

        if args.net == 'Alex':
            self.model = AlexNet()
        elif args.net == 'VGG':
            self.model = VGG16()

        print(torch.cuda.is_available())

        if (self.multi_gpu) and (torch.cuda.device_count() > 1):
            print('Use {} GPUs'.format(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

        self.lr = args.lr
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)


    def save_model(self, iter_):
        f = os.path.join(self.save_path, 'VGG_RGB_{}_epoch.ckpt'.format(iter_))
        torch.save(self.model.state_dict(), f)

    # Modificar Path
    def load_model(self, iter_):
        f = os.path.join(self.save_path, 'CKPY_ABDOMEN/UNET_CHEST{}iter.ckpt'.format(iter_))
        if self.multi_gpu:
            state_d = OrderedDict()
            for k, v in torch.load(r"D:\RED-CNN\save\REDCNN_38epoch_direct.ckpt"):
                n = k[7:]
                state_d[n] = v
            self.model.load_state_dict(state_d)
        else:
            self.model.load_state_dict(torch.load(r"D:\RED-CNN\save\specific\Abdomen\RED_CNN97epoch.ckpt"))


    def train(self, args):
        
        wandb.init(project="TG", config=args)
        train_losses = []
        total_iters = 0
        start_time = time.time()
        wandb.watch(self.model)
        for epoch in range(1, self.num_epochs):
            self.model.train(True)
            #self.val()
            #losses = 0
            for iter_, (x, y) in enumerate(self.data_loader):
                total_iters += 1

                # add 1 channel
                #x = x.unsqueeze(0).float().to(self.device)
                x = x.float().to(self.device)  # Ensure x is float and moved to the correct device
                y = y.to(self.device)

                pred = self.model(x)
                loss = self.criterion(pred, y)
                self.model.zero_grad()
                self.optimizer.zero_grad()
                
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())


                if (iter_+1) == len(self.data_loader):
                    print("STEP [{}], EPOCH [{}/{}] \nLOSS: {:.8f}, TIME: {:.1f}s".format(total_iters, epoch, 
                                                                                                        self.num_epochs,  
                                                                                                        loss.item(), 
                                                                                                        time.time() - start_time))

                    #val_x, val_y, val_loss, val_pred, psnr, ssim = self.val()
                    print("-----------------")
                    wandb.log({"loss": loss.item(),"Learning Rate": self.lr})
                    #wandb.log({"loss": loss.item(),"Learning Rate": self.lr, "quarter" : wandb.Image(x) ,"result" : wandb.Image(pred), "final result": wandb.Image(result) ,"full" : wandb.Image(y),
                    #        "val loss": val_loss, "val psnr": psnr, "val ssim": ssim ,"val quarter": wandb.Image(val_x), "val result": wandb.Image(val_pred), "val final result": wandb.Image(val_result), "val full": wandb.Image(val_y), "global_step":(epoch-1)})
            self.save_model(epoch)
            np.save(os.path.join(self.save_path, 'loss_{}_epoch.npy'.format(epoch)), np.array(train_losses))

    def val(self):
        loss = 0
        i = 0
        psnr = 0
        ssim = 0
        print('validação')
        with torch.no_grad():
            for _ , x , y in enumerate(self.val_loader):
                x = x.unsqueeze(0).float().to(self.device)  

                pred = self.model(x)
                cost = self.criterion(pred, y)
                loss += cost.item()

            print(loss/(len(self.val_loader)-i))

            return x, y, loss/(len(self.val_loader) - i), pred          
