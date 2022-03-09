import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from model.metric import psnr
from torchvision.utils import save_image
import torch.nn.functional as F
from torch import autograd
import os


patch_size = 256


def sgd(weight: torch.Tensor, grad: torch.Tensor, meta_lr) -> torch.Tensor:
    weight = weight - meta_lr * grad
    return weight
    
def padr(img):
    pad = 20
    pad_mod = 'reflect'
    img_pad = F.pad(input=img, pad=(pad,pad,pad,pad), mode=pad_mod)
    return img_pad
    
def padr_crop(img):
    pad = 20
    pad_mod = 'reflect'
    img = F.pad(input=img[:,:,pad:-pad,pad:-pad], pad=(pad,pad,pad,pad), mode=pad_mod)
    return img
    
class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader, test_data_loader,
                  lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer=optimizer, config=config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.test_data_loader = test_data_loader
        self.do_test = self.test_data_loader is not None
        self.do_test = True
        self.gamma = 1.0
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('Total_loss', writer=self.writer)
        self.test_metrics = MetricTracker('psnr', 'ssim', writer=self.writer)
        if os.path.isdir('../output')==False:
           os.makedirs('../output/')
        if os.path.isdir('../output/C')==False:
           os.makedirs('../output/C/')
        if os.path.isdir('../output/GT')==False:
           os.makedirs('../output/GT/')
        if os.path.isdir('../output/N_i')==False:
           os.makedirs('../output/N_i/')
        if os.path.isdir('../output/N_d')==False:
           os.makedirs('../output/N_d/')
        if os.path.isdir('../output/I')==False:
           os.makedirs('../output/I/')

    def _train_epoch(self, epoch):

        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (target, input_noisy, input_GT, std) in enumerate(self.data_loader):
            input_noisy = input_noisy.to(self.device)
            input_GT = input_GT.to(self.device)
            std = std.to(self.device)
            pad = 20
            input_noisy = padr(input_noisy)
            input_GT = padr(input_GT)

            self.optimizer.zero_grad()
            
            noise_w, noise_b, clean = self.model(input_noisy)
            noise_w1, noise_b1, clean1 = self.model(padr_crop((clean)))
            noise_w2, noise_b2, clean2 = self.model(padr_crop((clean+torch.pow(clean,self.gamma)*noise_w))) #1
            noise_w3, noise_b3, clean3 = self.model(padr_crop((noise_b)))

            

            noise_w4, noise_b4, clean4 = self.model(padr_crop((clean+torch.pow(clean,self.gamma)*noise_w-noise_b))) #2
            noise_w5, noise_b5, clean5 = self.model(padr_crop((clean-torch.pow(clean,self.gamma)*noise_w+noise_b))) #3
            noise_w6, noise_b6, clean6 = self.model(padr_crop((clean-torch.pow(clean,self.gamma)*noise_w-noise_b))) #4
            noise_w10, noise_b10, clean10 = self.model(padr_crop((clean+torch.pow(clean,self.gamma)*noise_w+noise_b))) #5

            noise_w7, noise_b7, clean7 = self.model(padr_crop((clean+noise_b))) #6
            noise_w8, noise_b8, clean8 = self.model(padr_crop((clean-noise_b))) #7
            noise_w9, noise_b9, clean9 = self.model(padr_crop((clean-torch.pow(clean,self.gamma)*noise_w))) #8

            
            input_noisy_pred = clean+torch.pow(clean,self.gamma)*noise_w+noise_b
            
            loss = self.criterion[0](input_noisy, input_noisy_pred, clean, clean1, clean2, clean3, noise_b, noise_b1, noise_b2, noise_b3, noise_w, noise_w1, noise_w2,std,self.gamma)
            
            loss_neg1 = self.criterion[1](clean, clean4, noise_w, noise_w4, noise_b, -noise_b4)
            loss_neg2 = self.criterion[1](clean, clean5, noise_w, -noise_w5, noise_b, noise_b5)
            loss_neg3 = self.criterion[1](clean, clean6, noise_w, -noise_w6, noise_b, -noise_b6)


            loss_neg4 = self.criterion[1](clean, clean7, torch.zeros_like(noise_w), noise_w7, noise_b, noise_b7)
            loss_neg5 = self.criterion[1](clean, clean8, torch.zeros_like(noise_w), noise_w8, noise_b, -noise_b8)
            loss_neg6 = self.criterion[1](clean, clean9, -noise_w, noise_w9, torch.zeros_like(noise_b), noise_b9)
            loss_neg7 = self.criterion[1](clean, clean10, noise_w, noise_w10, noise_b, noise_b10)    
                    
            loss_total = loss+.1*(loss_neg1+loss_neg2+loss_neg3+loss_neg4+loss_neg5+loss_neg6+loss_neg7)
            loss_total.backward()
            
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)


            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} TotalLoss: {:.6f}' .format(
                    epoch,
                    self._progress(batch_idx),
                    loss_total.item()
                ))


            if batch_idx == self.len_epoch:
                break

            del target
            del loss_total

        log = self.train_metrics.result()
        
        if self.do_test:
            if epoch>100 or epoch%10==0:
               test_log = self._test_epoch(epoch,save=False)
               log.update(**{'test_' + k: v for k, v in test_log.items()})


        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.writer.close()

        return log

    def _test_epoch(self, epoch,save=False):


        self.test_metrics.reset()

        #with torch.no_grad():   
        if save==True:
           os.makedirs('../output/C/'+str(epoch))
           os.makedirs('../output/N_d/'+str(epoch))
           os.makedirs('../output/N_i/'+str(epoch))
        for batch_idx, (target, input_noisy, input_GT, std) in enumerate(self.test_data_loader):
                input_noisy = input_noisy.to(self.device)
                input_GT = input_GT.to(self.device)
                pad = 20
                input_noisy = padr(input_noisy)
                input_GT = padr(input_GT)


                noise_w, noise_b, clean = self.model(input_noisy)

                size = [noise_b.shape[0],noise_b.shape[1],noise_b.shape[2]*noise_b.shape[3]]              
                noise_b_normal = (noise_b-torch.min(noise_b.view(size),-1)[0].unsqueeze(-1).unsqueeze(-1))/(torch.max(noise_b.view(size),-1)[0].unsqueeze(-1).unsqueeze(-1)-torch.min(noise_b.view(size),-1)[0].unsqueeze(-1).unsqueeze(-1))    
                noise_w_normal = (noise_w-torch.min(noise_w.view(size),-1)[0].unsqueeze(-1).unsqueeze(-1))/(torch.max(noise_w.view(size),-1)[0].unsqueeze(-1).unsqueeze(-1)-torch.min(noise_w.view(size),-1)[0].unsqueeze(-1).unsqueeze(-1)) 
                if save==True:
                    for i in range(input_noisy.shape[0]):
                        save_image(torch.clamp(clean[i,:,pad:-pad,pad:-pad],min=0,max=1).detach().cpu(), '../output/C/'+str(epoch)+'/'+target['dir_idx'][i]+'.PNG')
                        save_image(torch.clamp(input_GT[i,:,pad:-pad,pad:-pad],min=0,max=1).detach().cpu(), '../output/GT/' +target['dir_idx'][i]+'.PNG')
                        save_image(torch.clamp(noise_b_normal[i,:,pad:-pad,pad:-pad],min=0,max=1).detach().cpu(), '../output/N_i/'+str(epoch)+'/'+target['dir_idx'][i]+'.PNG')
                        save_image(torch.clamp(noise_w_normal[i,:,pad:-pad,pad:-pad],min=0,max=1).detach().cpu(), '../output/N_d/'+str(epoch)+'/'+target['dir_idx'][i]+'.PNG')
                        save_image(torch.clamp(input_noisy[i,:,pad:-pad,pad:-pad],min=0,max=1).detach().cpu(), '../output/I/' +target['dir_idx'][i]+'.PNG')
                

                
                self.writer.set_step((epoch - 1) * len(self.test_data_loader) + batch_idx, 'test')
                for met in self.metric_ftns:
                    if met.__name__=="psnr":
                       psnr = met(input_GT[:,:,pad:-pad,pad:-pad].to(self.device), torch.clamp(clean[:,:,pad:-pad,pad:-pad],min=0,max=1))
                       self.test_metrics.update('psnr', psnr)
                    elif met.__name__=="ssim":
                       self.test_metrics.update('ssim', met(input_GT[:,:,pad:-pad,pad:-pad].to(self.device), torch.clamp(clean[:,:,pad:-pad,pad:-pad],min=0,max=1)))
                self.writer.close()
              
                del target
 

        self.writer.close()
        return self.test_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

