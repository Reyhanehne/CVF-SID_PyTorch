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
    
class Test(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader, test_data_loader,
                  lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer=optimizer, config=config)
        self.config = config
        self.len_epoch = len_epoch
        self.test_data_loader = test_data_loader
        self.do_test = self.test_data_loader is not None
        self.do_test = True
        self.gamma = 1.0
        self.lr_scheduler = lr_scheduler


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
        
        log = self.train_metrics.result()
        self.writer.set_step(epoch)
        test_log = self._test_epoch(epoch,save=True)
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

