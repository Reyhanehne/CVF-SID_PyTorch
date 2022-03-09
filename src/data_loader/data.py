import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import numpy as np
import time 
import os
import scipy.io
from scipy.ndimage.interpolation import rotate
from multiprocessing import Pool


def process_image(train_noisy):
    STD_train = []
    for h in range(3,train_noisy.shape[1]-3):
        for w in range(3,train_noisy.shape[2]-3):
            STD_train.append(np.std((train_noisy[:,h-3:h+3,w-3:w+3,:]/255).reshape([-1,36,3]),1).reshape([-1,1,3]))   
    return np.mean(np.concatenate(STD_train,1),1)

def horizontal_flip(image, rate=0.5):
    image = image[:, ::-1, :]
    return image


def vertical_flip(image, rate=0.5):
    image = image[::-1, :, :]
    return image

def random_rotation(image, angle):
    h, w, _ = image.shape
    image = rotate(image, angle)
    return image
    
class benchmark_data(Dataset):

    def __init__(self, data_dir, task, transform=None):

        self.task = task
        self.data_dir = data_dir
        files_tmp = open(self.data_dir+'Scene_Instances.txt','r').readlines()
        self.Validation_Gt = scipy.io.loadmat(self.data_dir+'ValidationGtBlocksSrgb.mat')['ValidationGtBlocksSrgb']
        self.Validation_Noisy = scipy.io.loadmat(self.data_dir+'ValidationNoisyBlocksSrgb.mat')['ValidationNoisyBlocksSrgb']
        self.Validation_Gt = self.Validation_Gt.reshape([-1,256,256,3])
        self.Validation_Noisy = self.Validation_Noisy.reshape([-1,256,256,3])
        self.data_num = self.Validation_Noisy.shape[0]
        self.files = []
        for i in range(160):
            f = files_tmp[i].split("\n")[0]
            #if f[-1]=='N':
            if i >=0:
               self.files.append(f)   
        self.indices = self._indices_generator()
        self.patch_size = 40
        
       
        

        
        if os.path.exists(self.data_dir+'/'+'std.npy'):
           STD = np.load(self.data_dir+'/'+'std.npy')
        else:
           STD = process_image(self.Validation_Noisy)
           np.save(self.data_dir+'/'+'std.npy',STD)
        self.std = STD
        
              

    def __len__(self):
        return self.data_num

    def __getitem__(self, index):

        def data_loader():
            
            if self.task=="test": 
               Img_noisy = self.Validation_Noisy[index]
               Img_GT = self.Validation_Gt[index]
               Img_noisy = (np.transpose(Img_noisy,(2, 0, 1))/255)
               Img_GT = (np.transpose(Img_GT,(2, 0, 1))/255)   
               std = self.std[index]


            if self.task=="train":
               Img_noisy = self.Validation_Noisy[index]
               Img_GT = self.Validation_Gt[index]
               
               # Augmentation
               horizontal = torch.randint(0,2, (1,))
               vertical = torch.randint(0,2, (1,))
               rand_rot = torch.randint(0,4, (1,))
               rot = [0,90,180,270]
               if horizontal ==1:
                  Img_noisy = horizontal_flip(Img_noisy)
                  Img_GT = horizontal_flip(Img_GT)
               if vertical ==1:
                 Img_noisy = vertical_flip(Img_noisy)
                 Img_GT = vertical_flip(Img_GT)        
               Img_noisy = random_rotation(Img_noisy,rot[rand_rot])
               Img_GT = random_rotation(Img_GT,rot[rand_rot])         
                 
               Img_noisy = (np.transpose(Img_noisy,(2, 0, 1))/255)
               Img_GT = (np.transpose(Img_GT,(2, 0, 1))/255)   
               std = self.std[index]
               x_00 = torch.randint(0, Img_noisy.shape[1] - self.patch_size, (1,))
               y_00 = torch.randint(0, Img_noisy.shape[2] - self.patch_size, (1,))
               Img_noisy = Img_noisy[:, x_00[0]:x_00[0] + self.patch_size, y_00[0]:y_00[0] + self.patch_size]
               Img_GT = Img_GT[:, x_00[0]:x_00[0] + self.patch_size, y_00[0]:y_00[0] + self.patch_size]

 


            
            return np.array(Img_noisy, dtype=np.float32), np.array(Img_GT, dtype=np.float32),  np.array(std, dtype=np.float32), index #,Img_train, Img_train_noisy, std_train[0]


        def _timeprint(isprint, name, prevtime):
            if isprint:
                print('loading {} takes {} secs'.format(name, time() - prevtime))
            return time()

        if torch.is_tensor(index):
            index = index.tolist()

        input_noisy, input_GT, std, idx = data_loader()
        target = {
                  'dir_idx': str(idx)
                  }

        return target, input_noisy, input_GT, std 

    def _indices_generator(self):

        return np.arange(self.data_num,dtype=int)
   
        

if __name__ == "__main__":
    time_print = True

    prev = time()
