# CVF-SID_PyTorch

This repository contains the official code to reproduce the results from the paper:

**CVF-SID: Cyclic multi-Variate Function for Self-Supervised Image Denoising by Disentangling  Noise from Image**

\[[arXiv]()\] \[[presentation]()\] 


## Installation
Clone this repository into any place you want.
```
git clone https://github.com/Reyhanehne/CVF-SID_PyTorch.git
cd CVF-SID_PyTorch
```
### Dependencies
* Python 3.8.5
* PyTorch 1.7.1
* numpy
* Pillow
* torchvision
* scipy


## Expriments
### Reults of SIDD validation dataset

#### Pretrained model
Download \[[`config.json`]()\] and \[[`checkpoint-epoch#.pth`]()\] from the links and save them in `models/CVF_SID/SIDD_Val/` folder.
#### NOTE: The pretrained model is updated at March. 9th 2022. 

You can now go to src folder and test our CVF-SID by:
```
python test.py --device 0 --config ../models/CVF_SID/SIDD_Val/config.json --resume ../models/CVF_SID/SIDD_Val/model_best.pth
```

or you can train it by yourself as follows:
```
python train.py --device 0 --config config_SIDD_Val.json --tag SIDD_Val
```


## Citation
If you find our code or paper useful, please consider citing
