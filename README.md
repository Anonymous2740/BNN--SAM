# BNN-SAM: Improving Generalization of Binary Object Detector by Seeking Flat Minima

Pytorch implementation of our paper "BNN-SAM: Improving Generalization of Binary Object Detector by Seeking Flat Minima".


## Tips：

Any problem, please contact the first author. 

Our code is heavily based on MMDetection(https://github.com/open-mmlab/mmdetection).


## Environments：

- **Python 3.7**
- **MMDetection 2.x**
- **This repo uses: mmdet-v2.8.0 mmcv-1.2.7 cuda 11.1**


## How to train?

### 1. Training on a single GPU

python tools/train.py {CONFIG_FILE} [optional arguments]

### 2. Training on multiple GPUs



bash ./tools/dist_train.sh 
    {CONFIG_FILE} 
    {GPU_NUM} 
    [optional arguments]

## VOC Results：

Pretrained model is here: [VGG_BiReal](https://www.dropbox.com/s/nbf0o2h710bde91/vgg16_BiReal.pt?dl=0) [VGG_ReAct](https://www.dropbox.com/s/7pyjwgti958yjh3/vgg16_ReAct.pt?dl=0) 

| Method | Backbone  | W/A(bit) | Lr schd | Mem (MB) | OPs(x $10^9$) | mAP | Config | Download |
|:------------:|:---------:|:-------:|:-------:|:--------:|:--------------:|:------:|:------:|:--------:|
| BNN-SAM(BiReal) | VGG_BiReal| 1/1 | 1x      | 21.88  | 3.22      | 70.6  |[config](https://www.dropbox.com/s/s8fe1lkuc0wjzfr/ssd300_voc0712_SAM_BNN%28BiReal%29.py?dl=0) | [model](https://www.dropbox.com/s/zs37hxcn2lp5pdn/ssd300_voc0712_SAM_BNN%28BiReal%29_70.6.pth?dl=0) &#124; [log](https://www.dropbox.com/s/3rdnsof6dxxtglp/20220511_102220.log?dl=0) |
| BNN-SAM(ReAct) | VGG_ReAct|1/1 | 1x      | 21.88   | 3.22         | 72.5  |[config](https://www.dropbox.com/s/zl22tnh2hza0fur/ssd300_voc0712_SAM_BNN%28ReAct%29.py?dl=0) | [model](https://www.dropbox.com/s/g6fh3f4a5bpshxn/ssd300_voc0712_SAM_BNN%28ReAct%29_72.5.pth?dl=0) &#124; [log](https://www.dropbox.com/s/tiepx60mha5bcar/20220917_171443.log?dl=0) |

