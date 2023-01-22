# BNN-SAM: Improving Generalization of Binary Object Detector by Seeking Flat Minima

Pytorch implementation of our paper "BNN-SAM: Improving Generalization of Binary Object Detector by Seeking Flat Minima".

## Tips：

Any problem, please contact the first author.

Our code is heavily based on MMDetection(https://github.com/open-mmlab/mmdetection).


## Environments：

- **Python 3.7**
- **MMDetection 2.x**
- **This repo uses: mmdet-v2.0 mmcv-0.5.6 cuda 10.1**


## VOC Results：

Pretrained model is here: [vgg_bireal]() [vgg_React]() 

## Notes:

- **Batch:sample_per_gpu ✖️ gpu_num**


| Architecture | Backbone  | Style   | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
|:------------:|:---------:|:-------:|:-------:|:--------:|:--------------:|:------:|:------:|:--------:|
| Faster R-CNN | R-50      | pytorch | 1x      | 2.6   | -          | 79.5  |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712/faster_rcnn_r50_fpn_1x_voc0712_20200624-c9895d40.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712/20200623_015208.log.json) |
| Retinanet    | R-50      | pytorch | 1x      | 2.1   | -          | 77.3  |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pascal_voc/retinanet_r50_fpn_1x_voc0712.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/pascal_voc/retinanet_r50_fpn_1x_voc0712/retinanet_r50_fpn_1x_voc0712_20200617-47cbdd0e.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/pascal_voc/retinanet_r50_fpn_1x_voc0712/retinanet_r50_fpn_1x_voc0712_20200616_014642.log.json) |

