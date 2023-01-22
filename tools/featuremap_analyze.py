import argparse
import os
from pathlib import Path
from functools import partial
import cv2
import numpy as np
import sys
import os.path as osp

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

from mmcv import Config
from mmcv.runner import load_checkpoint

from mmdet.core import ( FeatureMapVis, show_tensor, imdenormalize, show_img, imwrite,
                           traverse_file_paths)
from mmdet.models import build_detector
from mmdet.datasets.builder import build_dataset
from mmdet.datasets.pipelines import Compose


import matplotlib.pyplot as plt

import torch

# import os
# import PySide2
# dirname = os.path.dirname(PySide2.__file__)
# plugin_path = os.path.join(dirname,'plugins','platforms')
# os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--img_dir', type=str, default='../demo', help='show img dir')
    # 显示预测结果
    parser.add_argument('--show', type=bool, default=True, help='show results')
    # 可视化图片保存路径
    parser.add_argument(
        '--output_dir', help='directory where painted images will be saved')
    args = parser.parse_args()
    return args


def forward(self, img, img_metas=None, return_loss=False, **kwargs):
    x = self.extract_feat(img)
    outs = self.bbox_head(x)
    return outs


def create_model(cfg, use_gpu=True):
    model = build_detector(cfg.model, train_cfg=None, test_cfg=None)

    # pretrained = torch.load(args.checkpoint,map_location="cuda:0")
    # baseline_key = list(pretrained['state_dict'].keys())
    
    # from collections import OrderedDict
    # baseline_state_dict = OrderedDict()
    # model_key = list(model.state_dict().keys())
    # # for i in range(len(model_key)):
    #     if model_key[i] == baseline_key[i]:
    #         baseline_state_dict[model_key[i]] = pretrained['state_dict'][baseline_key[i]]
    # torch.save(baseline_state_dict,'/home/ic611/workspace/hanhan/mmdetection/checkpoints/ssd300_voc0712_BiReal_Attention_71.6_convert.pth')
  

    load_checkpoint(model, args.checkpoint, map_location='cpu',strict=True)
    model.eval()
    if use_gpu:
        model = model.cuda()
    return model


def create_featuremap_vis(cfg, use_gpu=True, init_shape=(320, 320, 3)):
    model = create_model(cfg, use_gpu)
    model.forward = partial(forward, model) 
    featurevis = FeatureMapVis(model, use_gpu)
    featurevis.set_hook_style(init_shape[2], init_shape[:2])
    return featurevis


def _show_save_data(featurevis, img, img_orig, feature_indexs, filepath, is_show, output_dir):
    show_datas = []
    for feature_index in feature_indexs:
        feature_map = featurevis.run(img.copy(), feature_index=feature_index)[0]
        data = show_tensor(feature_index, feature_map[0], resize_hw=img.shape[:2], show_split=False, is_show=False)[0]
        
        am_data = cv2.addWeighted(data, 0.5, img_orig, 0.5, 0)
        show_datas.append(am_data)

    if output_dir is not None:
        filename = os.path.join(output_dir,
                                Path(filepath).name
                                )
        if len(show_datas) == 1:
            imwrite(show_datas[0], filename)
        else:
            for i in range(len(show_datas)):
                fname, suffix = os.path.splitext(filename)
                imwrite(show_datas[i], fname + '_{}'.format(str(i)) + suffix)
    if is_show:
        show_img(show_datas)


def show_featuremap_from_imgs(featurevis, feature_indexs, img_dir, mean, std, is_show, output_dir):
    if not isinstance(feature_indexs, (list, tuple)):
        feature_indexs = [feature_indexs]
    img_paths = traverse_file_paths(img_dir, 'jpg')
    for path in img_paths:
        data = dict(img_info=dict(filename=path), img_prefix=None)
        test_pipeline = Compose(cfg.data.test.pipeline)
        item = test_pipeline(data)
        img_tensor = item['img']
        img = img_tensor[0].cpu().numpy().transpose(1, 2, 0)  # 依然是归一化后的图片
        img_orig = imdenormalize(img, np.array(mean), np.array(std)).astype(np.uint8)
        _show_save_data(featurevis, img, img_orig, feature_indexs, path, is_show, output_dir)


def show_featuremap_from_datalayer(featurevis, feature_indexs, is_show, output_dir):
    if not isinstance(feature_indexs, (list, tuple)):
        feature_indexs = [feature_indexs]
    dataset = build_dataset(cfg.data.test)
    for item in dataset:
        img_tensor = item['img']
        img_metas = item['img_metas'][0].data
        filename = img_metas['filename']
        img_norm_cfg = img_metas['img_norm_cfg']
        img = img_tensor[0].cpu().numpy().transpose(1, 2, 0)  # 依然是归一化后的图片
        img_orig = imdenormalize(img, img_norm_cfg['mean'], img_norm_cfg['std']).astype(np.uint8)
        _show_save_data(featurevis, img, img_orig, feature_indexs, filename, is_show, output_dir)


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)

    use_gpu = False
    is_show = args.show
    init_shape = (320, 320, 3)  # 值不重要，只要前向一遍网络时候不报错即可
    # feature_index = [34,52,76,98, 120,142]  # 想看的特征图层索引(yolov3  218 214 210)
    # feature_index = [143,144,145,146,147,148,149,150] 
    # feature_index = [62,68,84,90,106,112,128,134]
    # feature_index = [84,90,95,98]#[62,68]  [106,112] [128,134]
    # feature_index = [84,90,62,68,106,112,128,134]
    # feature_index = [69,91,113,135]
    # feature_index = [36,38,40,42]

    # feature_index = [62,68,70,73,76,84,90,92,106,112,114,128,133,136]

    # # feature_index = [0,2,4,11,13]
    # # feature_index = [0,2]
    # feature_index = [143,144,145,146,147,148,149,150,151,152,153,154]


    '''
    ReAct
    '''
    # feature_index = [147,153,155,179,185,187,211,217,222,243,249,254]
    # feature_index = [147,153,158,179,185,190,211,217,222,243,249,254]

    
    # feature_index = [7,13,20,25,30]
    # # 7 means class feature mask
    # # 13 means loc feature mask
    # # 20 means the shared feature
    # # 25 means the class feature
    # # 30 means the loc feature

    # feature_index = [21,26]
    # feature_index = [86,92,106,111,116]# for ReAct_attention_s8_specific_3*3  cls_mask(86);loc_mask(92);share_feat(106);cls_feat(111);loc_feat(116)
    # 21 means the 3*3 conv for cls
    # 26 means the 3*3 conv for loc


    # feature_index = [7,13,20,25,30] # for ReAct_attention_specific_3*3  cls_mask(5);loc_mask(13); share_feat(20); cls feat(25);loc feat(30)
    feature_index = [53,59,66,71,76]




    featurevis = create_featuremap_vis(cfg, use_gpu, init_shape)
    # show_featuremap_from_datalayer(featurevis, feature_index, is_show, args.output_dir)

    

    mean = cfg.img_norm_cfg['mean']
    std = cfg.img_norm_cfg['std']
    show_featuremap_from_imgs(featurevis, feature_index, args.img_dir, mean, std, is_show, args.output_dir)