#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  :   {FirstElfin}
@License :   (C) Copyright 2013-2020, {DHWL}
@Contact :   {2968793701@qq.com}
@Software:   PyCharm
@File    :   test.py
@Time    :   11/22/19 11:55 AM
"""
import os
import xml.etree.ElementTree as ET
import shutil

ann_filepath = '/home/ic611/../data/VOCdevkit/VOC2007/Annotations/'
img_filepath = '/home/ic611/../data/VOCdevkit/VOC2007/JPEGImages/'
img_savepath = '/home/ic611/workspace/hanhan/mmdetection/tools/convert_datasets/VOCdevkit/VOC2007/JPEGImages/'
ann_savepath = '/home/ic611/workspace/hanhan/mmdetection/tools/convert_datasets/VOCdevkit/VOC2007/Annotations/'


# ann_filepath = '/home/ic611/../data/VOCdevkit/VOC2012/Annotations/'
# img_filepath = '/home/ic611/../data/VOCdevkit/VOC2012/JPEGImages/'
# img_savepath = '/home/ic611/workspace/hanhan/mmdetection/tools/convert_datasets/VOCdevkit/VOC2012/JPEGImages/'
# ann_savepath = '/home/ic611/workspace/hanhan/mmdetection/tools/convert_datasets/VOCdevkit/VOC2012/Annotations/'



if not os.path.exists(img_savepath):
    os.mkdir(img_savepath)

if not os.path.exists(ann_savepath):
    os.mkdir(ann_savepath)

# classes = ['bicycle', 'bus']
classes = ['bicycle', 'bus', 'car', 'motorbike']


def save_annotation(file):

    tree = ET.parse(ann_filepath + '/' + file)
    root = tree.getroot()
    result = root.findall("object")
    bool_num = 0
    for obj in result:
        if obj.find("name").text not in classes:
            root.remove(obj)
        else:
            bool_num = 1
    if bool_num:
        tree.write(ann_savepath + file)
        return True
    else:
        return False

def save_txt(file):
    name_img = img_filepath + os.path.splitext(file)[0] 
    shutil.copy(name_img, img_savepath)

def save_images(file):
    name_img = img_filepath + os.path.splitext(file)[0] + ".jpg"
    shutil.copy(name_img, img_savepath)
    return True


if __name__ == '__main__':
    for f in os.listdir(ann_filepath):
        if save_annotation(f):
            save_images(f)