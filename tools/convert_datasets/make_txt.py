
import os
import random
 
trainval_percent = 0.3  #训练集和验证集
train_percent = 0.5     #训练集
xmlfilepath = '/home/ic611/workspace/hanhan/mmdetection/tools/convert_datasets/VOCdevkit/VOC2007/Annotations'
txtsavepath = '/home/ic611/workspace/hanhan/mmdetection/tools/convert_datasets/VOCdevkit/VOC2007/ImageSets/Main'

# xmlfilepath = '/home/ic611/workspace/hanhan/mmdetection/tools/convert_datasets/VOCdevkit/VOC2012/Annotations'
# txtsavepath = '/home/ic611/workspace/hanhan/mmdetection/tools/convert_datasets/VOCdevkit/VOC2012/ImageSets/Main'


total_xml = os.listdir(xmlfilepath)
 
num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
te = int(tv)
trainval = random.sample(list, tv)#在指定长度中随机输出长度为tv的数据
train = random.sample(trainval, tr)
test = random.sample(trainval,te)
 
ftrainval = open(txtsavepath + '/trainval.txt', 'w')
ftest = open(txtsavepath + '/test.txt', 'w')
ftrain = open(txtsavepath + '/train.txt', 'w')
fval = open(txtsavepath + '/val.txt', 'w')
 
for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        for i in test:
        #ftest.write(name)
            ftest.write(name)
 
ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
print("well finished!")