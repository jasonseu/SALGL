# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2021-7-15
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2021 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
from xml.dom.minidom import parse
import numpy as np
from collections import defaultdict


data_dir = 'datasets/VOC2007'
anno_dir = os.path.join(data_dir, 'Annotations')
image_dir = os.path.join(data_dir, 'JPEGImages')
save_dir = 'data/voc2007'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
def parse_xml(xml_file):
    dom_tree = parse(xml_file)
    root = dom_tree.documentElement
    objects = root.getElementsByTagName('object')
    data = defaultdict(list)
    for obj in objects:
        label = obj.getElementsByTagName('name')[0].firstChild.data.lower()
        difft = obj.getElementsByTagName('difficult')[0].firstChild.data
        data[label].append(int(difft))
    return data

num_images = [0, 0]
num_labels = [0, 0]

print('processing train data...')
train_txt = os.path.join(data_dir, 'ImageSets/Main/trainval.txt')
train_imgIds = [t.strip() for t in open(train_txt)]
train_data = []
for img_id in train_imgIds:
    image_path = os.path.join(image_dir, '{}.jpg'.format(img_id))
    if not os.path.exists(image_path):
        raise Exception('file {} not found!'.format(image_path))
    xml_path = os.path.join(anno_dir, '{}.xml'.format(img_id))
    data = parse_xml(xml_path)
    labels = [label for label, _ in data.items()]
    train_data.append([image_path, labels])

    num_images[0] += 1
    num_labels[0] += len(labels)
    
label_set = sorted(list(set([t for pair in train_data for t in pair[1]])))
train_data = ['{}\t{}\n'.format(t[0], ','.join(list(t[1]))) for t in train_data]
with open(os.path.join(save_dir, 'train.txt'), 'w') as fw:
    fw.writelines(train_data)
with open(os.path.join(save_dir, 'label.txt'), 'w') as fw:
    for line in label_set:
        fw.write(line+'\n')


print('processing test data...')
label2id = {l:i for i, l in enumerate(label_set)}
test_txt = os.path.join(data_dir, 'ImageSets/Main/test.txt')
test_imgIds = [t.strip() for t in open(test_txt)]
test_data, ignore = [], []
for img_id in test_imgIds:
    image_path = os.path.join(image_dir, '{}.jpg'.format(img_id))
    if not os.path.exists(image_path):
        raise Exception('file {} not found!'.format(image_path))
    xml_path = os.path.join(anno_dir, '{}.xml'.format(img_id))
    data = parse_xml(xml_path)
    labels = [label for label, diffts in data.items() if sum(diffts) != len(diffts)]
    test_data.append([image_path, labels])

    num_images[1] += 1
    num_labels[1] += len(labels)
    
    temp = [0] * len(label_set)
    for label, diffts in data.items():
        k = label2id[label]
        temp[k] = 1 if sum(diffts) == len(diffts) else 0
        # 1 means this data will be ignored when evaluates AP for the class k
        # 0 means this data will be considered.
    ignore.append(temp)

print('===== Summary of Pascal VOC 2007 =====')
print('Total number of samples: {}, number of labels per image: {}'.format(sum(num_images), sum(num_labels)/sum(num_images)))
print('Train number of samples: {}, number of labels per image: {}'.format(num_images[0], num_labels[0]/num_images[0]))
print('Test  number of samples: {}, number of labels per image: {}'.format(num_images[1], num_labels[1]/num_images[1]))
    
test_data = ['{}\t{}\n'.format(t[0], ','.join(list(t[1]))) for t in test_data]
with open(os.path.join(save_dir, 'test.txt'), 'w') as fw:
    fw.writelines(test_data)

ignore = np.array(ignore)
np.save(os.path.join(save_dir, 'ignore.npy'), ignore)