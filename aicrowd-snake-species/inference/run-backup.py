#!/usr/bin/env python
import random
import json
import numpy as np
import argparse
import base64

import time
import traceback

import glob
import os
import json



from efficientnet_pytorch import EfficientNet


# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[3]:


from fastai.script import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.distributed import *
from fastprogress import fastprogress
from torchvision.models import *
from torch.optim import Optimizer



from fastai.script import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.distributed import *
from fastprogress import fastprogress
from torchvision.models import *
from torch.optim import Optimizer

from shutil import copyfile

from fastai.vision import *
from fastai.metrics import accuracy
from fastai.basic_data import *
import pandas as pd
from torch import optim
import re
import json
import cv2
from fastai.callbacks.hooks import num_features_model
from torch.nn import L1Loss

import numpy as np
import torch
import pandas as pd
import random
import string



"""
Expected ENVIRONMENT Variables

* AICROWD_TEST_IMAGES_PATH : abs path to  folder containing all the test images
* AICROWD_PREDICTIONS_OUTPUT_PATH : path where you are supposed to write the output predictions.csv
"""

class CustomDataset(Dataset):
    def __init__(self, j, aug=None):
        self.j = j
        if aug is not None: aug = get_aug(aug)
        self.aug = aug
    
    def __getitem__(self, idx):
        item = j2anno(self.j[idx])
        if self.aug: item = self.aug(**item)
        im, bbox = item['image'], np.array(item['bboxes'][0])
        im, bbox = self.normalize_im(im), self.normalize_bbox(bbox)
        
        return im.transpose(2,0,1).astype(np.float32), bbox.astype(np.float32)
    
    def __len__(self):
        return len(self.j)
    
    def normalize_im(self, ary):
        return ((ary / 255 - imagenet_stats[0]) / imagenet_stats[1])
    
    def normalize_bbox(self, bbox):
        return bbox / SZ

class SnakeDetector(nn.Module):
    def __init__(self, arch=models.resnet18):
        super().__init__() 
        self.cnn = create_body(arch)
        self.head = create_head(num_features_model(self.cnn) * 2, 4)
        
    def forward(self, im):
        x = self.cnn(im)
        x = self.head(x)
        return x.sigmoid_()

def run():
    ########################################################################
    # Register Prediction Start
    ########################################################################
    

    ########################################################################
    # Gather Input and Output paths from environment variables
    ########################################################################
    

    ########################################################################
    # Gather Image Names
    ########################################################################
    

    ########################################################################
    # Do your magic here to train the model
    ########################################################################
    src = (ImageList.from_folder(path='fastai-data').split_by_rand_pct(0.0).label_from_folder())
    tfms = get_transforms(do_flip=True,flip_vert=False,max_rotate=10.0,max_zoom=1.1,max_lighting=0.2,max_warp=0.2,p_affine=0.75,p_lighting=0.75)
    data = (src.transform(tfms, size=360, resize_method=ResizeMethod.SQUISH)
        .databunch(bs=32,num_workers=0).normalize(imagenet_stats))   

    learn = Learner(data, SnakeDetector(arch=models.resnet50), loss_func=L1Loss())
    learn.split([learn.model.cnn[:6], learn.model.cnn[6:], learn.model.head])
    state_dict = torch.load('fastai-data/models/snake-detection-model.pth')
    learn.model.load_state_dict(state_dict['model'])
    if not os.path.exists('preprocessed-images'):
        os.makedirs('preprocessed-images') #directory to store files

    src_new = (ImageList.from_folder(path='data\\images').split_by_rand_pct(0.0).label_from_folder()) # images where test images are stored
    
    for filename in src_new.items: 
        try:
            print("reading filename")
            im = cv2.imread(f"{filename}", cv2.IMREAD_COLOR)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (360,360), interpolation = cv2.INTER_AREA)
            im_height, im_width, _ = im.shape
            print(str(filename))
            orig_im = cv2.imread(f"{filename}", cv2.IMREAD_COLOR)
            orig_im_height, orig_im_width, _ = orig_im.shape
            to_pred = open_image(filename)
            _,_,bbox=learn.predict(to_pred)
            im_original = cv2.imread(f"{filename}", cv2.IMREAD_COLOR)
            im_original = cv2.cvtColor(im_original, cv2.COLOR_BGR2RGB)
            im_original.shape
            im_original_width = im_original.shape[1]
            im_original_height = im_original.shape[0]
            bbox_new = bbox
            bbox_new[0] = bbox_new[0]*im_original_width 
            bbox_new[2]= bbox_new[2]*im_original_width
            bbox_new[1] = bbox_new[1]*im_original_height
            bbox_new[3] = bbox_new[3]*im_original_height
            x_min, y_min, x_max, y_max = map(int, bbox_new)
            #cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
            im_original = im_original[y_min:y_max,x_min:x_max]
            im_original = cv2.cvtColor(im_original,cv2.COLOR_BGR2RGB)
            filename_str = str(filename)
            to_save = filename_str.replace('data\\images','preprocessed-images')
            print(to_save)
            cv2.imwrite(to_save,im_original)
        except:
            pass
    del learn
    gc.collect()

    print("DONE preprocessing")
    model_name = 'efficientnet-b5'
    image_size = EfficientNet.get_image_size(model_name)
    print(image_size)

    model = EfficientNet.from_pretrained(model_name)
    defaults.device = torch.device('cuda')

    np.random.seed(13)

    src = (ImageList.from_folder(path='fastai-data').split_by_rand_pct(0.2).label_from_folder())

    object_detection_results_path = os.getcwd()+"\\result-of-snakes-object-detection" #this folder has the results of preprocessing, predicts where the snakes are and crops them. Look into porting this to evaluator as well

    src.add_test_folder(object_detection_results_path)
    print(object_detection_results_path)
    print(src)

    tfms=([rotate(degrees=(-90,90), p=0.8)],[])

    bs=32
    data = (src.transform(tfms, size=image_size, resize_method=ResizeMethod.SQUISH)
        .databunch(bs=bs,num_workers=0).normalize(imagenet_stats))
    print(data.num_workers)
    model.add_module('_fc',nn.Linear(2048, data.c)) #Replace the final layer of b5 model with number of classes
    loss_func =LabelSmoothingCrossEntropy() #following EfficientNet paper
    RMSprop = partial(torch.optim.RMSprop) #Following EfficientNet paper

    learn = Learner(data, model, loss_func=loss_func, opt_func=RMSprop, metrics=[accuracy,FBeta(beta=1,average='macro')])

    learn.split([[learn.model._conv_stem, learn.model._bn0, learn.model._blocks[:19]],
             [learn.model._blocks[19:],learn.model._conv_head], 
             [learn.model._bn1,learn.model._fc]]) #for differential learning

    learn.load('b5-seed-13-round-3') #best single model - 85.2 local cv. Try ensemble later. with 83.1 and 85.2 models

    ########################################################################
    # Generate Predictions
    ########################################################################
   
    print("loaded")
    preds,_ = learn.TTA(ds_type=DatasetType.Test) # Test time augmentation

    probs_seed_13 = np.exp(preds)/np.exp(preds).sum(1)[:,None]
    
    del learn
    gc.collect() #garbage collect
    print("finished seed-13")
    model = EfficientNet.from_pretrained(model_name)

    model.add_module('_fc',nn.Linear(2048, data.c)) 

    learn = Learner(data, model, loss_func=loss_func, opt_func=RMSprop, metrics=[accuracy,FBeta(beta=1,average='macro')]) #mew learner

    learn.split([[learn.model._conv_stem, learn.model._bn0, learn.model._blocks[:19]],
             [learn.model._blocks[19:],learn.model._conv_head], 
             [learn.model._bn1,learn.model._fc]]) #not needed, but not takin chances

    learn.load('b5-seed-15-round-7') #83.6, 83.1 localcv model
    print("finished loading seed 15")
    preds,_ = learn.TTA(ds_type=DatasetType.Test)

    probs_seed_15 = np.exp(preds)/np.exp(preds).sum(1)[:,None]

    probs = (probs_seed_13 + probs_seed_15)/2
    
    probs_np = probs.numpy()
    print("finished seed-13")

    df_test = pd.read_csv('given_submission_sample_file.csv',low_memory=False) #use the given sample file to replace probabilites with our model predictions, This way, no need to worry about corrupted images


    df_classes = pd.read_csv('class.csv',low_memory=False) #class mapping

    data_dict = df_classes.set_index('class_idx')['original_class'].to_dict() #for look up

    probs_np = probs.numpy()

    df_testfile_mapping = pd.DataFrame(columns=['filename','map']) 

    df_testfile_mapping['filename']=df_test['filename']

    for i,row in df_testfile_mapping.iterrows():
        row['map']=i

    data_dict_filename = df_testfile_mapping.set_index('filename')['map'].to_dict() # for lookup, returns the index where the filename is found in the given submission file


    i = 0
    for test_file in data.test_ds.items:
        filename = (os.path.basename(test_file))
        map_val = int(data_dict_filename[filename])    
        for c in range(0,45):
            df_test.loc[map_val,data_dict[int(data.classes[c].split("-")[1])]]=probs_np[i][c]
        i += 1

    #around 7 predictions causes assertion error, for now submit them as class-204

    for i,row in df_test.iterrows():
        sum_temp=row[1:46].sum()
        low_limit = 1-1e-6
        high_limit = 1+1e-6
        
        if not (sum_temp>= low_limit and sum_temp <= high_limit):
            for c in range(1,46):
                df_test.iloc[i,c]=0.
            df_test.loc[i,'thamnophis_sirtalis']=1.    



    df_test.to_csv('generated-ensemble-2.csv',index=False)




if __name__ == "__main__":
    try:
        print("Crowd-api")
        run()
    except Exception as e:
        error = traceback.format_exc()
        print(error)
