# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 20:30:48 2019

@author: jschmidt
"""

import os, shutil, random

import pandas as pd
import sys
import gdal
import imp


os.chdir("G:/Meine Ablage/LVA/PhD Lectures/MachineLearningCourse")

import scripts.windturbines.functions_pattern_recognition as fpr
imp.reload(fpr)
from scripts.windturbines.functions_pattern_recognition import get_param
from scripts.windturbines.functions_pattern_recognition import cop_predict
from scripts.windturbines.functions_pattern_recognition import check_image
from scripts.windturbines.functions_pattern_recognition import read_params


# The path to the directory where the original
# dataset was uncompressed


#t_base_dir='data/windturbines/train'
#v_base_dir='data/windturbines/validation'
COUNTRY = "MIX"

train_dir = get_param(COUNTRY,"PATH_ML_IMAGES_TURBINES_TRAIN")
test_dir = get_param(COUNTRY,"PATH_ML_IMAGES_TURBINES_TEST")
validation_dir = get_param(COUNTRY,"PATH_ML_IMAGES_TURBINES_VALIDATION")

train_no_dir = get_param(COUNTRY,"PATH_ML_IMAGES_NOTURBINES_TRAIN")
test_no_dir = get_param(COUNTRY,"PATH_ML_IMAGES_NOTURBINES_TEST")
validation_no_dir = get_param(COUNTRY,"PATH_ML_IMAGES_NOTURBINES_VALIDATION")


#### delete directories if exist
#### create if not exist

shutil.rmtree(train_dir,ignore_errors=True)
shutil.rmtree(test_dir,ignore_errors=True)
shutil.rmtree(validation_dir,ignore_errors=True)
shutil.rmtree(train_no_dir,ignore_errors=True)
shutil.rmtree(test_no_dir,ignore_errors=True)
shutil.rmtree(validation_no_dir,ignore_errors=True)

os.makedirs(train_dir)
os.makedirs(test_dir)
os.makedirs(validation_dir)
os.makedirs(train_no_dir)
os.makedirs(test_no_dir)
os.makedirs(validation_no_dir)

src_dir_tb = get_param(COUNTRY,"PATH_RAW_IMAGES_TURBINES")
src_dir_notb = get_param(COUNTRY,"PATH_RAW_IMAGES_NOTURBINES")


#### copy turbine images
#### select subset depending on quality check file
quality_check = pd.read_csv(get_param(COUNTRY,"FILE_QUALITY_CHECK")).dropna(subset=["quality"])

quality_check_sub = quality_check.loc[quality_check['quality']>=90]

nmbfiles = quality_check_sub.shape[0]

cnt = 0

####################TODO: SELECT SUBSECTION DEPENDING ON QUALITY CHECK

share_train=round(0.7*nmbfiles)
share_validate=round(0.85*nmbfiles)

for i in range(0,nmbfiles-1):
    cnt+=1
    print(cnt)
    #file=str(int(quality_check_sub.iloc[i,:].loc["id"]))
    file=quality_check_sub.iloc[i,:].loc["id"]
    
    #print(file)
    src_=os.path.join(src_dir_tb,file)
    print(src_)
    if(os.path.isfile(src_)):
        if(cnt<share_train):
            print("train")
            src=os.path.join(src_dir_tb,file)
            dst=os.path.join(train_dir,file)
            shutil.copyfile(src,dst)
        if(cnt>share_train and cnt<share_validate):
            print("validation")
            src=os.path.join(src_dir_tb,file)
            dst=os.path.join(validation_dir,file)
            shutil.copyfile(src,dst)
        if(cnt>share_validate):
            print("test")
            src=os.path.join(src_dir_tb,file)
            dst=os.path.join(test_dir,file)
            shutil.copyfile(src,dst)
            
    #### copy no-turbine images

files = [x for x in os.listdir(src_dir_notb) if x.endswith(".png")]

nmbfiles=len(files)
share_train=round(0.7*nmbfiles)
share_validate=round(0.85*nmbfiles)

cnt = 0
for file in files:
    cnt+=1
    #file=file[0:-4]
    print(file)

    if(cnt<share_train):
        print("train")
        src=os.path.join(src_dir_notb,file)
        dst=os.path.join(train_no_dir,file)
        shutil.copyfile(src,dst)
            
    if(cnt>share_train and cnt<share_validate):
        print("validation")
        src=os.path.join(src_dir_notb,file)
        dst=os.path.join(validation_no_dir,file)
        shutil.copyfile(src,dst)
            
    if(cnt>share_validate):
        print("test")
        src=os.path.join(src_dir_notb,file)
        dst=os.path.join(test_no_dir,file)
        shutil.copyfile(src,dst)
            



print('total training turbine images:', len(os.listdir(train_dir)))
print('total testing turbine images:', len(os.listdir(test_dir)))
print('total validation turbine images:', len(os.listdir(validation_dir)))
print('total training no-turbine images:', len(os.listdir(train_no_dir)))
print('total testing no-turbine images:', len(os.listdir(test_no_dir)))
print('total validation no-turbine images:', len(os.listdir(validation_no_dir)))