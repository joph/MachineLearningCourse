# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 17:07:03 2019

@author: jschmidt
"""

import os

from keras import layers
from keras import models
from keras import optimizers
from shutil import copyfile
import pandas as pd
import io
from keras.preprocessing import image
import numpy as np
import gdal
from PIL import Image
import numpy as np
from skimage import transform

from matplotlib.pyplot import imshow

import imp




from keras.preprocessing.image import ImageDataGenerator

os.chdir("G:/Meine Ablage/LVA/PhD Lectures/MachineLearningCourse")

import scripts.windturbines.functions_pattern_recognition as fpr
imp.reload(fpr)
from scripts.windturbines.functions_pattern_recognition import get_param
from scripts.windturbines.functions_pattern_recognition import cop_predict
from scripts.windturbines.functions_pattern_recognition import check_image
from scripts.windturbines.functions_pattern_recognition import read_params

######get images from raw directory
######convert to png in temp
######predict
######if classified, copy

model = models.load_model('models/unfreezed-model-0056-0.01.h5')

list_errors = []

C = fpr.COUNTRIES
C = ['BR']

for COUNTRY in C:
    cnt = 0
    raw_dir = get_param(COUNTRY,"PATH_RAW_IMAGES_TURBINES")
    dest_dir = get_param(COUNTRY,"PATH_RAW_IMAGES_TURBINES_MACHINE_CLASSIFIED") 
    temp_dir = get_param(COUNTRY,"PATH_TEMP")

    files = [x for x in os.listdir(raw_dir) if x.endswith(".tif")]

    errors = files.copy()

    for f in files:
        #print(cnt)
    
        res = cop_predict(f,
            0.9,
            raw_dir,
            temp_dir,
            dest_dir,
            model)
    
        errors[cnt] = res
        cnt = cnt + 1
        
    list_errors.append(errors)
    
### AT under construction 558,559,561.png
### AT wrong 1041
COUNTRY = "AT"
wrong = get_param(COUNTRY,"PATH_RAW_IMAGES_TURBINES_MACHINE_CLASSIFIED") + "559.png"
check_image(wrong, model)
    
COUNTRY = "AT"
wrong = get_param(COUNTRY,"PATH_RAW_IMAGES_TURBINES_MACHINE_CLASSIFIED") + "1041.png"
check_image(wrong, model)


    
### BR wrongly classified 7654.png
COUNTRY = "BR"
wrong = get_param(COUNTRY,"PATH_RAW_IMAGES_TURBINES_MACHINE_CLASSIFIED") + "7654.png"
check_image(wrong, model)