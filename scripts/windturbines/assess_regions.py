# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 12:33:58 2019

@author: jschmidt
"""



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
import shutil

import timeit




from keras.preprocessing.image import ImageDataGenerator

os.chdir("G:/Meine Ablage/LVA/PhD Lectures/MachineLearningCourse")

import scripts.windturbines.functions_pattern_recognition as fpr
imp.reload(fpr)
from scripts.windturbines.functions_pattern_recognition import get_param
from scripts.windturbines.functions_pattern_recognition import cop_predict
from scripts.windturbines.functions_pattern_recognition import check_image
from scripts.windturbines.functions_pattern_recognition import read_params
from scripts.windturbines.functions_pattern_recognition import load
from scripts.windturbines.functions_pattern_recognition import assess_windparks_country

######get images from raw directory
######convert to png in temp
######predict
######if classified, copy

#model = models.load_model('models/unfreezed-model-0056-0.01.h5')
model = models.load_model('models/simple-model-027-0.988224-1.000000.h5')

#within this file, we assess new regions
COUNTRY = "FR"

raw_dir = get_param(COUNTRY,"PATH_RAW_IMAGES_ASSESSMENT")
temp_dir = get_param(COUNTRY,"PATH_TEMP")

dirs = os.listdir(raw_dir)

found_turbines = assess_windparks_country(raw_dir, dirs, temp_dir, model, threshold = 0.9)

####check image valid
img_path = "data/aerialImages/Google/RESOLUTION19/AT/keras/train/Turbines/35.png"
img_path = "data/aerialImages/Google/RESOLUTION19/CN/assessment/Anhui Laian Baoshan Wind/turbines/118.434_32.6796.png"
check_image(img_path,model,"heatmap_turbine.png")

####check image in-valid
#img_path = "data/aerialImages/Google/RESOLUTION19/AT/keras/train/NoTurbines/1.png"
#check_image(img_path,model,"heatmap_no_turbine.png")

