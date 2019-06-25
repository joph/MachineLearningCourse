# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 12:33:58 2019

@author: jschmidt
"""

import os
from keras import models
import imp
import scripts.windturbines.functions_pattern_recognition as fpr
imp.reload(fpr)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Get images from raw directory
# Convert to png in temp
# Predict
# If classified, copy

model = models.load_model('models/model-0099-0.04.h5')

# within this file, we assess new regions
COUNTRY = "CN"

raw_dir = fpr.get_param(COUNTRY, "PATH_RAW_IMAGES_ASSESSMENT")
temp_dir = fpr.get_param(COUNTRY, "PATH_TEMP")

dirs = os.listdir(raw_dir)



found_turbines = fpr.assess_windparks_country(raw_dir, dirs,
                                              temp_dir, model,
                                              threshold=0.5)

found_turbines_pd = pd.DataFrame(found_turbines)

plt.hist(found_turbines_pd.iloc[:,2])

found_turbines_pd.loc[(found_turbines_pd[0] == "121.111"), :]

baishi = pd.DataFrame(["121.11_40.9514",
          "121.104_40.964",
          "121.118_40.9682",
          "121.118_40.9696",
          "121.0984_40.9598",
          "121.0984_40.9612",
          "121.0998_40.9598",
          "121.0998_40.9612",
          "121.1012_40.9584",
          "121.1026_40.957",
          "121.1026_40.964",
          "121.1026_40.9584",
          "121.1068_40.9696",
          "121.1082_40.9696",
          "121.1096_40.9514"])

baishi_tif =  pd.DataFrame(baishi[0] + ".tif")

baishi_tif.columns = ["tifs"]

df = found_turbines_pd

df[df[3].isin(baishi_tif.tifs)]

## per park Baibuluo
found_turbines = fpr.assess_windparks_country(raw_dir, dirs,
                                              temp_dir, model,
                                              threshold=0.2, n = -3)

found_turbines_pd = pd.DataFrame(found_turbines)

plt.hist(found_turbines_pd.iloc[:,2])

found_turbines_pd.loc[(found_turbines_pd[0] == "121.111"), :]

baibuluo = pd.DataFrame(["121.11_40.9514",
          "121.104_40.964",
          "121.118_40.9682",
          "121.118_40.9696",
          "121.0984_40.9598",
          "121.0984_40.9612",
          "121.0998_40.9598",
          "121.0998_40.9612",
          "121.1012_40.9584",
          "121.1026_40.957",
          "121.1026_40.964",
          "121.1026_40.9584",
          "121.1068_40.9696",
          "121.1082_40.9696",
          "121.1096_40.9514"])

baibuluo_tif =  pd.DataFrame(baibuluo[0] + ".tif")

baibuluo_tif.columns = ["tifs"]

df = found_turbines_pd

df[df[3].isin(baibuluo_tif.tifs)]






# check image valid
#img_path = "data/aerialImages/Google/RESOLUTION19/AT/keras/train/Turbines/35.png"
#img_path = "data/aerialImages/Google/RESOLUTION19/CN/assessment/Anhui Laian Baoshan Wind/turbines/118.434_32.6796.png"
#fpr.check_image(img_path,model,"heatmap_turbine.png")

# check image in-valid
# img_path = "data/aerialImages/Google/RESOLUTION19/AT/keras/train/NoTurbines/1.png"
# check_image(img_path,model,"heatmap_no_turbine.png")
