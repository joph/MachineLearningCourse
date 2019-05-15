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

# Get images from raw directory
# Convert to png in temp
# Predict
# If classified, copy

model = models.load_model('models/simple-model-027-0.988224-1.000000.h5')

# within this file, we assess new regions
COUNTRY = "FR"

raw_dir = fpr.get_param(COUNTRY, "PATH_RAW_IMAGES_ASSESSMENT")
temp_dir = fpr.get_param(COUNTRY, "PATH_TEMP")

dirs = os.listdir(raw_dir)

found_turbines = fpr.assess_windparks_country(raw_dir, dirs,
                                              temp_dir, model,
                                              threshold=0.9)

# check image valid
#img_path = "data/aerialImages/Google/RESOLUTION19/AT/keras/train/Turbines/35.png"
#img_path = "data/aerialImages/Google/RESOLUTION19/CN/assessment/Anhui Laian Baoshan Wind/turbines/118.434_32.6796.png"
#fpr.check_image(img_path,model,"heatmap_turbine.png")

# check image in-valid
# img_path = "data/aerialImages/Google/RESOLUTION19/AT/keras/train/NoTurbines/1.png"
# check_image(img_path,model,"heatmap_no_turbine.png")
