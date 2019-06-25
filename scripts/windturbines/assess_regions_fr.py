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
                                              threshold=0.05)

found_turbines_pd = pd.DataFrame(found_turbines)

plt.hist(found_turbines_pd.iloc[:, 2])


turbines = [
            "-0.1558_49.0814.tif", 
            "2.1166_50.167.tif",
            "2.118_50.1656.tif",
            "2.744_50.154.tif", 
            "2.744_50.1484.tif", 
            "2.7412_50.1582.tif",
            "2.7426_50.1484.tif",
            "2.7426_50.1582.tif",
            "2.7454_50.154.tif",
            "2.7496_50.1512.tif",
            "2.7496_50.1582.tif",
            "2.7742_50.1452.tif",
            "2.7756_50.1438.tif",
            "2.777_50.1438.tif"
            
            ]

#parks = ["Airan", 
#         "Agenville",
#         
#         "Ablainzevelle", 
#         "Ablainzevelle",
#         "Ablainzevelle", 
#         "Ablainzevelle",
#         "Ablainzevelle",
#         "Achiet-le-Grand"
#         
#         
#         ]

p.loc[p[2]>0.5,:]



# check image valid
#img_path = "data/aerialImages/Google/RESOLUTION19/AT/keras/train/Turbines/35.png"
#img_path = "data/aerialImages/Google/RESOLUTION19/CN/assessment/Anhui Laian Baoshan Wind/turbines/118.434_32.6796.png"
#fpr.check_image(img_path,model,"heatmap_turbine.png")

# check image in-valid
# img_path = "data/aerialImages/Google/RESOLUTION19/AT/keras/train/NoTurbines/1.png"
# check_image(img_path,model,"heatmap_no_turbine.png")
