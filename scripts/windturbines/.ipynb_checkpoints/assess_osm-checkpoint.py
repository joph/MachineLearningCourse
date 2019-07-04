# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 13:28:53 2019

@author: jschmidt
"""

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



# Get images from raw directory
# Convert to png in temp
# Predict
# If classified, copy

model = models.load_model('models/simple-model-027-0.988224-1.000000.h5')

# within this file, we assess new regions
COUNTRY = "DE"

raw_dir = fpr.get_param(COUNTRY, "PATH_RAW_IMAGES_OSM")
temp_dir = fpr.get_param(COUNTRY, "PATH_TEMP")

lon_lat = pd.read_csv(fpr.get_param(COUNTRY, "FILE_OSM_TURBINE_LOCATIONS"))


files = []

for i in range(lon_lat.shape[0]):
    files.append(str(i + 1) + ".tif")

lon_lat["filenames"] = files

lon_lat["prediction"] = 0

for i in range(len(files)):
    
    
    f = files[i]
    
    if(i%10 == 0):
        print("File " + f)

        
    src = raw_dir + f
    dst = temp_dir + f[:-4] + "_osm-assessment.png"
            
    element = fpr.assess_location(f, src, dst, "", model)
    
    lon_lat.iloc[i, 3] = element[1]
    
                

