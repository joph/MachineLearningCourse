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
COUNTRY = "CN"

raw_dir = fpr.get_param(COUNTRY, "PATH_RAW_IMAGES_ASSESSMENT")
temp_dir = fpr.get_param(COUNTRY, "PATH_TEMP")

dirs = os.listdir(raw_dir)
allResults = []

i=0
found_turbines = fpr.assess_windparks_country(raw_dir, dirs[i:(i+1)],
                                              temp_dir, model)

assessment_out_file = fpr.get_param(COUNTRY, "PATH_RAW_IMAGES_ASSESSMENT") + 
                      "/assessment.csv" 

p = pd.DataFrame(found_turbines)

p.to_csv(assessment_out_file)


for i in range(262,len(dirs)):
    found_turbines = fpr.assess_windparks_country(raw_dir, dirs[i:(i+1)],
                                              temp_dir, model)
    p1 = pd.DataFrame(found_turbines)

    p = p.append(p1)
    
    p.to_csv(assessment_out_file)
    
    
    
