# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 10:52:02 2019

@author: jschmidt
"""


import imp
import scripts.windturbines.functions_pattern_recognition as fpr
imp.reload(fpr)

import pandas as pd


COUNTRY = "CN"

assessment_file = fpr.get_param(COUNTRY, "PATH_RAW_IMAGES_ASSESSMENT") + "/assessment.csv" 

p = pd.read_csv(assessment_file)

p.columns = ['id', 'lon', 'lat', 'prediction', 'filename', 'park']

p.hist(bins=20)

raw_dir = fpr.get_param(COUNTRY, "PATH_RAW_IMAGES_ASSESSMENT")

sum(p.prediction > 0.5)

fpr.copy_threshold_files(p, 0.5, raw_dir)

