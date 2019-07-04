# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 22:00:48 2019

@author: jschmidt
"""


import os, shutil, random

import pandas as pd
import sys
import gdal
import imp

from shutil import copyfile

import scripts.windturbines.functions_pattern_recognition as fpr
imp.reload(fpr)
from scripts.windturbines.functions_pattern_recognition import get_param
from scripts.windturbines.functions_pattern_recognition import cop_predict
from scripts.windturbines.functions_pattern_recognition import check_image
from scripts.windturbines.functions_pattern_recognition import read_params

COUNTRIES = ['AT','BR','CN','FR']

tgt_dir_tb = get_param("MIX","PATH_RAW_IMAGES_TURBINES")
tgt_dir_notb = get_param("MIX","PATH_RAW_IMAGES_NOTURBINES")

shutil.rmtree(tgt_dir_tb,ignore_errors=True)
shutil.rmtree(tgt_dir_notb,ignore_errors=True)

os.makedirs(tgt_dir_tb)
os.makedirs(tgt_dir_notb)


quality_check_list=[]

for COUNTRY in COUNTRIES:
    
    print("=================== "+COUNTRY+" ======================")
    src_dir_tb = get_param(COUNTRY,"PATH_RAW_IMAGES_TURBINES_MACHINE_CLASSIFIED")
    src_dir_notb = get_param(COUNTRY,"PATH_RAW_IMAGES_NOTURBINES")
    src_dir_notb_png = get_param(COUNTRY,"PATH_RAW_IMAGES_NOTURBINES_MACHINE_CLASSIFIED")

   

    files_tb = [x for x in os.listdir(src_dir_tb) if x.endswith(".png")]
    files_notb = [x for x in os.listdir(src_dir_notb) if x.endswith(".tif")]
    files_notb_png = [x for x in os.listdir(src_dir_notb_png) if x.endswith(".png")]
    
    for i in range(len(files_tb)):
        
        if i % 20 == 0:
            print("Current file: "+str(i))
        
        try:
        
            f_tb = files_tb[i]
            file_name_tb = f_tb[0:-4]+"_"+COUNTRY+f_tb[-4:len(f_tb)]
            copyfile(src_dir_tb+f_tb, tgt_dir_tb+file_name_tb)
            
        except:
            print("copy error")
            print(src_dir_tb+f_tb)
            print(tgt_dir_tb+file_name_tb)
        quality_check_list.append([file_name_tb,100])    
        
        
    ## we copy at most the same amount of non turbines as turbines
    for i in range(min(len(files_tb), len(files_notb))):
        
        if i % 20 == 0:
            print("Current file: "+str(i))
        
        try:
            
            f_notb = files_notb[i]
            gdal.Translate(tgt_dir_notb+f_notb[0:-4]+"_"+COUNTRY+".png",src_dir_notb+f_notb)
   
        
        except:
            print("copy error")
            print(src_dir_notb+files_notb[i])
            print(tgt_dir_notb+files_notb[i][0:-4]+"_"+COUNTRY+".png")
      
    
    for i in range(len(files_notb_png)):
        
        if i % 20 == 0:
            print("Current file: "+str(i))
        
        try:
        
            f_notb_png = files_notb_png[i]
            file_name_notb_png = f_notb_png[0:-4] + "_" + COUNTRY + f_notb_png[-4:len(f_tb)]
            copyfile(src_dir_notb_png + f_notb_png, tgt_dir_notb + file_name_notb_png)
            
        except:
            print("copy error")
            print(src_dir_notb_png + files_notb_png[i])
            print(tgt_dir_notb + f_notb_png[0:-4] + "_" + COUNTRY + f_notb_png[-4:len(f_tb)])
    
        
        
            
    
    
quality_check = pd.DataFrame(quality_check_list,columns=['id','quality'])
 
quality_check.to_csv(get_param("MIX","FILE_QUALITY_CHECK"))
    