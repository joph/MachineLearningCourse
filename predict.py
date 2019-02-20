# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 17:07:03 2019

@author: jschmidt
"""

import os

from keras import layers
from keras import models
from keras import optimizers
import pandas as pd
import io

from keras.preprocessing.image import ImageDataGenerator

os.chdir("G:/Meine Ablage/LVA/PhD Lectures/MachineLearningCourse")

test_base_dir='data/windturbines/test'

model=models.load_model('models/turbines.h5')


validation_generator = test_datagen.flow_from_directory(
        test_base_dir,
        #target_size=(256, 256),
        batch_size=202,
        class_mode='binary')

predictions = model.predict_generator(validation_generator,steps=1)

df = pd.read_csv(io.StringIO('\n'.join(validation_generator.filenames)), delim_whitespace=True,header=None)
df['b']=predictions[:,0]
