# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 10:13:16 2019

@author: jschmidt
"""


from PIL import Image
import numpy as np
from skimage import transform

from matplotlib.pyplot import imshow

import pandas as pd
import os
import gdal

from shutil import copyfile

import shutil

from keras import backend as K

from keras import models
from keras.preprocessing import image

import cv2

def read_params():
    files = [x for x in os.listdir("config/") if (x.endswith(".csv") and x.startswith("params"))]
    p = {}
    
    countries = []
    
    for i in files:
        country = i[6:-4]
        countries.append(country)
        
        p[country] = pd.read_csv("config/" + i)
    
    return((countries,p))
        
COUNTRIES, PARAMS = read_params()


def get_param(country,name):
    val=PARAMS[country].at[0,name]
    return(val)


def cop(dst, src, source_ext):
    src=src + source_ext
    dst=dst + ".png"
    if(not os.path.isfile(dst)): 
        
        if(source_ext == ".png"):
             copyfile(src,dst)
             
        else:
        
            try:
                gdal.Translate(dst,src)
            except:
                print("gdal error")
    

def load(filename):
   
   np_image = Image.open(filename)
   #imshow(np.asarray(np_image))
   
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (256, 256, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

def cop_predict(f,threshold,raw_dir,temp_dir,dest_dir,model):
    src = raw_dir+f
    dst = temp_dir+f[:-4]+".png"
   
    final_dst = dest_dir+f[:-4]+".png"
    if(not os.path.isfile(final_dst)):
        try:
            gdal.Translate(dst,src)
        except:
            print("Exception gdal " + f)
            return(f)
        
        print(src)
        print(dst)
        print(final_dst)
    
    
        image = load(dst) 
        predict = model.predict(image)[0]
        print(predict)
    
        if(predict>threshold):
            copyfile(dst,final_dst)
    
        os.remove(dst)
        os.remove(dst+".aux.xml")
    
        return("")
    
    return("")
    
    


    
#K.clear_session()


def check_image(file, model, filename):
    
    K.clear_session()
    
    model = models.load_model("models/simple-model-027-0.988224-1.000000.h5")
    
    img_path=file
    img = image.load_img(img_path, target_size=(256, 256))

    #imshow(img)

    # `x` is a float32 Numpy array of shape (224, 224, 3)
    x = image.img_to_array(img)

    # We add a dimension to transform our array into a "batch"
    # of size (1, 224, 224, 3)
    x = np.expand_dims(x, axis=0)

    # Finally we preprocess the batch
    # (this does channel-wise color normalization)
    #x=image.img_to_array(x)
    #x=np.expand_dims(x, axis=0)
    x /=255

    print(model.predict(x))

    m_out = model.output[:,0]

    # The is the output feature map of the `block5_conv3` layer,
    # the last convolutional layer in VGG16
    last_conv_layer = model.get_layer('conv2d_4')
    #last_conv_layer = model.get_layer("vgg16").get_layer('block5_conv3')

    # This is the gradient of the "african elephant" class with regard to
    # the output feature map of `block5_conv3`
    grads = K.gradients(m_out, last_conv_layer.output)[0]

    # This is a vector of shape (512,), where each entry
    # is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    # This function allows us to access the values of the quantities we just defined:
    # `pooled_grads` and the output feature map of `block5_conv3`,
    # given a sample image
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

    # These are the values of these two quantities, as Numpy arrays,
    # given our sample image of two elephants
    pooled_grads_value, conv_layer_output_value = iterate([x])

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the elephant class
    for i in range(127):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(conv_layer_output_value, axis=-1)


    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    #plt.matshow(heatmap)
    #plt.show()



    # We use cv2 to load the original image
    img = cv2.imread(img_path)
    
  #  imshow(img)

    # We resize the heatmap to have the same size as the original image
    
    #heatmap_ = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    heatmap_ = heatmap
    
    #heatmap_ = heatmap 
    #img = cv2.resize(img, (heatmap.shape[1], heatmap.shape[0]))
 #   imshow(heatmap_)


    # We convert the heatmap to RGB
    heatmap_ = np.uint8(255 * heatmap_)
#    imshow(heatmap_)

    # We apply the heatmap to the original image
    heatmap_ = cv2.applyColorMap(heatmap_, cv2.COLORMAP_JET)
    #imshow(heatmap_)

    heatmap_ = cv2.resize(heatmap_, (img.shape[1], img.shape[0]),cv2.INTER_AREA)
   

    superimposed_img = np.around(heatmap_ * 0.7 + img).astype(int)
    
    imshow(superimposed_img)
    
    image.save_img("presentations/figures/"+filename, superimposed_img)

    # Save the image to disk
    #cv2.imwrite('/Users/fchollet/Downloads/elephant_cam.jpg', superimposed_img)


def assess_windparks_country(raw_dir, dirs, temp_dir, model, threshold):
    
    lons_lats_found = []

    for directory in dirs:

        dir_all = raw_dir + directory + "/"
    
        files = [x for x in os.listdir(dir_all) if x.endswith(".tif")]

        dir_all_turbines = raw_dir + directory + "/turbines/"
    

        shutil.rmtree(dir_all_turbines, ignore_errors=True)
        os.makedirs(dir_all_turbines, exist_ok=True)

        
        
        classified_dir = dir_all_turbines    

        for f in files:
            src = dir_all + f
            dst = temp_dir + f[:-4]+".png"
            
            try:
                gdal.Translate(dst,src)
            except:
                print("Exception gdal " + f)
                continue
        
            print(src)
            print(dst)
   
    
            image = load(dst) 
            predict = model.predict(image)[0]
            print(predict)
            
            final_dst = classified_dir + f[:-4] + "_" + str(round(predict[0],3)) + ".png"
        
            if(predict>threshold):
                print("windturbine found at "+f[0:-4])
                element = str.split(f[0:-4],"_")
                element.append(predict)
                element.append(f)
                element.append(directory)
                lons_lats_found.append(element)
                shutil.copyfile(dst,final_dst)
    
            os.remove(dst)
            os.remove(dst+".aux.xml")


    return(lons_lats_found) 


    #input("Press Enter to continue...")   
    

    
    
    
   
    
    
        
        