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
import pandas
from keras.preprocessing import image
import numpy as np


from keras.preprocessing.image import ImageDataGenerator

os.chdir("G:/Meine Ablage/LVA/PhD Lectures/MachineLearningCourse")

params=pandas.read_csv("config/params.csv")

def get_param(name):
    val=params.at[0,name]
    return(val)


test_base_dir=get_param("PATH_ML_IMAGES_TURBINES_TEST")+"../"

model=models.load_model('models/model-027-0.988224-1.000000.h5')

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        test_base_dir,
        target_size=(256, 256),
        batch_size=32,
        class_mode='binary')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=32)
print('test acc:', test_acc)


#############predict all images iwth turbines

from PIL import Image
import numpy as np
from skimage import transform

from matplotlib.pyplot import imshow

validation_generator = test_datagen.flow_from_directory(
        test_base_dir,
        #target_size=(256, 256),
        batch_size=32,
        class_mode='binary')

def load(filename):
   
   np_image = Image.open(filename)
   #imshow(np.asarray(np_image))
   
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (256, 256, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image


#### PREDICT ALL IN AT DIRECTORIES

PATH_RAW_IMAGES_TURBINES_MACHINE_CLASSIFIED
PATH_RAW_IMAGES_TURBINES
files = [x for x in os.listdir(get_param("PATH_ML_IMAGES_TURBINES_TEST")) if x.endswith(".png")]

final_predictions = pd.DataFrame(files,columns=["Filename"])
final_predictions['prediction'] = 0

for i in range(len(files)):
    print(i)
    print(files[i])
    image = load(get_param("PATH_ML_IMAGES_TURBINES_TEST")+files[i])
    final_predictions.iloc[i,1]=model.predict(image)[0]
    
wrongly_classified=final_predictions.loc[final_predictions['prediction']<0.5]
    
####wrongly classified 906,907, 1035,1036,1037

for i in range(wrongly_classified.shape[0]):
    f=get_param("PATH_ML_IMAGES_TURBINES_TEST")+wrongly_classified.iloc[i,0]
    check_image(f)

#### PREDICT ALL IN BR DIRECTORIES
    
files = [x for x in os.listdir(get_param("PATH_ML_IMAGES_TURBINES_TRAIN")) if x.endswith(".png")]

final_predictions = pd.DataFrame(files,columns=["Filename"])
final_predictions['prediction'] = 0

for i in range(len(files)):
    print(i)
    print(files[i])
    image = load(get_param("PATH_ML_IMAGES_TURBINES_TRAIN")+files[i])
    predict=model.predict(image)[0]
    print(predict)
    final_predictions.iloc[i,1]=model.predict(image)[0]
    
classified_wind_turbines=final_predictions.loc[final_predictions['prediction']>0.2]
classified_wind_turbines.shape


from shutil import copyfile


for i in range(classified_wind_turbines.shape[0]):
    src= get_param("PATH_ML_IMAGES_TURBINES_TRAIN")+classified_wind_turbines.iloc[i,0]   
    dst= get_param("PATH_ML_IMAGES_TURBINES_TRAIN")+"classified/"+classified_wind_turbines.iloc[i,0] 
    try:
        copyfile(src,dst)
    except: 
        print("file exists already")

###all good with exception of 644, 823 if > 0.8
###all good with exception of 419, 644, 823, 1136 if > 0.2
        
f=get_param("PATH_ML_IMAGES_TURBINES_TRAIN")+classified_wind_turbines.loc[classified_wind_turbines['Filename']=='1136.png'].iloc[0,0]
check_image(f)


classified_no_wind_turbines=final_predictions.loc[final_predictions['prediction']<0.2]
classified_no_wind_turbines.shape 


for i in range(classified_no_wind_turbines.shape[0]):
    src= get_param("PATH_ML_IMAGES_TURBINES_TRAIN")+classified_no_wind_turbines.iloc[i,0]   
    dst= get_param("PATH_ML_IMAGES_TURBINES_TRAIN")+"classified_no/"+classified_no_wind_turbines.iloc[i,0] 
    try:
        copyfile(src,dst)
    except: 
        print("file exists already")



for i in range(classified_wind_turbines.shape[0]):
    f=get_param("PATH_ML_IMAGES_TURBINES_TRAIN")+classified_wind_turbines.iloc[i,0]
    check_image(f)





















validation_generator = test_datagen.flow_from_directory(
        test_base_dir,
        #target_size=(256, 256),
        batch_size=32,
        class_mode='binary')

#predictions = model.predict_generator(validation_generator,steps=1)



steps=276 / 32
predictions = model.predict_generator(validation_generator,steps=steps)




df = pd.read_csv(io.StringIO('\n'.join(validation_generator.filenames)), delim_whitespace=True,header=None)
df['b']=predictions[:,0]

import sys
i=0
i=200

image = load(test_base_dir+validation_generator.filenames[i])
i+=1
print(model.predict(image))

img_patch=test_base_dir+validation_generator.filenames[i]

img = image.load_img(img_patch, target_size=(256,256))

img_tensor=image.img_to_array(img)
img_tensor=np.expand_dims(img_tensor, axis=0)
img_tensor /=255

print(img_tensor.shape)

imshow(img_tensor[0])

from keras import models

layer_outputs= [layer.output for layer in model.layers[:8]]
activation_model=models.Model(inputs=model.input, outputs=layer_outputs)

activations=activation_model.predict(img_tensor)
first_layer_activation=activations[0]
print(first_layer_activation.shape)

import matplotlib.pyplot as plt
imshow(img_tensor[0])
plt.matshow(first_layer_activation[0,:,:,4],cmap='viridis')
plt.matshow(first_layer_activation[0,:,:,7],cmap='viridis')

layer_names=[]
for layer in model.layers[:8]:
    layer_names.append(layer.name)
    
images_per_row=16

for layer_name, layer_activation in zip(layer_names, activations):
    n_features=layer_activation.shape[-1]
    size=layer_activation.shape[1]
    
    n_cols=round(n_features / images_per_row)
    
    display_grid=np.zeros((size*n_cols,images_per_row*size))
    
    for col in range(n_cols):
        for row in range(images_per_row):
            #print(col*images_per_row+row)
            channel_image=layer_activation[0,:,:,col*images_per_row+row]
            
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *=64
            channel_image +=128
            channel_image=np.clip(channel_image,0,255).astype('uint8')
            display_grid[col*size:(col+1)*size,
                         row*size:(row+1)*size] = channel_image
                         
    scale=1. / size
    plt.figure(figsize=(scale*display_grid.shape[1],
                        scale*display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid,aspect='auto',cmap='viridis')
    
    


from keras import backend as K
    
#K.clear_session()


def check_image(file):
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

    import cv2

    # We use cv2 to load the original image
    img = cv2.imread(img_path)
    
  #  imshow(img)

    # We resize the heatmap to have the same size as the original image
    heatmap_ = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
 #   imshow(heatmap_)


    # We convert the heatmap to RGB
    heatmap_ = np.uint8(255 * heatmap_)
#    imshow(heatmap_)

    # We apply the heatmap to the original image
    heatmap_ = cv2.applyColorMap(heatmap_, cv2.COLORMAP_JET)
    #imshow(heatmap_)

#    imshow(heatmap_)

    # 0.4 here is a heatmap intensity factor

    superimposed_img = np.around(heatmap_ * 0.4 + img).astype(int)

    imshow(superimposed_img)

    # Save the image to disk
    #cv2.imwrite('/Users/fchollet/Downloads/elephant_cam.jpg', superimposed_img)





    #input("Press Enter to continue...")   
    
    
check_image(0)
