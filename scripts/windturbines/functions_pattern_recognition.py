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

from pathlib import Path

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

import pickle  # pip install dill --user
import matplotlib.pyplot as plt
from keras.applications import VGG16
import tensorflow as tf
import pandas as pd

from sklearn.cluster import DBSCAN

import cv2 as cv
import glob

gpu_fraction = 0.8
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

import cv2


def read_params():
    config_file = Path(__file__).parent.parent.parent / "config"

    files = [x for x in os.listdir(config_file) if (x.endswith(".csv") and x.startswith("params"))]
    p = {}

    countries = []
    resolutions = []

    for i in files:
        country = i[6:-6]
        countries.append(country)
        resolution = i[-6:-4]
        resolutions.append(country)

        try:
            t = p[country]
        except KeyError:
            p[country] = {}

        p[country][resolution] = pd.read_csv(config_file / i)

    return ((countries, p))


COUNTRIES, PARAMS = read_params()


def get_param(country, name, resolution=19):
    val = PARAMS[country][str(resolution)].at[0, name]
    return (val)


def cop(dst, src, source_ext):
    src = src + source_ext
    dst = dst + ".png"
    if (not os.path.isfile(dst)):

        if (source_ext == ".png"):
            copyfile(src, dst)

        else:

            try:
                gdal.Translate(dst, src)
            except:
                print("gdal error")


def load(filename):
    np_image = Image.open(filename)
    # imshow(np.asarray(np_image))

    np_image = np.array(np_image).astype('float32') / 255
    np_image = transform.resize(np_image, (256, 256, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


def cop_predict(f, threshold, raw_dir, temp_dir, dest_dir, model):
    src = raw_dir + f
    dst = temp_dir + f[:-4] + ".png"

    final_dst = dest_dir + f[:-4] + ".png"

    if (not os.path.isfile(final_dst)):
        try:
            gdal.Translate(dst, src)
        except:
            print("Exception gdal " + f)
            return (f)

        print(src)
        print(dst)
        print(final_dst)

        image = load(dst)
        predict = model.predict(image)[0]
        print(predict)

        if (predict < threshold):
            copyfile(dst, final_dst)

        os.remove(dst)
        os.remove(dst + ".aux.xml")

        return ("")

    return ("")


# K.clear_session()


def check_image(file, model, filename):
    K.clear_session()

    model = models.load_model(model)

    img_path = file
    img = image.load_img(img_path, target_size=(256, 256))

    # imshow(img)

    # `x` is a float32 Numpy array of shape (224, 224, 3)
    x = image.img_to_array(img)

    # We add a dimension to transform our array into a "batch"
    # of size (1, 224, 224, 3)
    x = np.expand_dims(x, axis=0)

    # Finally we preprocess the batch
    # (this does channel-wise color normalization)
    # x=image.img_to_array(x)
    # x=np.expand_dims(x, axis=0)
    x /= 255

    print(model.predict(x))

    m_out = model.output[:, 0]

    # The is the output feature map of the `block5_conv3` layer,
    # the last convolutional layer in VGG16
    last_conv_layer = model.get_layer('conv2d_4')
    # last_conv_layer = model.get_layer("vgg16").get_layer('block5_conv3')

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
    # plt.matshow(heatmap)
    # plt.show()

    # We use cv2 to load the original image
    img = cv2.imread(img_path)

    #  imshow(img)

    # We resize the heatmap to have the same size as the original image

    # heatmap_ = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap_ = heatmap

    # heatmap_ = heatmap
    # img = cv2.resize(img, (heatmap.shape[1], heatmap.shape[0]))
    #   imshow(heatmap_)

    # We convert the heatmap to RGB
    heatmap_ = np.uint8(255 * heatmap_)
    #    imshow(heatmap_)

    # We apply the heatmap to the original image
    heatmap_ = cv2.applyColorMap(heatmap_, cv2.COLORMAP_JET)
    # imshow(heatmap_)

    heatmap_ = cv2.resize(heatmap_, (img.shape[1], img.shape[0]), cv2.INTER_AREA)

    superimposed_img = np.around(heatmap_ * 0.7 + img).astype(int)

    imshow(superimposed_img)

    image.save_img("presentations/figures/" + filename, superimposed_img)

    # Save the image to disk
    # cv2.imwrite('/Users/fchollet/Downloads/elephant_cam.jpg', superimposed_img)


def copyfile_to_png(src, dst):

    if (os.path.exists(dst)):
        return ()

    src_ds = gdal.Open(src)

    # Open output format driver, see gdal_translate --formats for list
    format = "PNG"
    driver = gdal.GetDriverByName(format)

    # Output to new format
    try:
        dst_ds = driver.CreateCopy(dst, src_ds, 0)

        dst_ds = None
        src_ds = None
    except:
        print("error")
        print(dst)
        print(src)

    try:
        os.remove(dst + ".aux.xml")
    except:
        #    print(dst + ".aux.xml")
        pass
    # Properly close the datasets to flush to disk


def copy_threshold_files(p, threshold_low, threshold_high, raw_dir, sub_dst_directory, directory_given=True,
                         resolution=19):
    cnt = 0

    old_country = ""

    raw_dir_dst = raw_dir

    for index, row in p.iterrows():

        cnt = cnt + 1

        probability = row[2]
        name = row[3]
        directory = row[4]
        country = row[5]

        dir_all_turbines = raw_dir_dst + sub_dst_directory

        if (not directory_given and not (old_country == country)):
            raw_dir = get_param(country, "PATH_RAW_IMAGES_OSM", resolution)

        old_country = country

        if (probability < threshold_high and probability > threshold_low):

            dir_all = raw_dir + directory + "/"

            src = dir_all + name

            dst = dir_all_turbines + "/" + name[:-4] + "_" + str(probability) + "_" + directory + ".png"

            try:
                src_ds = gdal.Open(src)

                # Open output format driver, see gdal_translate --formats for list
                format = "PNG"
                driver = gdal.GetDriverByName(format)

                # Output to new format
                dst_ds = driver.CreateCopy(dst, src_ds, 0)

                # Properly close the datasets to flush to disk
                dst_ds = None
                src_ds = None

                if (cnt % 100 == 0):
                    print("src: " + src)
                    print("dst: " + dst)

                os.remove(dst + ".aux.xml")
            except Exception as e:
                print(e)


def assess_windparks_country(raw_dir, dirs, temp_dir, model):
    lons_lats_found = []

    for directory in dirs:

        dir_all = raw_dir + directory + "/"

        files = [x for x in os.listdir(dir_all) if x.endswith(".tif")]

        print("Currently assessing " + directory)

        # dir_all_turbines = raw_dir + directory + "/turbines/"
        # dir_no_turbines = raw_dir + directory + "/no_turbines/"

        for f in files:
            src = dir_all + f
            dst = temp_dir + f[:-4] + ".png"

            element = assess_location(f, src, dst, directory, model)

            lons_lats_found.append(element)

    return (lons_lats_found)


def assess_location(f, src, dst, directory, model):
    if (not os.path.isfile(src)):
        element = str.split(f[0:-4], "_")
        element.append(-1)
        element.append(f)
        element.append(directory)
        return (element)

    src_ds = gdal.Open(src)

    # Open output format driver, see gdal_translate --formats for list
    format = "PNG"
    driver = gdal.GetDriverByName(format)
    try:
        # Output to new format
        dst_ds = driver.CreateCopy(dst, src_ds, 0)

        # Properly close the datasets to flush to disk
        dst_ds = None
        src_ds = None

        image = load(dst)
        predict = model.predict(image)[0]

        element = str.split(f[0:-4], "_")
        element.append(predict[0])
        element.append(f)
        element.append(directory)

        os.remove(dst)
        os.remove(dst + ".aux.xml")
        return (element)

    except Exception as e:
        print("Exception gdal " + f)
        print(e)
        element = str.split(f[0:-4], "_")
        element.append(-1)
        element.append(f)
        element.append(directory)
        return (element)


def copy_learning_files(COUNTRY, RESOLUTION, threshold_park_size=10, delete=False):
    print("COUNTRY: " + COUNTRY + " RESOLUTION: " + str(RESOLUTION))

    train_dir = get_param(COUNTRY, "PATH_ML_IMAGES_TRAIN", RESOLUTION)
    test_dir = get_param(COUNTRY, "PATH_ML_IMAGES_TEST", RESOLUTION)

    #### delete directories if exist
    #### create if not exist
    if (delete == True):
        shutil.rmtree(train_dir, ignore_errors=True)
        shutil.rmtree(test_dir, ignore_errors=True)

        os.makedirs(train_dir)
        os.makedirs(test_dir)

    src_dir_tb = get_param(COUNTRY, "PATH_RAW_IMAGES_TURBINES", RESOLUTION)
    src_dir_notb = get_param(COUNTRY, "PATH_RAW_IMAGES_NOTURBINES", RESOLUTION)

    turbines = pd.read_csv(get_param(COUNTRY, "FILE_TURBINE_LOCATIONS"))

    # turbines = turbines.iloc[1:10,]

    turbines['id'] = np.arange(1, turbines.shape[0] + 1)

    predictions_cs = get_param(COUNTRY, "PATH_RAW_IMAGES_TURBINES", 19)

    # turbines_pred = pd.read_feather(predictions_cs + "all_predictions.feather")

    turbines_pred = pd.read_feather(predictions_cs + "all_predictions.feather")

    turbines = turbines[turbines_pred['prediction'] > 0.99]

    ####filter single turbines
    EARTH_RADIUS_KM = 6371.0088

    kms_per_radian = EARTH_RADIUS_KM
    min_distance_km = 0.5
    epsilon = min_distance_km / kms_per_radian

    turbine_locations = turbines[["xlong", "ylat"]]

    clustering = DBSCAN(eps=epsilon, min_samples=2, algorithm='ball_tree',
                        metric='haversine').fit(np.radians(turbine_locations))

    cluster_per_location = clustering.labels_

    turbines["cluster"] = cluster_per_location

    turbines['occur'] = turbines.groupby('cluster')['cluster'].transform('count')

    turbines = turbines[turbines['occur'] > threshold_park_size]

    nmb_samples_all = turbines.shape[0]

    # turbines = turbines[turbines['id'] <= nmb_samples_all]

    share_train = turbines.shape[0] * 0.8

    print("share_train")
    print(share_train)

    print("Copying turbine files")
    cnt = 0
    for i in range(0, turbines.shape[0]):
        cnt += 1

        # print(file)
        file_s = str(turbines.iloc[[i]]['id'].values[0]) + '.tif'
        file_d = str(turbines.iloc[[i]]['id'].values[0]) + '.png'

        # print(file)

        src = os.path.join(src_dir_tb, file_s)

        if (cnt % 1000 == 0):
            print(cnt)
            print(src)

        # print(src)

        if (os.path.isfile(src)):

            dst = ""

            if (cnt < share_train):
                dst = os.path.join(train_dir, file_d)

            if (cnt > share_train):
                dst = os.path.join(test_dir, file_d)

            copyfile_to_png(src, dst)

        #### copy no-turbine images

    print("Copying non-turbine files")

    cnt = 0
    for i in range(0, turbines.shape[0]):
        cnt += 1

        # print(file)
        file_s = str(turbines.iloc[[i]]['id'].values[0]) + '.tif'
        file_d = str(turbines.iloc[[i]]['id'].values[0]) + 'no.png'

        # print(file)

        src = os.path.join(src_dir_notb, file_s)

        if (cnt % 1000 == 0):
            print(cnt)
            print(src)

        # print(src)

        if (os.path.isfile(src)):

            dst = ""

            if (cnt < share_train):
                dst = os.path.join(train_dir, file_d)

            if (cnt > share_train):
                dst = os.path.join(test_dir, file_d)

            copyfile_to_png(src, dst)

            #### copy no-turbine images

    print('total training turbine images ', len(os.listdir(train_dir)))
    print('total testing turbine images ', len(os.listdir(test_dir)))

    pd_out_tb = pd.DataFrame(turbines['id'].map(str) + '.png')
    pd_out_tb['type'] = 'T'

    pd_out_notb = pd.DataFrame(turbines['id'].map(str) + 'no.png')
    pd_out_notb['type'] = 'F'

    pd_out = pd.concat([pd_out_tb, pd_out_notb])

    print('total training turbine images ', len(os.listdir(train_dir)))
    print('total testing turbine images ', len(os.listdir(test_dir)))

    share_train_i = int(len(os.listdir(train_dir)) / 2)

    maxs = int(pd_out.shape[0] / 2)

    ind1 = list(range(0, share_train_i))
    ind11 = list(range(maxs, (maxs + share_train_i)))
    ind1.extend(ind11)

    ind2 = list(range(share_train_i, maxs))
    ind21 = list(range((maxs + share_train_i), 2 * maxs))
    ind2.extend(ind21)

    pd_out_train = pd_out.iloc[ind1]
    pd_out_test = pd_out.iloc[ind2]

    pd_out_train.to_csv(train_dir + "list.csv")
    pd_out_test.to_csv(test_dir + "list.csv")


def remove_erroneous_files(COUNTRY, RESOLUTION):
    train_dir = get_param(COUNTRY, "PATH_ML_IMAGES_TRAIN", RESOLUTION)
    test_dir = get_param(COUNTRY, "PATH_ML_IMAGES_TEST", RESOLUTION)

    imgs_names = glob.glob(train_dir + '/*.png')
    for imgname in imgs_names:
        img = cv.imread(imgname)
        if img is None:
            print(imgname)
            os.remove(imgname)

    imgs_names = glob.glob(train_dir + '/*.png')
    for imgname in imgs_names:
        img = cv.imread(imgname)
        if img is None:
            print(imgname)
            os.remove(imgname)


def train_model_res(RESOLUTION, COUNTRY, epochs=1):
    train_dir = get_param(COUNTRY, "PATH_ML_IMAGES_TRAIN", RESOLUTION)
    test_dir = get_param(COUNTRY, "PATH_ML_IMAGES_TEST", RESOLUTION)

    pd_train = pd.read_csv(train_dir + "list.csv")
    pd_test = pd.read_csv(test_dir + "list.csv")

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       horizontal_flip=True,
                                       fill_mode="nearest",
                                       zoom_range=0.2,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       rotation_range=30,
                                       validation_split=0.25)

    train_generator = train_datagen.flow_from_dataframe(dataframe=pd_train,
                                                        directory=train_dir,
                                                        x_col="id",
                                                        y_col="type",
                                                        has_ext=True,
                                                        class_mode="binary",
                                                        subset="training",
                                                        batch_size=20)

    val_generator = train_datagen.flow_from_dataframe(dataframe=pd_train,
                                                      directory=train_dir,
                                                      x_col="id",
                                                      y_col="type",
                                                      has_ext=True,
                                                      class_mode="binary",
                                                      subset="validation",
                                                      batch_size=20)

    #                                                    target_size=(img_width, img_height),

    # Note that the validation data should not be augmented!
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_dataframe(dataframe=pd_test,
                                                      directory=test_dir,
                                                      x_col="id",
                                                      y_col="type",
                                                      has_ext=True,
                                                      class_mode="binary",
                                                      batch_size=20)

    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(256, 256, 3))

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    conv_base.trainable = True

    set_trainable = False
    for layer in conv_base.layers:
        if layer.name == 'block4_conv1':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-6),
                  metrics=['acc'])

    mcp_save = ModelCheckpoint(
        'models/model-resolution-' + str(RESOLUTION) + '-unfreezed-resolution-{epoch:04d}-{val_loss:.4f}.h5',
        save_best_only=True, monitor='val_loss', mode='min')

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=10,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=10,
        callbacks=[mcp_save],
        verbose=2)

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

    m = min(val_loss)
    e = val_loss.index(m) + 1

    return (f'models/model-resolution-{RESOLUTION}-unfreezed-resolution-{e:04d}-{m:.4f}.h5')
