# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:44:55 2019

@author: jschmidt
"""

import datetime

import matplotlib.pyplot as plt
import numpy as np

from sentinelhub import WmsRequest, BBox, CRS, MimeType, CustomUrlParam, get_area_dates
from s2cloudless import S2PixelCloudDetector, CloudMaskRequest




def overlay_cloud_mask(image, mask=None, factor=1./255, figsize=(15, 15), fig=None):
    """
    Utility function for plotting RGB images with binary mask overlayed.
    """
    if fig == None:
        plt.figure(figsize=figsize)
    rgb = np.array(image)
    plt.imshow(rgb * factor)
    if mask is not None:
        cloud_image = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        cloud_image[mask == 1] = np.asarray([255, 255, 0, 100], dtype=np.uint8)
        plt.imshow(cloud_image)
        
        
def plot_probability_map(rgb_image, prob_map, factor=1./255, figsize=(15, 30)):
    """
    Utility function for plotting a RGB image and its cloud probability map next to each other. 
    """
    plt.figure(figsize=figsize)
    plot = plt.subplot(1, 2, 1)
    plt.imshow(rgb_image * factor)
    plot = plt.subplot(1, 2, 2)
    plot.imshow(prob_map, cmap=plt.cm.inferno)

def plot_cloud_mask(mask, figsize=(15, 15), fig=None):
    """
    Utility function for plotting a binary cloud mask.
    """
    if fig == None:
        plt.figure(figsize=figsize)
    plt.imshow(mask, cmap=plt.cm.gray)
    
    

def plot_previews(data, dates, cols=4, figsize=(15, 15)):
    """
    Utility to plot small "true color" previews.
    """
    width = data[-1].shape[1]
    height = data[-1].shape[0]
    
    rows = data.shape[0] // cols + (1 if data.shape[0] % cols else 0)
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
    for index, ax in enumerate(axs.flatten()):
        if index < data.shape[0]:
            caption = '{}: {}'.format(index, dates[index].strftime('%Y-%m-%d'))
            ax.set_axis_off()
            ax.imshow(data[index] / 255., vmin=0.0, vmax=1.0)
            ax.text(0, -2, caption, fontsize=12, color='g')
        else:
            ax.set_axis_off()
            
            
bbox_coords_wgs84 = [16.946114, 47.904161, 16.954185, 47.901910]
bounding_box = BBox(bbox_coords_wgs84, crs=CRS.WGS84)

INSTANCE_ID = '2643361c-738f-4f62-abcf-92320edf92f6'
#INSTANCE_ID = '5706c37f-6be7-4e5d-a69f-a332543cb0c1'

LAYER_NAME = 'TRUE_COLOR'
#AYER_NAME = '1_TRUE_COLOR'



wms_true_color_request = WmsRequest(layer=LAYER_NAME,
                                    bbox=bounding_box, 
                                    time=('2015-01-01', '2015-12-31'), 
                                    width=600, height=None,
                                    image_format=MimeType.PNG,
                                    instance_id=INSTANCE_ID)

wms_true_color_imgs = wms_true_color_request.get_data()

plot_previews(np.asarray(wms_true_color_imgs), wms_true_color_request.get_dates(), cols=4, figsize=(15, 10))

bands_script = 'return [B01,B02,B04,B05,B08,B8A,B09,B10,B11,B12]'

wms_bands_request = WmsRequest(layer=LAYER_NAME,
                               custom_url_params={
                                   CustomUrlParam.EVALSCRIPT: bands_script,
                                   CustomUrlParam.ATMFILTER: 'NONE'
                               },
                               bbox=bounding_box, 
                               time=('2017-12-01', '2017-12-31'), 
                               width=600, height=None,
                               image_format=MimeType.TIFF_d32f,
                               instance_id=INSTANCE_ID)


wms_bands = wms_bands_request.get_data()

cloud_detector = S2PixelCloudDetector(threshold=0.4, average_over=4, dilation_size=2)


cloud_probs = cloud_detector.get_cloud_probability_maps(np.array(wms_bands))

cloud_masks = cloud_detector.get_cloud_masks(np.array(wms_bands))

image_idx = 0
overlay_cloud_mask(wms_true_color_imgs[image_idx], cloud_masks[image_idx])

plot_probability_map(wms_true_color_imgs[image_idx], cloud_probs[image_idx])

plot_cloud_mask(cloud_masks[image_idx])

all_cloud_masks = CloudMaskRequest(ogc_request=wms_bands_request, threshold=0.1)

fig = plt.figure(figsize=(15, 10))
n_cols = 4
n_rows = int(np.ceil(len(wms_true_color_imgs) / n_cols))

for idx, [prob, mask, data] in enumerate(all_cloud_masks):
    ax = fig.add_subplot(n_rows, n_cols, idx + 1)
    image = wms_true_color_imgs[idx]
    overlay_cloud_mask(image, mask, factor=1, fig=fig)
    
plt.tight_layout()

all_cloud_masks.get_dates()

fig = plt.figure(figsize=(15, 10))
n_cols = 4
n_rows = int(np.ceil(len(wms_true_color_imgs) / n_cols))

for idx, cloud_mask in enumerate(all_cloud_masks.get_cloud_masks(threshold=0.7)):
    ax = fig.add_subplot(n_rows, n_cols, idx + 1)
    plot_cloud_mask(cloud_mask, fig=fig)
    
plt.tight_layout()

fig = plt.figure(figsize=(15, 10))
n_cols = 4
n_rows = int(np.ceil(len(wms_true_color_imgs) / n_cols))

for idx, cloud_mask in enumerate(all_cloud_masks.get_cloud_masks()):
    ax = fig.add_subplot(n_rows, n_cols, idx + 1)
    plot_cloud_mask(cloud_mask, fig=fig)
    
plt.tight_layout()















            
    