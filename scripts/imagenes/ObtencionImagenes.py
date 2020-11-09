
from HubRequest import *
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import json
from TP2_utils import *

from sentinelhub import MimeType, CRS, BBox, SentinelHubRequest, SentinelHubDownloadClient, \
    DataCollection, bbox_to_dimensions, DownloadRequest

def normalizar(val):
    val = val/255*3.5
    if (val > 1):
        val = 1
    return val*255


#from utils import plot_image

#betsiboka_coords_wgs84 = [-62.05, -33.75, -62.07, -33.8]
betsiboka_coords_wgs84 = [46.16, -16.15, 46.51, -15.58]
resolution = 40
betsiboka_bbox = BBox(bbox=betsiboka_coords_wgs84, crs=CRS.WGS84)
betsiboka_size = bbox_to_dimensions(betsiboka_bbox, resolution=resolution)

print(f'Image shape at {resolution} m resolution: {betsiboka_size} pixels')

evalscript_true_color = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04"]
            }],
            output: {
                bands: 3
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B04, sample.B03, sample.B02];
    }
"""

request_true_color = SentinelHubRequest(
    evalscript=evalscript_true_color,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L1C,
            time_interval=('2020-06-12', '2020-06-13'),
        )
    ],
    responses=[
        SentinelHubRequest.output_response('default', MimeType.TIFF)
    ],
    bbox=betsiboka_bbox,
    size=betsiboka_size,
    config=config
)



true_color_imgs = request_true_color.get_data()
#print(json.dumps(request_true_color))

print(f'Returned data is of type = {type(true_color_imgs)} and length {len(true_color_imgs)}.')
print(f'Single element in the list is of type {type(true_color_imgs[-1])} and has shape {true_color_imgs[-1].shape}')

image = true_color_imgs[0]
print(f'Image type: {image.dtype}')

# plot function
# factor 1/255 to scale between 0-1
# factor 3.5 to increase brightness
#plot_image(image, factor=3.5/255, clip_range=(0,1))
img = image.map(normalizar)
#img = np.array(image, np.float32)
#img = img*3.5
#img[:,:,:] = min(img[:,:,:],255)
#img = np.array(img, "uint8")
cv.imshow("imagen", img)
cv.waitKey(0)
cv.destroyAllWindows()

