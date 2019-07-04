import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

import skimage
import skimage.io as skio

import os
from os import path

import warnings
warnings.simplefilter("ignore")

import time

import cv2

COLORS = ((0,0,0), (51, 51, 255), (255, 51, 51), (51, 255, 51), (255,255,0), (0,255,255), (0,127,255), (128,0,255), (102,102,255), (255,102,102), (102,255,102) )


plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Make sure that caffe is on the python path:
caffe_root = '.'  # this file is expected to be in {caffe_root}/examples
import os
#os.chdir(caffe_root)
import sys
sys.path.insert(0, caffe_root + "/caffe/python")

from inspect import getmembers, isfunction

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()



from google.protobuf import text_format
from caffe.proto import caffe_pb2 as cpb2

#print cpb2

# load PASCAL VOC labels

voc_labelmap_file = "data/VOC_toyota/labelmap_voc.prototxt"
file = open(voc_labelmap_file, 'r')

voc_labelmap = cpb2.LabelMap()


text_format.Merge(str(file.read()), voc_labelmap)

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    classindex = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                classindex.append(labelmap.item[i].label)
                break
        assert found == True
    return labelnames, classindex

model_def = 'deploy.prototxt'
model_weights = 'trained.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                1)     # use test mode (e.g., don't perform dropout)

#caffe.TEST

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB


# set net to batch size of 1
image_resize = 500
net.blobs['data'].reshape(1,3,image_resize,image_resize)


def predict(imgpath, outdir):
    
    start_time = time.time()
    imagename = imgpath.split('/')[-1]

    image = cv2.imread(imgpath)
    cpimg = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = skimage.img_as_float(image).astype(np.float32)

    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image

    # Forward pass.
    detections = net.forward()['detection_out']

    # Parse the outputs.
    det_label = detections[0,0,:,1]
    det_conf = detections[0,0,:,2]
    det_xmin = detections[0,0,:,3]
    det_ymin = detections[0,0,:,4]
    det_xmax = detections[0,0,:,5]
    det_ymax = detections[0,0,:,6]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.7]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels, top_class_index = get_labelname(voc_labelmap, top_label_indices)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]


    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    if top_conf.shape[0] > 0:
        for i in xrange(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = int(round(top_ymax[i] * image.shape[0]))
            score = top_conf[i]
            label = top_labels[i]
            color_index = top_class_index[i]
            name = '%s: %.2f'%(label, score)
            #if label != "sky" and label != "road":
            cv2.rectangle(cpimg, (xmin, ymin), (xmax, ymax), COLORS[color_index], 2)
            cv2.putText(cpimg, name, (xmin, ymin + 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, COLORS[color_index] , 1)


        output_img = path.join(outdir,imagename)

        cv2.imwrite(output_img,cpimg)
    else:
        output_img = path.join(outdir ,imagename)
        print output_img

        cv2.imwrite(output_img,cpimg)

    end_time = time.time()

    exec_time = end_time - start_time

    print 'Detect %s in %s seconds' % (imagename, exec_time)


if __name__ == '__main__':
    inputlist = open('frames.txt','r')
    lines = inputlist.readlines()
    for line in lines:
        line = line.replace('\n','')
        predict(line, 'outdir')

