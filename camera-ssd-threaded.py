# --------------------------------------------------------
# Camera Single-Shot Multibox Detector (SSD) sample code
# for Tegra X2/X1
#
# This program captures and displays video from IP CAM,
# USB webcam, or the Tegra onboard camera, and do real-time
# object detection with Single-Shot Multibox Detector (SSD)
# in Caffe. Refer to the following blog post for how to set
# up and run the code:
#
#   https://jkjung-avt.github.io/camera-ssd-threaded/
#
# Written by JK Jung <jkjung13@gmail.com>
# --------------------------------------------------------

import os
import sys
import time
import argparse
import threading
import numpy as np
import cv2
from google.protobuf import text_format


CAFFE_ROOT = '/home/nvidia/project/ssd-caffe/'
sys.path.insert(0, CAFFE_ROOT + 'python')
import caffe
from caffe.proto import caffe_pb2


DEFAULT_PROTOTXT = CAFFE_ROOT + 'models/VGGNet/coco/SSD_300x300/deploy.prototxt'
DEFAULT_MODEL    = CAFFE_ROOT + 'models/VGGNet/coco/SSD_300x300/VGG_coco_SSD_300x300_iter_400000.caffemodel'
DEFAULT_LABELMAP = CAFFE_ROOT + 'data/coco/labelmap_coco.prototxt'

WINDOW_NAME = 'CameraSSDDemo'
BBOX_COLOR  = (0, 255, 0)  # green
PIXEL_MEANS = np.array([[[104.0, 117.0, 123.0]]], dtype=np.float32)

# The following 2 global variables are shared between threads
THREAD_RUNNING = False
IMG_HANDLE = None


def parse_args():
    # Parse input arguments
    desc = ('This script captures and displays live camera video, '
            'and does real-time object detection with Single-Shot '
            'Multibox Detector (SSD) in Caffe on Jetson TX2/TX1')
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--file', dest='use_file',
                        help='use a video file as input (remember to '
                        'also set --filename)',
                        action='store_true')
    parser.add_argument('--filename', dest='filename',
                        help='video file name, e.g. test.mp4',
                        default=None, type=str)
    parser.add_argument('--rtsp', dest='use_rtsp',
                        help='use IP CAM (remember to also set --uri)',
                        action='store_true')
    parser.add_argument('--uri', dest='rtsp_uri',
                        help='RTSP URI, e.g. rtsp://192.168.1.64:554',
                        default=None, type=str)
    parser.add_argument('--latency', dest='rtsp_latency',
                        help='latency in ms for RTSP [200]',
                        default=200, type=int)
    parser.add_argument('--usb', dest='use_usb',
                        help='use USB webcam (remember to also set --vid)',
                        action='store_true')
    parser.add_argument('--vid', dest='video_dev',
                        help='device # of USB webcam (/dev/video?) [1]',
                        default=1, type=int)
    parser.add_argument('--width', dest='image_width',
                        help='image width [1280]',
                        default=1280, type=int)
    parser.add_argument('--height', dest='image_height',
                        help='image height [720]',
                        default=720, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='run Caffe in CPU mode (default: GPU mode)',
                        action='store_true')
    parser.add_argument('--prototxt', dest='caffe_prototxt',
                        help='[{}]'.format(DEFAULT_PROTOTXT),
                        default=DEFAULT_PROTOTXT, type=str)
    parser.add_argument('--model', dest='caffe_model',
                        help='[{}]'.format(DEFAULT_MODEL),
                        default=DEFAULT_MODEL, type=str)
    parser.add_argument('--labelmap', dest='labelmap_file',
                        help='[{}]'.format(DEFAULT_LABELMAP),
                        default=DEFAULT_LABELMAP, type=str)
    parser.add_argument('--confidence', dest='conf_th',
                        help='confidence threshold [0.3]',
                        default=0.3, type=float)
    args = parser.parse_args()
    return args


def open_cam_rtsp(uri, width, height, latency):
    gst_str = ('rtspsrc location={} latency={} ! '
               'rtph264depay ! h264parse ! omxh264dec ! '
               'nvvidconv ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink').format(uri, latency, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_cam_usb(dev, width, height):
    # We want to set width and height here, otherwise we could just do:
    #     return cv2.VideoCapture(dev)
    gst_str = ('v4l2src device=/dev/video{} ! '
               'video/x-raw, width=(int){}, height=(int){} ! '
               'videoconvert ! appsink').format(dev, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_cam_onboard(width, height):
    # On versions of L4T prior to 28.1, add 'flip-method=2' into gst_str
    gst_str = ('nvcamerasrc ! '
               'video/x-raw(memory:NVMM), '
               'width=(int)2592, height=(int)1458, '
               'format=(string)I420, framerate=(fraction)30/1 ! '
               'nvvidconv ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink').format(width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_window(width, height):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, 'Camera SSD Object Detection Demo '
                                    'for Jetson TX2/TX1')

#
# This 'grab_img' function is designed to be run in the sub-thread.
# Once started, this thread continues to grab new image and put it
# into the global IMG_HANDLE, until THREAD_RUNNING is set to False.
#
def grab_img(cap):
    global THREAD_RUNNING
    global IMG_HANDLE
    while THREAD_RUNNING:
        _, IMG_HANDLE = cap.read()
        if IMG_HANDLE is None:
            print('grab_img(): cap.read() returns None...')
            break
    THREAD_RUNNING = False


def preprocess(src):
    '''Preprocess the input image for SSD
    '''
    img = cv2.resize(src, (300, 300))
    img = img.astype(np.float32) - PIXEL_MEANS
    return img


def postprocess(img, out):
    '''Postprocess the ouput of the SSD object detector
    '''
    h, w, c = img.shape
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)


def detect(origimg, net):
    img = preprocess(origimg)
    img = img.transpose((2, 0, 1))

    tic = time.time()
    net.blobs['data'].data[...] = img
    out = net.forward()
    dt = time.time() - tic
    box, conf, cls = postprocess(origimg, out)
    #print('Detection took {:.3f}s, found {} objects'.format(dt, len(box)))
    print('Detection took {:.3f}s'.format(dt))

    return (box, conf, cls)


def show_bounding_boxes(img, box, conf, cls, cls_dict, conf_th):
    for bb, cf, cl in zip(box, conf, cls):
        cl = int(cl)
        # Only keep non-background bounding boxes with confidence value
        # greater than threshold
        if cl == 0 or cf < conf_th:
            continue
        x_min, y_min, x_max, y_max = bb[0], bb[1], bb[2], bb[3]
        cv2.rectangle(img, (x_min,y_min), (x_max,y_max), BBOX_COLOR, 2)
        txt_loc = (max(x_min, 5), max(y_min-3, 20))
        cls_name = cls_dict.get(cl, 'CLASS{}'.format(cl))
        txt = '{} {:.2f}'.format(cls_name, cf)
        cv2.putText(img, txt, txt_loc, cv2.FONT_HERSHEY_DUPLEX, 0.8,
                    BBOX_COLOR, 1)


def read_cam_and_detect(net, cls_dict, conf_th):
    global THREAD_RUNNING
    global IMG_HANDLE
    show_help = True
    full_scrn = False
    help_text = '"Esc" to Quit, "H" for Help, "F" to Toggle Fullscreen'
    font = cv2.FONT_HERSHEY_PLAIN
    while THREAD_RUNNING:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            # Check to see if the user has closed the window
            # If yes, terminate the program
            break

        img = IMG_HANDLE
        if img is not None:
            box, conf, cls = detect(img, net)
            show_bounding_boxes(img, box, conf, cls, cls_dict, conf_th)

            if show_help:
                cv2.putText(img, help_text, (11, 20), font, 1.0,
                            (32, 32, 32), 4, cv2.LINE_AA)
                cv2.putText(img, help_text, (10, 20), font, 1.0,
                            (240, 240, 240), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, img)

        key = cv2.waitKey(1)
        if key == 27: # ESC key: quit program
            break
        elif key == ord('H') or key == ord('h'): # Toggle help message
            show_help = not show_help
        elif key == ord('F') or key == ord('f'): # Toggle fullscreen
            full_scrn = not full_scrn
            if full_scrn:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL)


def main():
    global THREAD_RUNNING
    args = parse_args()
    print('Called with args:')
    print(args)

    if not os.path.isfile(args.caffe_prototxt):
        sys.exit('File not found: {}'.format(args.caffe_prototxt))
    if not os.path.isfile(args.caffe_model):
        sys.exit('File not found: {}'.format(args.caffe_model))
    if not os.path.isfile(args.labelmap_file):
        sys.exit('File not found: {}'.format(args.labelmap_file))

    # Initialize Caffe
    if args.cpu_mode:
        print('Running Caffe in CPU mode')
        caffe.set_mode_cpu()
    else:
        print('Running Caffe in GPU mode')
        caffe.set_device(0)
        caffe.set_mode_gpu()
    net = caffe.Net(args.caffe_prototxt, args.caffe_model, caffe.TEST)

    # Build the class (index/name) dictionary from labelmap file
    lm_handle = open(args.labelmap_file, 'r')
    lm_map = caffe_pb2.LabelMap()
    text_format.Merge(str(lm_handle.read()), lm_map)
    cls_dict = {x.label:x.display_name for x in lm_map.item}

    # Open camera
    if args.use_file:
        cap = cv2.VideoCapture(args.filename)
        # ignore image width/height settings here
    elif args.use_rtsp:
        cap = open_cam_rtsp(args.rtsp_uri,
                            args.image_width,
                            args.image_height,
                            args.rtsp_latency)
    elif args.use_usb:
        cap = open_cam_usb(args.video_dev,
                           args.image_width,
                           args.image_height)
    else: # By default, use the Jetson onboard camera
        cap = open_cam_onboard(args.image_width,
                               args.image_height)

    if not cap.isOpened():
        sys.exit('Failed to open camera!')

    # Start the sub-thread, which is responsible for grabbing images
    THREAD_RUNNING = True
    th = threading.Thread(target=grab_img, args=(cap,))
    th.start()

    # Grab image and do object detection (until stopped by user)
    open_window(args.image_width, args.image_height)
    read_cam_and_detect(net, cls_dict, args.conf_th)

    # Terminate the sub-thread
    THREAD_RUNNING = False
    th.join()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()