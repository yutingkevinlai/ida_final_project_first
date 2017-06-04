from pylab import *
#import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import glob
import random
import sys, os
import cv2
project_home = '/home/kevin/caffe_practice/ida_final_project/'
caffe_root = '/home/kevin/Downloads/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe

MODEL = 'ida_deploy.prototxt'
WEIGHTS = 'ida_iter_10000.caffemodel'

def deploy(img):
    #os.environ['GLOG_minloglevel'] = '2' 
    #imgname = sys.argv[1]
    caffe.set_device(0)
    caffe.set_mode_gpu()
    net = caffe.Net(project_home+MODEL, project_home+WEIGHTS, caffe.TEST)
    # image mean
    mu = np.load(project_home + 'ida_mean.npy')
    # print 'mean-subtracted values:', zip('BGR', mu)
    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1)) # move image channels to outermost dimension
    transformer.set_mean('data', mu) # subtract the dataset-mean value in each channel
    # !!!! Don't need to rescale, because the mean image is [0, 1]
    #transformer.set_raw_scale('data', 255) # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0)) # swap channels from RGB to BGR
    
    #img = plt.imread(imgname)
    img = transformer.preprocess('data', img)
    img = img[...,None]    
    img = img.transpose(3,0,1,2)    
    #    
    net.blobs['data'].data[...] = img
    output = net.forward()
    output_prob = output['prob'][0]
    #print output_label
    print output_prob
    print 'prob', output_prob.argmax()

if __name__=='__main__':
    os.environ['GLOG_minloglevel'] = '2' 
    imgname = sys.argv[1]
    img = plt.imread(imgname)
    deploy(img)
