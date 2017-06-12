'''
This code is for deploying the image into trained Caffe net
Created by Yu-Ting Lai (0560032)
'''

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

    #### Set caffe GPU computing device ####
    caffe.set_device(0)
    caffe.set_mode_gpu()

    #### Construct Caffe net for classification ####
    net = caffe.Net(project_home+MODEL, project_home+WEIGHTS, caffe.TEST)

    #### Load image mean created before ####
    mu = np.load(project_home + 'ida_mean.npy')

	#### Transform the loaded image mean to input format in Caffe ####
    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1)) # move image channels to outermost dimension
    transformer.set_mean('data', mu) # subtract the dataset-mean value in each channel
    # !!!! Don't need to rescale, because the mean image is [0, 1]
    #transformer.set_raw_scale('data', 255) # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0)) # swap channels from RGB to BGR
    
    #### Apply transformer constructed above to do image preprocessing ####
    img = transformer.preprocess('data', img)
    img = img[...,None]    
    img = img.transpose(3,0,1,2)    
    
    #### Feed the img into neural network, and print the output ####   
    net.blobs['data'].data[...] = img
    output = net.forward()
    output_prob = output['prob'][0]
    # print output_label
    print output_prob
    print 'This is pose : ', output_prob.argmax()
    #first_idx = output_prob.argmax()
    #output_prob[first_idx] = 0
    #print 'Second Pose : ', output_prob.argmax()

if __name__=='__main__':

    #### read input image from argument ####
    os.environ['GLOG_minloglevel'] = '2' 
    imgname = sys.argv[1]
    img = plt.imread(imgname)
    deploy(img)
