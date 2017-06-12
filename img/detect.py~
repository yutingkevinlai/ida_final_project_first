'''
This code is for multi-object scene segmentation and pose estimation
Created by Yu-Ting Lai (0560032)
'''
from pylab import *
#import matplotlib.image as mpimg
#from sklearn.cluster import KMeans
#from scipy.cluster.vq import kmeans, vq
import matplotlib.pyplot as plt
import numpy as np
import glob
import random
import sys, os
import cv2
import math
import scipy
project_home = '/home/kevin/caffe_practice/ida_final_project/'
caffe_root = '/home/kevin/Downloads/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
sys.path.insert(0, project_home)
import deploy

# ------------------------------------ #
# ----- Load image from argument ----- #
# ------------------------------------ #
img_name = sys.argv[1]
img = cv2.imread(img_name, 1)
img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
[h, w, c] = img.shape
test_img = img

Z = img.reshape((-1,3))
print Z.shape
Z = np.float32(Z)


# --------------------------------- #
# --------- Apply K-Means --------- #
# --------------------------------- #
print 'Calculating K-Means'
# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# Set flags (Just to avoid line break in the code)
flags = cv2.KMEANS_RANDOM_CENTERS
# Apply KMeans
ret,labels,centers = cv2.kmeans(Z,4,criteria,10,flags)

#print labels, centers
res_img = np.zeros((h,w,c))
center = np.uint8(centers)
print center
y = labels.flatten()

# find the suitable center idx
ans1 = np.where(np.logical_and(center[:,0]>30, center[:,0]<80))# 30 80
print ans1
ans2 = np.where(np.logical_and(center[:,1]>20, center[:,1]<150))# 20 150
print ans2
ans_socket = np.intersect1d(ans1,ans2)
ans = ans_socket
print ans
idx = np.asarray(ans)
print idx

for i in range(h):
	for j in range(w):
		for k in range(len(idx)):
			if(y[i*w+j]==idx[k]):
				res_img[i][j] = center[y[i]]

#print img.shape
res_img = res_img.reshape((img.shape))

cv2.imshow('K-Means Result', res_img)
cv2.waitKey()
cv2.destroyAllWindows()


# ----------------------------------------- #
# ----- Apply Nearest Neighbor Search ----- #
# ----------------------------------------- #
print 'Calculating nearest neighbor'
# zero padding with 1
pad_img = cv2.copyMakeBorder(res_img,1,1,1,1,cv2.BORDER_CONSTANT,value=[0,0,0])

diff = np.zeros((1,4))
c = 0
cluster_label = np.zeros((h*w,3))

# calculate intensity difference around nearest neighbors
for i in range(1,h+1):
	for j in range(1,w+1):
		cluster_label[((i-1)*w+(j-1)),0] = i-1
		cluster_label[((i-1)*w+(j-1)),1] = j-1

		if((pad_img[i,j].all() == 0)):
			continue
		
		diff[0][0] = pad_img[i,j][0] - pad_img[i-1,j-1][0]		
		diff[0][1] = pad_img[i,j][0] - pad_img[i-1,j][0]
		diff[0][2] = pad_img[i,j][0] - pad_img[i-1,j+1][0]
		diff[0][3] = pad_img[i,j][0] - pad_img[i,j-1][0]
		#print diff
		if(abs(min(diff[0]))>40):
			c = c + 1
			cluster_label[((i-1)*w+(j-1)),2] = c
		else:
			l = argmin(diff)
			if l == 0:
				cluster_label[((i-1)*w+(j-1)),2] = cluster_label[((i-1)*w+(j-1)-w-1),2]
			elif l == 1:
				cluster_label[((i-1)*w+(j-1)),2] = cluster_label[((i-1)*w+(j-1)-w),2]
			elif l == 2:
				cluster_label[((i-1)*w+(j-1)),2] = cluster_label[((i-1)*w+(j-1)-w+1),2]
			elif l == 3:
				cluster_label[((i-1)*w+(j-1)),2] = cluster_label[((i-1)*w+(j-1)-1),2]

# calculate member numbers for each group
group = np.zeros((1,c))
mean = np.zeros((2,c))
for i in range(h*w):
	for k in range(c):
		if cluster_label[i,2] == k+1:
			#print k
			group[0,k] = group[0,k] + 1
			mean[0,k] = (cluster_label[i,0]+mean[0,k]*(group[0,k]-1)) / group[0,k]
			mean[1,k] = (cluster_label[i,1]+mean[1,k]*(group[0,k]-1)) / group[0,k]
#print group
#print mean
'''
for i in range(len(mean[0])):
	print i
	y = int(round(mean[0,i]))
	x = int(round(mean[1,i]))
	test_img[y,x] = 255
'''

# --------------------------------- #
# ----- Remove Small Clusters ----- #
# --------------------------------- #
print 'Removing small clusters'

# remove clusters that have members below 10
count = 0
for i in range(len(group[0])):
	if int(group[0,count]) < 50:
		new_group = np.delete(group,count,1)
		group = new_group
		new_mean = np.delete(mean,count,1)
		mean = new_mean
		'''
		for j in range(len(cluster_label[:,2])):
			if cluster_label[j,2] == count:		
				new_cluster_label = np.delete(cluster_label,count,0)
				cluster_label = new_cluster_label
		'''
		count = count - 1
	count = count + 1
#print 'Current group : ', group
#print len(cluster_label[:,2])

# -------------------------------- #
# ----- Merge Close Clusters ----- #
# -------------------------------- #
print 'Merging close clusters'
thresh = 90

d = np.zeros((len(mean[0]),len(mean[0])))
merge_check = 1
while (merge_check):
#for it in range(100):
	d = np.zeros((len(mean[0]),len(mean[0])))
	for i in range(len(mean[0])):
		for j in range(len(mean[0])):
			d[i,j] = math.sqrt((mean[0,i]-mean[0,j])**2 + (mean[1,i]-mean[1,j])**2)
			if i == j:
				d[i,j] = 100000
	merge_check = 0
	#print d.shape
	
	for i in range(len(d[0])):
		for j in range(len(d[0])):
			if((d[i,j]<thresh)and(group[0,i]>group[0,j])):
				merge_check = 1
				mean[0,i] = (mean[0,i]*group[0,i] + mean[0,j]*group[0,j])/(group[0,i]+group[0,j])
				mean[1,i] = (mean[1,i]*group[0,i] + mean[1,j]*group[0,j])/(group[0,i]+group[0,j])
				group[0,i] = group[0,i] + group[0,j]	
				# np.delete usage: np.delete(item, index, axis(row0/col1))				
				new_mean = np.delete(mean,j,1) 
				mean = new_mean
				new_d = np.delete(d,j,1)
				d = new_d
				new_d = np.delete(d,j,0)
				d = new_d
				#print d.shape
				new_group = np.delete(group,j,1)
				group = new_group
				#print group.shape
				break
		if merge_check == 1:
			break
print 'cluster centers : \n', mean
print 'numbers in each cluster : \n', group
'''
# ---------------------------------------------- #
# ----- Local Operation for large clusters ----- #
# ---------------------------------------------- #
for i in range(len(group[0])):
	if (group[0][i] > 18000):
		y = int(round(mean[0,i]))
		x = int(round(mean[1,i]))
for i in range(h*w):
	if (cluster_label[i,2] > 10)and(cluster_label[i,2] < 100):
		print cluster_label[i,2]





'''

# ------------------------------------------------ #
# ---- Draw Center Points and Bounding Boxes ----- #
# ------------------------------------------------ #

# padding with constant zeros
constant = cv2.copyMakeBorder(test_img,40,40,40,40,cv2.BORDER_CONSTANT,value=[0,0,0])

print test_img.shape
for i in range(len(mean[0])):
	y = int(round(mean[0,i]))+40
	x = int(round(mean[1,i]))+40
	test_img[y,x] = 255
	cv2.rectangle(constant,(x-90,y-90),(x+90,y+90),(0,255,0),1)

cv2.imshow('Bounding Boxes', constant)
cv2.waitKey()
cv2.destroyAllWindows()



# --------------------------------- #
# ----- Calculate the Results ----- #
# --------------------------------- #

for i in range(len(mean[0])):
	# -------------------------------------------------- #
	# ----- Extract the Bounding Boxes into Images ----- #
	# -------------------------------------------------- #
	y = int(round(mean[0,i]))+40
	x = int(round(mean[1,i]))+40
	#print y,x
	tmp_img = constant[y-90:y+90,x-90:x+90]
	tmp_sz = len(tmp_img[0])
	print tmp_sz
	tmp_img = cv2.resize(tmp_img, None, fx=224/tmp_sz, fy=224/tmp_sz, interpolation=cv2.INTER_CUBIC)
	print tmp_img.shape

	# --------------------------------------------- #
	# ----- Feed the Data into the Classifier ----- #
	# --------------------------------------------- #
	print 'Calculating Pose'
	tmp_img = uint8(tmp_img)
	deploy.deploy(tmp_img)
	cv2.imshow('Image', tmp_img)
	cv2.waitKey()
	cv2.destroyAllWindows()


