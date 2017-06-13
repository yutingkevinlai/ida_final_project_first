# IDA Final Project First Part

This is a project of Intelligence Data Analysis course.

committed by Yu-Ting Lai

## First Part: One Object Pose Classification in A Random Stacking Scene

### Step 1: Labels creation

Create two folders, ```training_labels``` and ```testing_labels```, and run

```
source create_lmdb.sh
```

You will see:
1. ```labels.txt``` in both ```training_labels``` and ```testing_labels``` folders, and they represent the labels of training and testing data respectively.

2. ```training_lmdb``` and ```testing_lmdb``` folders, and these are the format we will give in our network.

### Step 2: Create mean binaryproto

We need to subtract the mean image, so type

```
source create_image_mean.sh
```

You will get ```ida_mean.binaryproto``` in the directory, this is the input format for Caffe to read mean image

### Step 3: Start training

We will use alexnet pre-trained model to finetune our weights, so in the directory, type

```
source train.sh
```

the log will be write into ```ida_log.txt```

### Step 4: Deploy caffemodel (verification)

1. We need to subtract image mean when apply our caffemodel, first type

```
python binaryproto_to_npy.py 
```

this will convert ```ida_mean.binaryproto``` into ```ida_mean.npy```

2. Verify the trained caffemodel

```
python deploy.py <PATH/TO/YOUR/IMAGE/DIR>/verify/xxxx.png(.jpg)
```

for example

```
python deploy.py img/verify/5_0011.jpg
```

### Step 5: Apply caffemodel to do pose classification for multi-object scene

```
cd <PATH/TO/YOUR/IMAGE/DIR>
python detect.py YOUR/IMAGE/DIR
``` 

for example

```
python detect.py 4_0002.png
```

## Second Part: Two Objects Pose Classification in A Random Stacking Scene

Please go to [here](https://github.com/KevinXlab/ida_final_project_two) for more information.



