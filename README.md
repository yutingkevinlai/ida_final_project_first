# ida_final_project

This is a project of Intelligence Data Analysis course

## First: Single Object Pose Classification

### Step 1: Labels creation

Create two folders, [training_labels] and [testing_labels], and run

```
source create_lmdb.sh
```

You will see:
1. labels.txt in both [training_labels] and [testing_labels] folders, and they represent the labels of training and testing data respectively.

2. [training_lmdb] and [testing_lmdb] are the format we will give in our network.

### Step 2: Create mean binaryproto

We need to subtract the mean image, so type

```
source create_image_mean.sh
```

You will get [ida_mean.binaryproto] in the directory

### Step 3: Start training

We will use alexnet pre-trained model to finetune our weights, so in the directory, type

```
caffe train --solver=ida_solver.prototxt --weights=bvlc_alexnet.caffemodel |& tee ida_log.txt
```

the log will be write into ida_log.txt

### Step 4: Deploy caffemodel

1. We need to subtract image mean when apply our caffemodel, first type

```
python binaryproto_to_npy.py 
```

this will convert [ida_mean.binaryproto] into [ida_mean.npy]

2. Applying the caffemodel to test our network

```
python deploy.py YOUR/IMAGE/DIR
```

## Second: Multiple Object Poses Classification

```
cd img/
python detect.py YOUR/IMAGE/DIR
``` 

for example

```
python detect.py 4_0002.png
```





