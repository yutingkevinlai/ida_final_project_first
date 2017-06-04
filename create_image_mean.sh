#!/usr/bin/env sh
# compute te mean image from the training lmdb

TOOLS=/home/kevin/Downloads/caffe/build/tools
DIR=/home/kevin/caffe_practice/ida_final_project

$TOOLS/compute_image_mean $DIR/training_lmdb $DIR/ida_mean.binaryproto

echo "Done."
