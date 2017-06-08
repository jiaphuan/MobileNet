#!/bin/bash
#
# This script performs the following operations:
# 1. Trains a MobileNet model on the Imagenet training set.
# 2. Evaluates the model on the Imagenet validation set.
#
# Usage:
# ./scripts/train_mobilenet_on_imagenet.sh

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=/home/jiaphuan/exp/mobilenet201706080118

# Where the dataset is saved to.
DATASET_DIR=/home/linjuny/dataset/imagenet2012/

# Run training.
CUDA_VISIBLE_DEVICES=2 python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=imagenet \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=mobilenet \
  --preprocessing_name=mobilenet \
  --width_multiplier=1.0 \
  --max_number_of_steps=1000000 \
  --batch_size=128 \
  --save_interval_secs=240 \
  --save_summaries_secs=240 \
  --log_every_n_steps=100 \
  --optimizer=sgd \
  --rmsprop_decay=0.9 \
  --opt_epsilon=1.0\
  --learning_rate=0.1 \
  --learning_rate_decay_type=polynomial \
  --end_learning_rate=0. \
  --learning_rate_decay_factor=0.1 \
  --momentum=0.9 \
  --num_epochs_per_decay=100.0 \
  --weight_decay=0.0001 \
  --num_clones=1

# Run evaluation.
#CUDA_VISIBLE_DEVICES='' python eval_image_classifier.py \
#  --checkpoint_path=${TRAIN_DIR} \
#  --eval_dir=${TRAIN_DIR} \
#  --dataset_name=imagenet \
#  --dataset_split_name=validation \
#  --dataset_dir=${DATASET_DIR} \
#  --model_name=mobilenet
