#!/bin/bash
TRAIN_LOGDIR=logdir/train
TEST_LOGDIR=logdir/test
TRAIN_DATASET_PATH=datasets/train_Rain100H/dataset.tfrecords
TEST_DATASET_PATH=datasets/test_Rain100H/dataset.tfrecords
TRAIN_STEPS=20000
TEST_STEPS=100
TRAIN_SAVE_NUM=4
TEST_SAVE_NUM=10

BATCH_SIZE=64
CHANNEL_NUM=24
NET_DEPTH=7

LEARNING_RATE=5e-3
STEPS_PER_SAVE=1000
SHUFFLE_BUFFER_SIZE=3000
STEPS_PER_LOG=100

cd ../

python3 train.py --batch_size=$BATCH_SIZE --channel_num=$CHANNEL_NUM --net_depth=$NET_DEPTH --dataset_path=$TRAIN_DATASET_PATH --buffer_size=$SHUFFLE_BUFFER_SIZE --learning_rate=$LEARNING_RATE --num_steps=$TRAIN_STEPS --save_num=$TRAIN_SAVE_NUM --steps_per_log=$STEPS_PER_LOG --steps_per_save=$STEPS_PER_SAVE --logdir=$TRAIN_LOGDIR
python3 eval.py --batch_size=1 --channel_num=$CHANNEL_NUM --net_depth=$NET_DEPTH --dataset_path=$TEST_DATASET_PATH --ckpt_dir=$TRAIN_LOGDIR --num_steps=$TEST_STEPS --save_num=$TEST_SAVE_NUM --logdir=$TEST_LOGDIR

cd ./scripts
