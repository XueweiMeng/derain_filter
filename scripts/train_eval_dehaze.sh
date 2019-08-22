#!/bin/bash
FILTER_TYPE=dehaze
TRAIN_LOGDIR=logdir/train
TEST_LOGDIR=logdir/test
TRAIN_DATASET_PATH=defog_dataset/train_defog/dataset.tfrecords
TEST_DATASET_PATH=defog_dataset/test_defog/dataset.tfrecords
TRAIN_STEPS=120000
TEST_STEPS=200
TRAIN_SAVE_NUM=4
TEST_SAVE_NUM=10

BATCH_SIZE=64
CHANNEL_NUM=24
NET_DEPTH=7

LEARNING_RATE=1e-3
STEPS_PER_SAVE=1000
SHUFFLE_BUFFER_SIZE=3000
STEPS_PER_LOG=100

cd ../

python3 train.py --filter_type=$FILTER_TYPE --batch_size=$BATCH_SIZE --channel_num=$CHANNEL_NUM --net_depth=$NET_DEPTH --dataset_path=$TRAIN_DATASET_PATH --buffer_size=$SHUFFLE_BUFFER_SIZE --learning_rate=$LEARNING_RATE --num_steps=$TRAIN_STEPS --save_num=$TRAIN_SAVE_NUM --steps_per_log=$STEPS_PER_LOG --steps_per_save=$STEPS_PER_SAVE --logdir=$TRAIN_LOGDIR
python3 eval.py --filter_type=$FILTER_TYPE --batch_size=1 --channel_num=$CHANNEL_NUM --net_depth=$NET_DEPTH --dataset_path=$TEST_DATASET_PATH --ckpt_dir=$TRAIN_LOGDIR --num_steps=$TEST_STEPS --save_num=$TEST_SAVE_NUM --logdir=$TEST_LOGDIR

cd ./scripts
