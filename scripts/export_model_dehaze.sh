#!/bin/bash
OUTPUT_DIR='./'
CKPT_DIR=logdir/train
CHANNEL_NUM=24
NET_DEPTH=7

cd ..

python3 export_dehaze.py --output_dir=$OUTPUT_DIR --ckpt_dir=$CKPT_DIR --channel_num=$CHANNEL_NUM --net_depth=$NET_DEPTH

cd ./scripts
