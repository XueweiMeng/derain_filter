#!/bin/bash
PATH_TO_DERAIN_DATASET='/path/to/dataset'
cd ..
python build_dt.py --images_folder=datasets/Rain100H/train --dataset_folder=datasets/train_Rain100H --mode=patch --crop_size=64 --stride=32 #for train 
#python build_dt.py --images_folder=datasets/Rain100H --dataset_folder=datasets/test_Rain100H --mode=full  # for test

cd ./scripts
