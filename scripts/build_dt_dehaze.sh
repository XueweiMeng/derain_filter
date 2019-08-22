#!/bin/bash
PATH_TO_DERAIN_DATASET='/path/to/dataset'
cd ..
#dehaze
python build_dt.py --images_folder=defog_dataset/train_datasets --dataset_folder=defog_dataset/train_defog --mode=patch --crop_size=64 --stride=64 #for train
python build_dt.py --images_folder=defog_dataset/test_datasets --dataset_folder=defog_dataset/test_defog --mode=full  # for test

cd ./scripts
