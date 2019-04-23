#!/bin/bash
PATH_TO_DERAIN_DATASET='/data/lixia/Derain/Rain100H'
cd ..

 python build_dt.py --images_folder=/data/lixia/Derain/Rain100H/train --dataset_folder=datasets/train_Rain100H --mode=patch --crop_size=64 --stride=32
python build_dt.py --images_folder=/data/lixia/Derain/Rain100H/test --dataset_folder=datasets/test_Rain100H --mode=full

cd ./scripts
