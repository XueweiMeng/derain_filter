#!/bin/bash
PATH_TO_DERAIN_DATASET='/path/to/dataset'
cd ..

python build_dt.py --images_folder=datasets/Rain100H --dataset_folder=datasets/test_Rain100H --mode=full

cd ./scripts
