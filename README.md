# Derain and Dehaze Filter in FFmpeg

This repo contains TensorFlow implementations of following single image deraining models:
* CAN &mdash; "Recurrent Squeeze-and-Excitation Context Aggregation Net for Single Image Deraining" [[arxiv]](https://arxiv.org/abs/1807.05698)

This repo is a part of GSoC project for derain and dehaze filter in FFmpeg.

## Prerequisite
1. Python>=3.6
2. Opencv>=3.1.0
3. Tensorflow>=1.8.0
4. numpy>=1.12.1
5. tqdm

## Dataset Preparation for Derain filter
1. Download the derain dataset from http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html. 
2. Download "Rain100H" for testing, get "rain_data_test_Heavy.gz"
3. Download "Train100H" for training, get "rain_data_train_Heavy.zip"
4. mkdir datasets
5. unzip rain_data_train_Heavy.zip -d datasets/Rain100H_train
6. mkdir datasets/Rain100H_test
7. tar -xvf rain_data_test_Heavy.gz -C datasets/Rain100H_test

## Dataset Preparation for Dehaze filter
1. Download the dehaze dataset from https://sites.google.com/view/reside-dehaze-datasets/reside-standard?authuser=0.
2. Download 'ITS(Indoor Training Set)' for training and testing, get "ITS_v2" dir
3. cd scripts
4. ./pre_dataset.sh 

## Model training
1. cd scripts
2. ./build_dt.sh (./build_dt_dehaze.sh for dehaze filter)
3. ./train_eval.sh (./train_eval_dehaze.sh for dehaze filter)

## Model generation
1. cd scripts
2. ./export_model.sh (./export_model_dehaze.sh for dehaze filter)

## Benchmark results

Model        | PSNR  | SSIM  |
---------    | :---: | :---: |
derain_CAN   | 23.30 | 70.62 |

## Usage of derain filter
Native model: The native model file for derain filter in FFmpeg is models_for_test/derain/derain_RESCAN.model. It can be used in FFmpeg derain filter directly by the following command (The images in "testsets/derain_dataset" dir can be used as the test images):

    ffmpeg -i derain_input.mp4 -vf derain=model=derain_RESCAN.model derain_output.mp4 (Native)

Tensorflow model: The tensorflow model file for derain filter in FFmpeg is models_for_test/derain/derain_RESCAN.pb. To enable this model you need to install the TensorFlow for C library (see https://www.tensorflow.org/install/install_c) and configure FFmpeg with --enable-libtensorflow. 

    ffmpeg -i derain_input.mp4 -vf derain=model=derain_RESCAN.pb:dnn_backend=1 derain_output.mp4 (Tensorflow)

## Usage of dehaze filter
Native model: The native model file for dehaze filter in FFmpeg is models_for_test/dehaze/dehaze_RESCAN.model. It can be used in FFmpeg dehaze filter directly by the following command (The images in "testsets/dehaze_dataset" dir can be used as the test images):
   
    ffmpeg -i dehaze_input.mp4 -vf derain=filter_type=1:model=dehaze_RESCAN.model dehaze_output.mp4 (Native)

Tensorflow model: The tensorflow model file for dehaze filter in FFmpeg is models_for_test/dehaze/dehaze_RESCAN.pb. To enable this model you need to install the TensorFlow for C library(see https://www.tensorflow.org/install/install_c) and configure FFmpeg with --enable-libtensorflow.

    ffmpeg -i dehaze_input.mp4 -vf derain=filter_type=1:model=dehaze_RESCAN.pb:dnn_backend=1 dehaze_output.mp4 (Tensorflow)   



## Thanks to the Third Party Libs
[SR](https://github.com/HighVoltageRocknRoll/sr)
[RESCAN](https://github.com/XiaLiPKU/RESCAN)
