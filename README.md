# Derain Filter in FFmpeg

This repo contains TensorFlow implementations of following single image deraining models:
* CAN &mdash; "Recurrent Squeeze-and-Excitation Context Aggregation Net for Single Image Deraining" [[arxiv]](https://arxiv.org/abs/1807.05698)

This repo is a part of GSoC project for derain filter in FFmpeg.

## Prerequisite
1. Python>=3.6
2. Opencv>=3.1.0
3. Tensorflow>=1.8.0
4. numpy>=1.12.1
5. tqdm

## Dataset Preparation
1. Download the derain dataset from http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html. 
2. Download "Rain100H" for testing, get "rain_data_test_Heavy.gz"
3. Download "Train100H" for training, get "rain_data_train_Heavy.zip"
4. mkdir datasets
5. unzip rain_data_train_Heavy.zip -d datasets/Rain100H_train
6. mkdir datasets/Rain100H_test
7. tar -xvf rain_data_test_Heavy.gz -C datasets/Rain100H_test

## Model training
1. cd scripts
2. ./build_dt.sh
3. ./train_eval.sh

## Model generation

1. cd scripts
2. ./export_model.sh

## Benchmark results

Model | PSNR  | SSIM  |
----- | :---: | :---: |
CAN   | 23.23 | 70.68 |

## Usage of derain filter
The model file for derain filter in FFmpeg is models_for_test/derain_RESCAN.model. It is a native version model, so can be used in FFmpeg derain filter directly by the following command (The images in "testsets" dir can be used as the test images):

    ffmpeg -i derain_input.mp4 -vf derain=model=derain_RESCAN.model derain_output.mp4 (Native)

## Thanks to the Third Party Libs
[SR](https://github.com/HighVoltageRocknRoll/sr)
[RESCAN](https://github.com/XiaLiPKU/RESCAN)
