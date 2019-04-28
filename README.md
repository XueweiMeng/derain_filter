# Derain Filter in FFmpeg

This repo contains TensorFlow implementations of following single image deraining models:
* SCAN &mdash; "Recurrent Squeeze-and-Excitation Context Aggregation Net for Single Image Deraining" [[arxiv]](https://arxiv.org/abs/1807.05698)

This repo is a part of GSoC project for derain filter in ffmpeg.

## Dataset Preparation
You can download the derain dataset from http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html for training or testing.

## Model training
1. Prepare Rain100H dataset as repo [RESCAN](https://github.com/XiaLiPKU/RESCAN)
2. cd scripts
3. ./build_dt.sh
4. .train_eval.sh

## Model generation

1. cd scripts
2. ./export_model.sh

## Benchmark results

This test set is produced with generate_datasets.sh script and consists of test part of DIV2K dataset.

Model | PSNR  | SSIM  |
----- | :---: | :---: |
CAN   | 22.18 | 66.03 |

## Usage of derain filter
The model file for derain filter in FFmpeg is models_for_test/derain_RESCAN.model. It is a native version model, so can be used in FFmpeg derain filter directly by the following command (The images in "testsets" dir can be used as the test images):

    ffmpeg -i derain_input.mp4 -vf derain=model=derain_RESCAN.model derain_output.mp4

## Thanks to the Third Party Libs
[SR](https://github.com/HighVoltageRocknRoll/sr)
[RESCAN](https://github.com/XiaLiPKU/RESCAN)
