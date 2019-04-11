# derain_filter
This repo is a part of GSoC2019 project for derain filter in FFmpeg.

This repo contains TensorFlow implementations of image derain filter based on the following models (we made some modifications to ESPCN in order to make it work for derain task):
* ESPCN &mdash; "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network" [[arxiv]](https://arxiv.org/abs/1609.05158)

## Dataset Preparation
You can download the derain dataset from http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html for training or test.

## Model Usage
The model file for derain filter in FFmpeg is models_for_test/derain_espcn.model. It is a native version model, so can be used in FFmpeg derain filter directly by the following command:

    ffmpeg -i derain_input.mp4 -vf derain=model=derain_espcn.model derain_output.mp4
