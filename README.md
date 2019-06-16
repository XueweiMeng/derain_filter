# Derain Filter in FFmpeg

This repo contains TensorFlow implementations of following single image deraining models:
* SCAN &mdash; "Recurrent Squeeze-and-Excitation Context Aggregation Net for Single Image Deraining" [[arxiv]](https://arxiv.org/abs/1807.05698)

This repo is a part of GSoC project for derain filter in FFmpeg.

## Prerequisite
1. Python>=3.6
2. Opencv>=3.1.0
3. Tensorflow>=1.8.0
4. numpy>=1.12.1
5. tqdm

## Dataset Preparation
You can download the derain dataset from http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html for training or testing.

Rain100H: [http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html]<br>

We concatenate the two images(B and O) together as default inputs. If you want to change this setting, just modify dataset.py.
Moreover, there should be three folders 'train', 'val', 'test' in the dataset folder.
After download the datasets, don't forget to transform the format!

## Model training
1. Prepare Rain100H dataset
2. Put the dateset in dir 'datasets/Rain100H', there should be three folders 'train', 'val', 'test' in the dir
3. cd scripts
4. ./build_dt.sh
5. ./train_eval.sh

## Model generation

1. cd scripts
2. ./export_model.sh

## Benchmark results

This test set is produced with generate_datasets.sh script and consists of test part of DIV2K dataset.

Model | PSNR  | SSIM  |
----- | :---: | :---: |
CAN   | 23.23 | 70.68 |

## Usage of derain filter
The model file for derain filter in FFmpeg is models_for_test/derain_RESCAN.model. It is a native version model, so can be used in FFmpeg derain filter directly by the following command (The images in "testsets" dir can be used as the test images):

    ffmpeg -i derain_input.mp4 -vf derain=model=derain_RESCAN.model derain_output.mp4 (Native)

## Thanks to the Third Party Libs
[SR](https://github.com/HighVoltageRocknRoll/sr)
[RESCAN](https://github.com/XiaLiPKU/RESCAN)
