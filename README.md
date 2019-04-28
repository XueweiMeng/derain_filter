# Image super resolution

This repo contains TensorFlow implementations of following single image deraining models:
* SCAN &mdash; "Recurrent Squeeze-and-Excitation Context Aggregation Net for Single Image Deraining" [[arxiv]](https://arxiv.org/abs/1807.05698)

This repo is a part of GSoC project for super resolution filter in ffmpeg.

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

## Thanks to the Third Party Libs
[SR](https://github.com/HighVoltageRocknRoll/sr)
[RESCAN](https://github.com/XiaLiPKU/RESCAN)
