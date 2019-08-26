#!/bin/bash
cd ..

mkdir datasets
unzip rain_data_train_Heavy.zip -d datasets/Rain100H_train
mkdir datasets/Rain100H_test
tar -xvf rain_data_test_Heavy.gz -C datasets/Rain100H_test

cd ./scripts
