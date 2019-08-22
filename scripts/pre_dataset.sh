#!/bin/bash
cd ..

unzip ITS_v2/clear.zip -d ITS_v2
unzip ITS_v2/hazy.zip -d ITS_v2

python preDefogData.py --clear_folder=ITS_v2/clear --hazy_folder=ITS_v2/hazy

cd ./scripts
