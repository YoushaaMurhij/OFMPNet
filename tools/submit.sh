#!/bin/bash

method=$1
description=$2
split=$3
weight=$4
account=$5
author=$6

echo "Running inference script on ${split} set"
python3 tools/inference.py --method $method --description $description --split $split --weight_path $weight --account $account --author $author; 

echo "Compressing ${split} results for submission ..."
tar czvf /home/user/ofmpnet/Waymo_Dataset/inference/torch/${method}_${split}.tar.gz  \
      -C /home/user/ofmpnet/Waymo_Dataset/inference/torch/${split} .
echo "Deleting raw prediction protos ..."
rm -r /home/user/ofmpnet/Waymo_Dataset/inference/torch/${split}
echo "Compression is done. Good luck!"