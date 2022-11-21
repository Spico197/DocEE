#!/bin/bash

for filetype in train dev test
do
    wget -O Data/WikiEvents/$filetype.jsonl https://gen-arg-data.s3.us-east-2.amazonaws.com/wikievents/data/$filetype.jsonl
done
