#!/bin/bash
read -r -p"This script will install Object Detection API, Are you sure? [Y/n]" response
response=${response,,}
if [[ $response =~ ^(yes|y| ) ]]; then
    echo 'git clone https://github.com/tensorflow/models.git'
    git clone https://github.com/tensorflow/models.git tmp/
    cp -rf tmp/.git object_detection/
    cd object_detection
    git checkout -f master
    rm -rf tmp
    echo 'protoc object_detection/protos/*.proto --python_out=.'
    protoc object_detection/protos/*.proto --python_out=.
fi