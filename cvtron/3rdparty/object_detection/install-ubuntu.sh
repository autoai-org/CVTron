#!/bin/bash
read -r -p"This script will install Object Detection API, Are you sure? [Y/n]" response
response=${response,,}
if [[ $response =~ ^(yes|y| ) ]]; then
    echo 'protoc object_detection/protos/*.proto --python_out=.'
    protoc object_detection/protos/*.proto --python_out=.
fi