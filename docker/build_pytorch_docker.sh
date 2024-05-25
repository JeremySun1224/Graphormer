#!/bin/bash

# script parameters
docker_image_name=$1
docker_image_version=$2

docker build -f pytorch.Dockerfile -t ${docker_image_name}:${docker_image_version} .
