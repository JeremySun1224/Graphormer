#!/bin/bash

# script parameters
docker_run_name=$1
docker_image_name=$2
docker_image_version=$3
ssh_port=$4
tensorboard=$5

#docker run --gpus all --cpus 48 --shm-size 16G --memory 500gb -itd \
docker run --gpus all --shm-size 16G -itd --memory 500gb\
  --ipc=host \
	--ulimit memlock=-1 \
	--ulimit stack=67108864 \
  -p ${ssh_port}:22 \
  -p ${tensorboard}:6006 \
  -v /etc/localtime:/etc/localtime \
  -v /root/workspace:/root/workspace \
  --name ${docker_run_name} \
  -e ROOT_PASS="yl123456" \
  ${docker_image_name}:${docker_image_version}


# A30
#   --ipc=host \
#	--ulimit memlock=-1 \
#	--ulimit stack=67108864 \


