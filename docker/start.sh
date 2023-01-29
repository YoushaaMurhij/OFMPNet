#!/bin/bash

cd "$(dirname "$0")"
cd ..

workspace_dir=$PWD

if [ "$(docker ps -aq -f status=exited -f name=ofpnet)" ]; then
    docker rm ofpnet;
fi

docker run -it -d --rm \
    --gpus all \
    --net host \
    -e "NVIDIA_DRIVER_CAPABILITIES=all" \
    -e "DISPLAY" \
    -e "QT_X11_NO_MITSHM=1" \
    --shm-size="40g" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --name ofpnet \
    -v $workspace_dir/:/home/user/ofpnet/:rw \
    -v /media/hdd/benchmarks/Waymo_Motion/waymo_open_dataset_motion_v_1_1_0/uncompressed/:/home/user/ofpnet/Waymo_Dataset/:rw \
    x64/ofpnet:latest 

docker exec -it --user "user" ofpnet \
    /bin/bash -c "
    export PYTHONPATH=\"${PYTHONPATH}:/home/user/ofpnet\";
    cd /home/user/ofpnet
    "