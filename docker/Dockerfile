FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

RUN apt-get update && apt-get install wget -yq
RUN apt-get install build-essential g++ gcc -y
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -y && \
    apt-get autoremove -y && \ 
    apt-get install -y --no-install-recommends \
    python3 python3-pip \
    ca-certificates  lsb-release build-essential git && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install pip --upgrade
RUN python3 -m pip install numpy matplotlib tqdm 
RUN pip3 install waymo-open-dataset-tf-2-6-0 
RUN pip3 install tensorflow-addons
# RUN pip install keras==2.6
RUN pip3 install wandb
RUN pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113

ARG USER=user
ARG UID=1000
ARG GID=1000
# default password
ARG PW=user 

RUN useradd -m ${USER} --uid=${UID} && echo "${USER}:${PW}" | chpasswd && adduser ${USER} sudo

WORKDIR /home/${USER}
RUN cd /home/${USER} && mkdir -p ofmpnet && chown -R ${UID}:${GID} ./

USER ${UID}:${GID}
