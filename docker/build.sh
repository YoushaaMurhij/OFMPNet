#!/bin/bash
docker build ./docker \
             -f docker/Dockerfile \
             -t x64/ofmpnet:latest 