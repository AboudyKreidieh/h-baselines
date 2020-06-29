#!/bin/bash

# directory of the current file
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Create a directory to store the pre-trained models in.
mkdir "$DIR/../pretrained" && pushd "$DIR/../pretrained"

# TD3
mkdir TD3 && pushd TD3
wget https://cher-results.s3.us-east-2.amazonaws.com/TD3/AntGather.zip && unzip AntGather.zip && rm AntGather.zip
wget https://cher-results.s3.us-east-2.amazonaws.com/TD3/AntMaze.zip && unzip AntMaze.zip && rm AntMaze.zip
wget https://cher-results.s3.us-east-2.amazonaws.com/TD3/highway-v1.zip && unzip i210-v1.zip && rm highway-v1.zip
popd

# HRL
mkdir HRL && pushd HRL
wget https://cher-results.s3.us-east-2.amazonaws.com/HRL/AntGather.zip && unzip AntGather.zip && rm AntGather.zip
wget https://cher-results.s3.us-east-2.amazonaws.com/HRL/AntMaze.zip && unzip AntMaze.zip && rm AntMaze.zip
wget https://cher-results.s3.us-east-2.amazonaws.com/HRL/highway-v1.zip && unzip highway-v1.zip && rm highway-v1.zip
popd

# HIRO
mkdir HIRO && pushd HIRO
wget https://cher-results.s3.us-east-2.amazonaws.com/HIRO/AntGather.zip && unzip AntGather.zip && rm AntGather.zip
wget https://cher-results.s3.us-east-2.amazonaws.com/HIRO/AntMaze.zip && unzip AntMaze.zip && rm AntMaze.zip
wget https://cher-results.s3.us-east-2.amazonaws.com/HIRO/highway-v1.zip && unzip highway-v1.zip && rm highway-v1.zip
popd

# HAC
mkdir HAC && pushd HAC
wget https://cher-results.s3.us-east-2.amazonaws.com/HAC/AntGather.zip && unzip AntGather.zip && rm AntGather.zip
wget https://cher-results.s3.us-east-2.amazonaws.com/HAC/AntMaze.zip && unzip AntMaze.zip && rm AntMaze.zip
wget https://cher-results.s3.us-east-2.amazonaws.com/HAC/highway-v1.zip && unzip highway-v1.zip && rm highway-v1.zip
popd

# CHER
mkdir CHER && pushd CHER
wget https://cher-results.s3.us-east-2.amazonaws.com/CHER/AntGather.zip && unzip AntGather.zip && rm AntGather.zip
wget https://cher-results.s3.us-east-2.amazonaws.com/CHER/AntMaze.zip && unzip AntMaze.zip && rm AntMaze.zip
wget https://cher-results.s3.us-east-2.amazonaws.com/CHER/highway-v1.zip && unzip highway-v1.zip && rm highway-v1.zip
popd

# Exit the pre-trained directory.
popd
