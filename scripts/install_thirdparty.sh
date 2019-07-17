#!/usr/bin/env bash

# all third party tools are install in the h-baselines conda environment
source activate h-baselines

mkdir thirdparty && cd thirdparty

# install flow
git clone https://github.com/flow-project/flow
pushd flow
git checkout 9eec578535508626c4823cdb79b779a3d3953202
pip install -e .
popd
