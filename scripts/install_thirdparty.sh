#!/usr/bin/env bash

mkdir thirdparty && cd thirdparty

# install flow
git clone https://github.com/flow-project/flow
pushd flow
git checkout 9eec578535508626c4823cdb79b779a3d3953202
pip install -e .
popd

# install sumo
pushd flow
if [[ "$OSTYPE" == "linux-gnu" ]]; then
    # Linus
    ./scripts/setup_sumo_ubuntu.sh
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # Mac OSX
    ./scripts/setup_sumo_osx.sh
fi
