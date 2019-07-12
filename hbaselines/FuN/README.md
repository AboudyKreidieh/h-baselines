Continuing on [dmakian's](https://github.com/dmakian) implementation of https://arxiv.org/abs/1703.01161

Since this is based on a deprecated universe-starter-agent, the following dependencies are required:

# Dependencies

* Python 2.7 or 3.5
* [Golang](https://golang.org/doc/install)
* [six](https://pypi.python.org/pypi/six) (for py2/3 compatibility)
* [TensorFlow](https://www.tensorflow.org/) 0.12
* [tmux](https://tmux.github.io/) (the start script opens up a tmux session with multiple windows)
* [htop](https://hisham.hm/htop/) (shown in one of the tmux windows)
* [gym](https://pypi.python.org/pypi/gym)
* gym[atari]
* libjpeg-turbo (`brew install libjpeg-turbo`)
* [universe](https://pypi.python.org/pypi/universe)
* [opencv-python](https://pypi.python.org/pypi/opencv-python)
* [numpy](https://pypi.python.org/pypi/numpy)
* [scipy](https://pypi.python.org/pypi/scipy)

# Contents of Repo

Two models are created and tested inside this repo. One is the Feudal Network and the other is a typical LSTM network. The reason behind this is to be able to benchmark the Feudal Network performance against a regular LSTM ( as was done in https://arxiv.org/abs/1703.01161 ) 
