# dgl_gat_nodeflow
This is a demo for training GATs on giant graphs with dgl nodeflow. We
use [GraphSAGE](http://snap.stanford.edu/graphsage/)'s reddit dataset for this example.

## Dependencies
- [dgl](https://github.com/dmlc/dgl)
    - You may need to install from source for the latest version.
- [PyTorch](https://pytorch.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [tensorboardX](https://github.com/lanpa/tensorboardX)
- [TensorFlow](https://www.tensorflow.org/) (for the use of tensorboard)

## Usage
`python reddit.py`

GPU will be used if available.

The training process can be monitored with tensorboard. To launch the tensorboard, do
`tensorboard --logdir=.`
