# Learning Deep Learning

https://ldlbook.com/

## Data

MINST:

```
mkdir -p data/minst
curl -L https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/t10k-images-idx3-ubyte.gz | gunzip > data/minst/t10k-images-idx3-ubyte
curl -L https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/t10k-labels-idx1-ubyte.gz | gunzip > data/minst/t10k-labels-idx1-ubyte
curl -L https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/train-images-idx3-ubyte.gz | gunzip > data/minst/train-images-idx3-ubyte
curl -L https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/train-labels-idx1-ubyte.gz | gunzip > data/minst/train-labels-idx1-ubyte
```
