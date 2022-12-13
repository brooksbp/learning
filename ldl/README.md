# Learning Deep Learning

https://ldlbook.com/

## Environment

Win10:

```
conda create --name tf python=3.9
conda activate tf
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install --upgrade pip
pip install "tensorflow<2.11"
```

## Data

MINST:

```
mkdir -p data/minst
curl -L https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/t10k-images-idx3-ubyte.gz | gunzip > data/minst/t10k-images-idx3-ubyte
curl -L https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/t10k-labels-idx1-ubyte.gz | gunzip > data/minst/t10k-labels-idx1-ubyte
curl -L https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/train-images-idx3-ubyte.gz | gunzip > data/minst/train-images-idx3-ubyte
curl -L https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/train-labels-idx1-ubyte.gz | gunzip > data/minst/train-labels-idx1-ubyte
```
