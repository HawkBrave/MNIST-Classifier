from neuralnet import Model
import numpy as np


def fetch(url):
    import requests
    import gzip
    import hashlib
    import os

    fp = os.path.join("./dataset", hashlib.md5(url.encode('utf-8')).hexdigest())

    if os.path.isfile(fp):
        with open(fp, 'rb') as f:
            data = f.read()
    else:
        with open(fp, 'wb') as f:
            data = requests.get(url).content
            f.write(data)
    return np.frombuffer(gzip.decompress(data), dtype=np.uint8)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def load_mnist():
    x_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[16:].reshape((-1, 784))
    y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
    x_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[16:].reshape((-1, 784))
    y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

    x_train = [np.reshape(x, (784, 1)) for x in x_train]
    y_train = [vectorized_result(y) for y in y_train]
    x_test = [np.reshape(x, (784, 1)) for x in x_test]

    return (zip(x_train, y_train), zip(x_test, y_test))


training_data, test_data = load_mnist()
training_data = list(training_data)
test_data = list(test_data)

nn = Model([784, 30, 10])
nn.sgd(training_data, 30, 10, 0.1, test_data=test_data)
