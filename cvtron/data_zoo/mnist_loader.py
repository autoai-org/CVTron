import tensorlayer.files as tlf


def load_mnist(shape=(-1, 784), path='data'):
    return tlf.load_mnist_dataset(shape=path)
