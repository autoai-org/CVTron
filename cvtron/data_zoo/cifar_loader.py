import tensorlayer.files as tlf


def load_cifar10(shape=(-1, 32, 32, 3), path='data', plotable=False, second=3):
    return tlf.load_cifar10_dataset(shape=(-1, 32, 32, 3), path='data', plotable=False, second=3)


def load_cifar100(shape=(-1, 32, 32, 3), path='data', plotable=False, second=3):
    # TODO Fix Cifar 100 loader according to tensorlayer's implementation
    pass
