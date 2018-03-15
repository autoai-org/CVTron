import tensorlayer.files as tlf


def load_cyclegan(filename='summer2winter_yosemite', path='data'):
    return tlf.load_cyclegan_dataset(filename='summer2winter_yosemite', path='data')
