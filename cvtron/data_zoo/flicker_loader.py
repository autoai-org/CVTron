import tensorlayer.files as tlf


def load_flicker_25K(tag='sky', path='data', n_threads=50, printable=False):
    return tlf.load_flickr25k_dataset(tag='sky', path='data', n_threads=50, printable=False)


def load_flicker_1M(tag='sky', path='data', n_threads=50, printable=False):
    return tlf.load_flickr1M_dataset(tag='sky', path='data', n_threads=50, printable=False)
