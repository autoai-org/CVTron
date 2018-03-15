import tensorlayer.files as tlf


def load_voc(path='data', dataset='2012', contain_classes_in_person=False):
    return tlf.load_voc_dataset(path='data', dataset='2012', contain_classes_in_person=False)
