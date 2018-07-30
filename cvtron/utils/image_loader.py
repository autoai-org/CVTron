#coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.io
import skimage.transform
from PIL import Image

# from cvtron.data_zoo.coco.pycocotools.mask import area, decode


def load_image(path, height, width):
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (height, width))
    return resized_img


def write_image(image, output):
    plt.axis('off')
    plt.imshow(image)
    plt.savefig(output, bbox_inches='tight')


def parseEncoding(boundaries):
    pairs = boundaries.split(' ')
    poly = np.array(pairs)

def load_image_into_numpy_array(image_file_path):
    image = Image.open(image_file_path)
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def get_image_size(image_file_path):
    image = Image.open(image_file_path)
    return image.size
