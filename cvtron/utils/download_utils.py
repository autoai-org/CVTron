#coding:utf-8
from tqdm import tqdm 
import requests
import os 
from urllib.request import urlopen
import sys
sys.path.append('..')
from model_zoo.constant import VGG_NPY_URL
from model_zoo.constant import INCEPTION_CKPT_URL

def download(url,path):
    file_size = int(urlopen(url).info().get('Content-Length', -1))
    if not os.path.exists(path):
        os.makedirs(path)
    filename = url.split('/')[-1]
    dest = os.path.join(path,filename)
    if os.path.exists(dest):
        first_byte = os.path.getsize(dest)
    else:
        first_byte = 0
    if first_byte >= file_size:
        return file_size
    header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
    pbar = tqdm(
        total=file_size, initial=first_byte,
        unit='B', unit_scale=True, desc=filename)
    req = requests.get(url, headers=header, stream=True)
    with(open(dest, 'ab')) as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(1024)
    pbar.close()
    return file_size

def download_vgg_19(path=os.path.expanduser('~/.cvtron/model_zoo')):
    download(VGG_NPY_URL,path)

def download_inception_v3(path=os.path.expanduser('~/.cvtron/model_zoo')):
    download(INCEPTION_CKPT_URL,path)