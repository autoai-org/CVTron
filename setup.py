from setuptools import setup, find_packages
from os import path
import sys


here = path.abspath(path.dirname(__file__))

install_requires = [
    'tensorflow',
    'tensorlayer',
    'tqdm',
    'requests'
]

setup(
    name='cvtron',
    version='0.0.1',
    description='Out-of-the-Box Computer Vision Library',
    url='https://github.com/cv-group',
    author='Xiaozhe Yao',
    author_email='ad@askfermi.me',
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='Computer Vision',
    packages=find_packages(exclude=['docs', 'tests*']),
    test_suite='nose.collector',
    install_requires=install_requires,
    extras_require={
    }
)