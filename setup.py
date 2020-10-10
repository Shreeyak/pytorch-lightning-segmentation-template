#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='seg-lapa',
    version='0.1.0',
    description='Semantic Segmentation on the LaPa dataset using Pytorch Lightning',
    author='shreeyak',
    author_email='shreeyak.sajjan@gmail.com',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url='https://github.com/Shreeyak/pytorch-lightning-segmentation-lapa',
    install_requires=[
        'pytorch-lightning>=1.0.0rc2',
        'torch==1.6.0',
        'torchvision==0.7.0'
    ],
    packages=find_packages(),
)
