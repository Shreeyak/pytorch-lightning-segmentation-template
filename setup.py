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
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)
