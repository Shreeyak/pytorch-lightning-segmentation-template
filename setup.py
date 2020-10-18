#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='seg_lapa',
    version='0.1.0',
    description='Semantic Segmentation on the LaPa dataset using Pytorch Lightning',
    author='shreeyak',
    author_email='shreeyak.sajjan@gmail.com',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url='https://github.com/Shreeyak/pytorch-lightning-segmentation-lapa',
    install_requires=[
        'pytorch-lightning==1.0.2',
        'torch==1.6.0',
        'torchvision==0.7.0',
        'gdown==3.12.2',
        'albumentations==0.4.6',
        'opencv-python==4.4.0',
        'hydra-core==1.0.3',
        'wandb==0.10.7',
    ],
    packages=find_packages(),
)
