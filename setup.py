#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='seg_lapa',
    version='0.1.0',
    description='Semantic Segmentation on the LaPa dataset using Pytorch Lightning',
    author='john doe',
    author_email='example@gmail.com',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url='https://github.com/Shreeyak/pytorch-lightning-segmentation-template',
    python_requires='>=3.8',
    install_requires=[
        'pytorch-lightning==1.0.8',
        'torch==1.6.0',
        'torchvision==0.7.0',
        'gdown==3.12.2',
        'albumentations==0.5.1',
        'opencv-python==4.4.0.44',
        'hydra-core==1.0.4',
        'wandb==0.10.11',
        'trafaret==2.1.0',
        'pydantic==1.7.2',
    ],
    packages=find_packages(),
)
