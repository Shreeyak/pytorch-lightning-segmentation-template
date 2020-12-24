#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="seg_lapa",
    version="0.2.0",
    description="Semantic Segmentation on the LaPa dataset using Pytorch Lightning",
    author="john doe",
    author_email="example@gmail.com",
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url="https://github.com/Shreeyak/pytorch-lightning-segmentation-template",
    python_requires=">=3.8",
    install_requires=[
        "pytorch-lightning==1.1.2",
        "torch==1.7.1",
        "torchvision==0.8.2",
        "gdown==3.12.2",
        "albumentations==0.5.2",
        "opencv-python==4.4.0.44",
        "hydra-core==1.0.4",
        "wandb==0.10.12",
        "pydantic==1.7.3",
    ],
    packages=find_packages(),
)
