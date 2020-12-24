---

<div align="center">  

# Sementation Lapa  

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![code style: prettier](https://img.shields.io/badge/code_style-prettier-ff69b4.svg?style=flat-square)](https://github.com/prettier/prettier)

</div>


## Description  
This an example project showcasing Pytorch Lightning for semantic segmentation of the
[LaPa dataset](https://github.com/JDAI-CV/lapa-dataset) with Deeplabv3+.  

## Install  
First, install dependencies  
```bash
# clone project  
git clone git@github.com:Shreeyak/pytorch-lightning-segmentation-lapa.git

# install project in development mode
cd pytorch-lightning-segmentation-lapa
pip install -e .  

# Setup git precommits
pip install -r requirements-dev.txt
pre-commit install
```  

#### Developer dependencies
This repository uses git pre-commit hooks to auto-format code.
These developer dependencies are in requirements-dev.txt.
The other files describing pre-commit hooks are: `pyproject.toml`, `.pre-commit-config.yaml`


## Usage
Download the Lapa dataset at https://github.com/JDAI-CV/lapa-dataset  
It can be placed at `seg_lapa/data`.

Run training.  
 ```bash
# Run training
python -m seg_lapa.train dataset.data_dir=<path_to_data>  

# Run on multiple gpus
python -m seg_lapa.train dataset.data_dir=<path_to_data> train.gpus=\"0,1\"  
```

## Using this template for your own project
To use this template for your own project:
1. Search and replace `seg_lapa` with your project name
2. Edit setup.py with new package name, requirements and other details
3. Replace the model, dataloaders, loss function, metric with your own!
4. Update the readme! Add your own links to your paper at the top, add citation info at bottom.

This template was based on the Pytorch-Lightning
[seed project](https://github.com/PyTorchLightning/deep-learning-project-template).

### Notes
#### Absolute imports
This project is setup as a package. One of the advantages of setting it up as a
 package is that it is easy to import modules from anywhere.
 To avoid errors with pytorch-lightning, always use absolute imports:

```python
from seg_lapa.loss_func import CrossEntropy2D
from seg_lapa import metrics
import seg_lapa.metrics as metrics
```


### Citation  
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```  
