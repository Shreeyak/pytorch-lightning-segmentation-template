---

<div align="center">    
 
# Sementation Lapa     

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->


<!--  
Conference   
-->   
</div>
 
## Description   
This an example project showcasing Pytorch Lightning for semantic segmentation of the 
[LaPa dataset](https://github.com/JDAI-CV/lapa-dataset) with Deeplabv3+.  

## How to run   
First, install dependencies   
```bash
# clone project   
git clone git@github.com:Shreeyak/pytorch-lightning-segmentation-lapa.git

# install project   
cd pytorch-lightning-segmentation-lapa
pip install -e .   
pip install -r requirements.txt
 ```   
Run training. The script will download the data to seg-lapa/data.   
 ```bash
# module folder
cd seg-lapa

# run module (example: mnist as your main contribution)   
python seg-lapa.py    
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from project.datasets.mnist import mnist
from project.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer

# model
model = LitClassifier()

# data
train, val, test = mnist()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
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
