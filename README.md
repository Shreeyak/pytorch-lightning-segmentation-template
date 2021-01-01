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
```shell script
# clone project  
git clone git@github.com:Shreeyak/pytorch-lightning-segmentation-lapa.git

# install project in development mode
cd pytorch-lightning-segmentation-lapa
pip install -e .  

# Setup git precommits
pip install -r requirements-dev.txt
pre-commit install
```  

#### Note: Cuda 11, Dec 2020
As of Dec 2020, systems using Cuda 11 (such as those with Ampere GPUs)
need to use different syntax to install pytorch. For such systems, install
the correct version of pytorch using:

```shell script
pip install -r requirements-cuda11.txt
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

### Callbacks

The callbacks can be configured from the config files or
[command line overrides](https://hydra.cc/docs/next/advanced/override_grammar/basic/).
To disable a config, simply remove them from the config. More callbacks can easily be added to the config system
as needed. The following callbacks are added as of now:

- [Early Stopping](https://pytorch-lightning.readthedocs.io/en/latest/generated/pytorch_lightning.callbacks.EarlyStopping.html#pytorch_lightning.callbacks.EarlyStopping)
- [Model Checkpoint](https://pytorch-lightning.readthedocs.io/en/latest/generated/pytorch_lightning.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint)
- [Log Media](#logmedia)  

CLI override Examples:

```shell script
# Disable the LogMedia callback.
python -m seg_lapa.train "~callbacks.log_media"

# Set the EarlyStopping callback to wait for 20 epochs before terminating.
python -m seg_lapa.train callbacks.early_stopping.patience=20
```

#### LogMedia

The LogMedia callback is used to log media, such as images and point clouds, to the logger and to local disk.
It is also used to save the config files for each run. The `LightningModule` adds data to a queue, which is
fetched within the `LogMedia` callback and logged to the logger and/or disk.

To customize this callback for your application, override or modify the following methods:

 - `LogMedia._get_preds_from_lightningmodule()`
 - `LogMedia._log_images_to_wandb()`
 - `LogMedia._save_results_to_disk()`

LogMedia currently supports the Weights and Biases logger only.

#### EarlyStopping

This is PTL's built-in callback. Here's some tips on how to configure early stopping:

```
Args:
        monitor: Monitor a key validation metric (eg: mIoU). Monitoring loss is not a good idea as it is an unreliable
                 indicator of model performance. Two models might have the same loss but different performance
                 or the loss might start increasing, even though performance does not decrease.

        min_delta: Project-dependent - choose a value for your metric below which you'd consider the improvement
                   negligible.
                   Example: For segmentation, I do not care for improvements less than 0.05% IoU in general.
                            But in kaggle competitions, even 0.01% would matter.

        patience: Patience is the number of val epochs to wait for to see an improvement. It is affected by the
                  ``check_val_every_n_epoch`` and ``val_check_interval`` params to the PL Trainer.

                  Takes experimentation to figure out appropriate patience for your project. Train the model
                  once without early stopping and see how long it takes to converge on a given dataset.
                  Choose the number of epochs between when you feel it's started to converge and after you're
                  sure the model has converged. Reduce the patience if you see the model continues to train for too long.
```

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
