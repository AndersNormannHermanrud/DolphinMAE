# Intro
This is the repo for my master thesis "A Novel In-Domain Pre-Training Approach Towards General Marine Mammal Classification"

It's a bit messy and as some architectures and methods were rapidly tested, sorry

## Experiment and Checkpoint Location
The files you are interested in are probably /MAE/* where the thesis experiment lies

Checkpoints used can be found at:
https://1drv.ms/u/c/b81db2433fab2ab7/EdPoSdJqxnxPgoPOUa-UfXgBvdXIjwSbaKsE71Ep5SPBcg?e=aAVUeM

Most others are legacy code or experiments from earlier project phases. If someone finds this useful I will probably clean up

## Setup
This uses AudioMAE as a baseline, so visit their GitHub and follow the setup.

In short what you need to do is:
* Set up the included env
* Download the right TIMM patch, I put in the /MAE/timm/ folder with local referencing for convenience
* Download audioMAE's util folder and put it as /MAE/util/
* To do an experiment go through the code at the top level and set paths, as they are currently local paths. No standard domain dataset exist so i did not standardize dataset loading
* Make yourself familiar with the code before running anything, pretraining takes weeks with models compute, so prepare accordingly

## Architecture
This is a pytorch lightning based experiment runner

First wav files are gathered from the defined directories iteratively, then segmented into spectrograms and fed to the model

Current structure are
1. pretrain_audiomae.py that sets the high level experiment and model configuration and is the main runner. Same for the finetune_audiomae.py experiments
2. That calls the respective *_dataloader.py that controls spectrogram settings and loading configuration

## Ending words
This is a bit of a skeleton, as it's not designed for widespread use per se, but to prove my thesis points. If you find this useful, are working with this for extended periods of time or have any questions contact me