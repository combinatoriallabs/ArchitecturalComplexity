# Architectural Complexity
This is the public-facing repo for code associated to the paper "On the Architectural Complexity of Neural Networks".

# Setup
## Notes
This code was developed for use on linux (specifically, ubuntu 24 and centos 10) and may **not work well on other operating systems**. 

## First Steps
* Clone this repository somewhere and `cd` into the main repo folder.
* Ensure python 3.12 is installed, then run: `python3 -m pip install -r requirements.txt`
* `cd` into `Scripts/`.
* Run `python3 ./VisualizeDataset.py`.
    - You will be prompted to enter a dataset path. The codebase will use the provided path to store dataset files. **Datasets require ~10gb disk space**.
    - You will see notices about missing weights. Instructions for downloading the weights are given below. **Weights require ~20gb disk space**.
* If setup was successful, this will cycle through each combination of dataset and baseline architecture and display scatter plots of all samples.

## Downloading Weights
This repository contains only the diagnostic data collected from training each sampled architecture. To experiment with the trained weights, download the .zip archive from here. Extract and overwrite (or merge, if you have run experiments since cloning this repo) the directories. This will place various `Model.pth` files into their corresponding hash id folders inside the `Dataset/` directory.

# Usage
## Default Mode
When run without any arguments, `VisualizeDataset.py` displays all sampled architectures on a scatter plot w/ a linear parameter axis vs. top-1 accuracy. Samples are color-coded according to their operation complexity, and shape coded according to their arity complexity. Click on any scatter point to display the TEM, TOMs, and other information about the sample.

## Parameter vs. Accuracy Mode
Run `python3 ./VisualizeDataset.py -PA` to use parameter vs. accuracy mode. This opens a plot of the samples from the specified dataset/model combo superimposed with the baseline performance numbers.

## Analysis Mode
Run `python3 ./VisualizeDataset.py -x {INDEPENDENT_VARIABLE}` to open a combined plot of all dataset/model combos displayed with the X-axis set to `INDEPENDENT_VARIABLE`. The supported variables are:
* Params -- number of parameters.
* OpC -- operation complexity.
* TC -- tensor complexity.
* AC -- arity complexity.
* OC -- order complexity.
* CAC -- coupling arity complexity.
* NL -- number of non-linear activations.

### Notes
All modes accept the additional arguments: `-m {MODEL} -d {DATASET}`. `MODEL` should be either `ResNet` or `SWIN`, whereas `DATASET` should be either `CIFAR10`, `CIFAR100F`, or `TinyImagenet`.

## Training Models
This codebase is based on experiment-to-disk hashmaps. There is one such hashmap for each dataset/model combo. Additionally, the `Trainer/` folder is setup as a reletively empty experiment cache. Any models trained after cloning this repository will be stored in this cache by default.

The `Train.py` script is the starting point for conducting new experiments. It is suggested to open this script in your editor of choice, as a basic config is loaded there. After modifying this config to your liking, run `python ./Train.py`. If you have already trained a model for that configuration, the training run information will be displayed. If not, you will be prompted to launch a training run for the configuration. You can also pass `-Y` to `Train.py` to automatically train missing configurations.

By default, Train.py is setup to retrain the "red star" architecture. Simply run `python ./Train.py` if you want to reproduce results for this sample. If you want to do something else, follow the instructions in the sections below.

### Re-training Sampled Architectures
* Obtain the ADFNN ID for the sampled architecture you wish to re-train. These can be obtained by clicking on the scatter plots in default mode. Here are the IDs required to reproduce the results:
    * Red Star Architecture: 3cad4fc28af6344afe380ee3bd9092a7
    * Random samples:
        * CIFAR10
            * ResNet Good: 1b9ac8126536997fc35f1a54cd80b8f0
            * ResNet Bad: afaffe46bf7b5fe1abb75c74383f6315
            * Swin Good: 8e9da473fe9c90377b0bfe333d9f334b
            * Swin Bad: 3929dbffcfffee00ab68f02ae605dfbd
        * CIFAR100
            * ResNet Good: 7745a06402c0629c772e76f13380829e
            * ResNet Bad: b506540fe79ac1fae4ecdadc0cc4bc00
            * Swin Good: 8d3cb2f252dbea0ea1cb706f4477d2a4
            * Swin Bad: 14241a9d9d78b4fc6cd2368f3f1beaaf
        * Tiny Imagenet
            * ResNet Good: f5adf50b4860b2ded3b2da087a5cae0c
            * ResNet Bad: 044dc36d105057f7ea5ecd497d8c2efb
            * Swin Good: d5fe2790f8fa23c7a08372289c5466ba
            * Swin Bad: be0fd9efd2bb0bb5da4fc89446de7b49
* Ensure the `Model` key in the `Train.py` config on line 18 is set to `"ADFNN"`.
* Enter the dataset on line 28.
* Enter the ID to the `ADFNNModelID` key in the `Train.py` config on line 20.
    * Set the `ADFNNCacheDir` key on line 16 appropriately. For example, if you wish to re-train an architecture from the (`CIFAR10`, `ResNet34H`) setting, set the path to `../Dataset/CIFAR10/ResNet34H/ADFNNCache/`.
* Set the `ADFNNInputModule` according to the baseline architecture. This is accomplished by (un)commenting lines 22/23.
* Set the `ADFNNOutputModule` according to the baseline architecture. This is accomplished by (un)commenting lines 24/25. Use `"None"` for ResNet samples, and `"AvgDim0"` for SWIN samples.
    * **WARNING:** failure to set the input/output modules correctly for your chosen ADFNN ID *will* result in shape errors!
* Verify that the optimization parameters are set correctly. The max learning rate can be adjusted on lines 37 and 38. Values of 0.0075/0.001 were used for the ResNet/SWIN samples, respectively.

### Re-training Baselines
Re-training baseline models is much easier. Simply change the `Model` key on line 18 to whichever architecture you are interested in. Modify any other values, e.g., `Dataset`, `LearningRate`, then run `python ./Train.py`. Names for all the width-scaled models used to compute the parameter-accuracy curves can be found in the `ADFNNSearchParams.py` file.

### Running New Experiments
The experiment-to-disk hashmap was designed to encourage experimentation by eliminating any need to worry about storing/organizing the results. It is encouraged to modify configs, train models, and extend this codebase. If you make any interesting observations, do let us know!