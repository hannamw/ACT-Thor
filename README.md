# ACT-Thor-Models
Repository for the Models' section in the ACT-Thor paper presented at COLING 2022.

The code for the main experiments is all in the `multimodal` folder: there can be found the main classes
for the transformation models (Action-Matrix, Concat-Linear and Concat-Multi) and the functions for training,
evaluation and testing.

The `visual_features` directory instead contains all utilities for the visual encoders and the `data` 
submodule provides the datastructures to load the dataset. The other files instead are used in the preliminary 
experiments of target object detection for testing visual feature extractors.

**Note**: when firstly loaded, the `VecTransformDataset` object will try to extract and store visual 
vectors for all images in the dataset with the specified visual feature extractor (MOCA or CLIP).


### Main methods

All the main methods can be launched as `multimodal/vector_transform.py` optional 
arguments (all the following are **mutually exclusive** and can be used 1 at a time):
* `--exp_fcn_hyperparams`: runs a grid search for the best performing Concat-Multi model;
* `--exp_hold_out`: for 2 feature extractors (MOCA, CLIP) and 3 model types (Action-Matrix, Concat-Linear
and Concat-Multi) tests the holding out procedures specified within the method by the `tested_procedures`
list variable;
* `--exp_regression`: for the hold-out procedures specified by the `tested_procedures` list variable and the 
2 visual feature extractors, perform an evaluation of the least-squares regression method 
for training action matrices of the Action-Matrix model; 
* `--exp_cs_size`: originally intended for testing contrast set of different sizes, now executes `exp_hold_out`
with a better plot and saving test set predictions and other results (preferable method for evaluation, used
for graphs contained in the paper);
* `--exp_nearest_neighbors`: while saving all after-vectors and predicted-vectors in `.pkl`
files, computes a ranking of all vectors by similarity with the prediction for each sample, for every model
saved as `.pth` in the specified top directory, while also computing some metrics (AP, MAP) and 
preparing images containing the first `k` ranked pictures for the 3 models for each sample;
* `--exp_action_crossval`: generates action-object associations and iterates, by fixing one action at a time,
across several objects available for that action, in order to test how generalization capabilities change
with regard to different unseen objects with a roughly similar seen set, for all models stored in the folder 
specified by the `--load_path` argument;
* `--exp_neighbor_mds`: projects all after-vectors and predicted-vectors in a 2D space with the MDS method,
to visualize their distribution in the feature space in relation with actions (represented as colors), for all
models that can be found in the directory specified by the `--load_path` argument.

All methods involving training (`exp_fcn_hyperparams`, `exp_hold_out`, `exp_regression`, `exp_cs_size`)
are by default run for **5 iterations**: this allows to obtain the result as an average of several 
statistical iterations, and eventually also get the variance for each model's performance. The number
of iterations can be changed with the `--statistical_iterations` argument.

Other arguments can set data/save paths and hyperparamenters for experiments (batch size, learning rate,
optimizer, use of Weights & Biases, ...); see more in the default configuration (`__default_train_config__`) 
in `multimodal/vector_transform.py`. 

The data folder can be changed with the `--data_path` argument or directly in 
the `__default_dataset_fname__` and `__default_dataset_path__` variables of `visual_features/data/__init__.py`. When
using the `--data_path` argument, please change also the annotation `.csv` file name to match the default name 
(`dataset_with_new_splits.csv`) or change the variable `__default_dataset_fname__` to your custom name.


In order to save model parameters after training, use the `--save_models` argument. They will be saved in separate
directories, along with their configuration parameters, in `.pth` format and divided by model type and holding-out
procedure used. 


### Note on `exp_nearest_neighbors`

This method creates a separate directory for each model ending in `'_neighbors'`. 
During the execution, both predicted-vectors and after-vectors will be saved as python dictionaries of lists
of PyTorch tensors, in the `.pkl` format used by `pickle`. However, when the ranking is computed it is saved
in a `.csv` file to be reused if needed.

**The `'_neighbors'` folders are needed for the execution of the two other analysis scripts: 
`exp_action_crossval` and  `exp_neighbor_mds`.** These scripts allow to load vectors from a specified directory
with the argument `--load_path`, which should contain all pre-computed vectors saved in `.pkl` format.


### Plots
The `plots.py` file contains methods for plotting all the figures in the paper; roughly,
each figure has its own method, plus other methods for graphs not reported in the paper. 
See the correspondent methods in `multimodal/vector_transform.py` for individual explanations.


### Requirements

Every model in this repository will automatically use CUDA if available.

System requirements:
* CUDA 11.4 (also tested with 10.6, but not guaranteed to work)
* Python 3.6

Run `pip install -r requirements.txt` for installing all dependencies.