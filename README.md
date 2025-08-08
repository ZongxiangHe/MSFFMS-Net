
## How to run:

### Requirements
Code has been tested to work on:
+ Python 3.8
+ PyTorch 1.6, 1.8
+ CUDA 10.0, 10.1
+ using additional packages as listed in requirements.txt

### Datasets
* DAGM available [here.](https://www.kaggle.com/mhskjelvareid/dagm-2007-competition-dataset-optical-inspection)
* KolektorSDD available [here.](https://www.vicos.si/Downloads/KolektorSDD)
* KolektorSDD2 available [here.](https://www.vicos.si/Downloads/KolektorSDD2)
* Severstal Steel Defect Dataset available [here.](https://www.kaggle.com/c/severstal-steel-defect-detection/data)

Cross-validation splits, train/test splits and weakly/fully labeled splits for all datasets are located in `splits` directory of this repository, alongside the instructions on how to use them.

Results will be written to `./results-comind`  folders.

### Usage of training/evaluation code
The following python files are used to train/evaluate the model:
+ `train_net.py` Main entry for training and evaluation
+ `models.py` Model file for network
+ `data/dataset_catalog.py` Contains currently supported datasets

#### Running code

If you wish to do it the other way you can do it by running `train_net.py` and passing the parameters as keyword arguments.
Bellow is an example of how to train a model for a single fold of `KSDD` dataset.

    python -u train_net.py  \
        --GPU=0 \
        --DATASET=KSDD \
        --RUN_NAME=RUN_NAME \
        --DATASET_PATH=/path/to/dataset \
        --RESULTS_PATH=/path/to/save/results \
        --SAVE_IMAGES=True \
        --DILATE=7 \
        --EPOCHS=50 \
        --LEARNING_RATE=1.0 \
        --DELTA_CLS_LOSS=0.01 \
        --BATCH_SIZE=1 \
        --WEIGHTED_SEG_LOSS=True \
        --WEIGHTED_SEG_LOSS_P=2 \
        --WEIGHTED_SEG_LOSS_MAX=1 \
        --DYN_BALANCED_LOSS=True \
        --GRADIENT_ADJUSTMENT=True \
        --FREQUENCY_SAMPLING=True \
        --TRAIN_NUM=33 \
        --NUM_SEGMENTED=33 \
        --FOLD=0

Some of the datasets do not require you to specify `--TRAIN_NUM` or `--FOLD`-
After training, each model is also evaluated.

For KSDD you need to combine the results of evaluation from all three folds, you can do this by using `join_folds_results.py`:

    python -u join_folds_results.py \
        --RUN_NAME=SAMPLE_RUN \
        --RESULTS_PATH=/path/to/save/results \
        --DATASET=KSDD 
        
You can use `read_results.py` to generate a table of results f0r all runs for selected dataset.        

