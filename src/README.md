# Baseline models for EPIC-SOUNDS

The code in this repo is a condensed clone from [https://github.com/facebookresearch/SlowFast](https://github.com/facebookresearch/SlowFast). It contains code to train, validate and compute test scores on the [EPIC-SOUNDS Dataset](https://epic-kitchens.github.io/epic-sounds/) for both [Auditory Slow-Fast](https://github.com/ekazakos/auditory-slow-fast) and [Self-Supervised Audio Spectrogram Transformer (SSAST).](https://github.com/YuanGongND/ssast)

## Pretrained Models

You can download our pretrained models on EPIC-SOUNDS:

- [Slow-Fast](https://github.com/ekazakos/auditory-slow-fast)
- [SSAST](https://github.com/YuanGongND/ssast)

## Setup

Requirements:

- [PyTorch 1.13.1](https://pytorch.org/)
    - CUDA 11.6: `conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia`
    - CUDA 11.7: `conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`
- [libroa](https://librosa.org/) `conda install -c conda-forge librosa`
- [wandb](https://wandb.ai/site) `conda install -c conda-forge wandb`
- [h5py](https://www.h5py.org/) `conda install -c anaconda h5py`
- [fvcore](https://github.com/facebookresearch/fvcore/) `conda install -c conda-forge fvcore`
- [simplejson](https://simplejson.readthedocs.io/en/latest/) `conda install -c conda-forge simplejson`
- [psutil](https://psutil.readthedocs.io/en/latest/) `conda install -c conda-forge psutil`
- [SpecAugment with Pytorch](https://github.com/zcaceres/spec_augment) `git clone https://github.com/pytorch/audio.git torchaudio; cd torchaudio; python setup.py install`

### Downloading EPIC-SOUNDS

The dataset in this codebase uses the [HDF5 version](https://epic-kitchens.github.io/epic-sounds/) of EPIC-SOUNDS. To install, complete the steps as follow:

- From the [annotation repository of EPIC-SOUNDS](https://github.com/epic-kitchens/epic-sounds-annotations) e.g. the parent directory of this `src` folder, download: `EPIC_Sounds_train.pkl`, `EPIC_Sounds_validation.pkl` and `EPIC_Sounds_recognition_test.pkl`. `EPIC_Sounds_train.pkl`, `EPIC_Sounds_validation.pkl` may be used for training and validation, whilst `EPIC_Sounds_recognition_test.pkl` can be used to generate an output that can be converted into a submission to the [Audio-Bassed Interaction Recognition Challenge](https://github.com/epic-kitchens/C9-epic-sounds).
- Download the HDF5 file [here](https://epic-kitchens.github.io/epic-sounds/)

**NOTE** For this directory, you should install and pass arugments for the `.pkl` annotation files only, rather than the `.csv` files.

## Training/validating on EPIC-SOUNDS

To fine-tune Slow-Fast on EPIC-Sounds, run:

```(python)
python tools/run_net.py \
--cfg configs/EPIC-Sounds/slowfast/SLOWFASTAUDIO_8x8_R50.yaml \
NUM_GPUS num_gpus \
OUTPUT_DIR /path/to/outpur_dir \
EPICSOUNDS.AUDIO_DATA_FILE /path/to/EPIC_audio.hdf5 \
EPICSOUNDS.ANNOTATIONS_DIR /path/to/annotations \
TRAIN.CHECKPOINT_FILE_PATH /path/to/SLOWFAST_EPIC_SOUNDS.pyth
```

To fine-tune Self-Supervised Spectrogram Transformer, run:

```(python)
python tools/run_net.py \
--cfg configs/EPIC-Sounds/ssast/SSAST_b_vit_p16.yaml \
NUM_GPUS num_gpus \
OUTPUT_DIR /path/to/outpur_dir \
EPICSOUNDS.AUDIO_DATA_FILE /path/to/EPIC_audio.hdf5 \
EPICSOUNDS.ANNOTATIONS_DIR /path/to/annotations \
TRAIN.CHECKPOINT_FILE_PATH /path/to/SSAST_EPIC_SOUNDS.pyth
```

To train either model from scratch, remove the argument `TRAIN.CHECKPOINT_FILE_PATH`. To train a linear probe model, add the argument `MODEL.FREEZE_BACKBONE True`

Similarly, to validate Slow-Fast on EPIC-Sounds, run:

```(python)
python tools/run_net.py \
--cfg configs/EPIC-Sounds/slowfast/SLOWFASTAUDIO_8x8_R50.yaml \
TRAIN.ENABLE False \
TEST.ENABLE True \
NUM_GPUS num_gpus \
OUTPUT_DIR /path/to/outpur_dir \
EPICSOUNDS.AUDIO_DATA_FILE /path/to/EPIC_audio.hdf5 \
EPICSOUNDS.ANNOTATIONS_DIR /path/to/annotations \
TEST.CHECKPOINT_FILE_PATH /path/to/experiment_dir/checkpoints/checkpoint_best.pyth
```

To validate Self-Supervised Spectrogram Transformer, run:

```(python)
python tools/run_net.py \
--cfg configs/EPIC-Sounds/ssast/SSAST_b_vit_p16.yaml \
TRAIN.ENABLE False \
TEST.ENABLE True \
NUM_GPUS num_gpus \
OUTPUT_DIR /path/to/outpur_dir \
EPICSOUNDS.AUDIO_DATA_FILE /path/to/EPIC_audio.hdf5 \
EPICSOUNDS.ANNOTATIONS_DIR /path/to/annotations \
TEST.CHECKPOINT_FILE_PATH /path/to/experiment_dir/checkpoints/checkpoint_best.pyth
```

To obtain model predictions on the test set for a given model, run:

```(python)
python tools/run_net.py \
--cfg configs/EPIC-Sounds/slowfast/SLOWFASTAUDIO_8x8_R50.yaml \
TRAIN.ENABLE False \
TEST.ENABLE True \
NUM_GPUS num_gpus \
OUTPUT_DIR /path/to/outpur_dir \
EPICSOUNDS.AUDIO_DATA_FILE /path/to/EPIC_audio.hdf5 \
EPICSOUNDS.ANNOTATIONS_DIR /path/to/annotations \
TEST.CHECKPOINT_FILE_PATH /path/to/experiment_dir/checkpoints/checkpoint_best.pyth \
EPICSOUNDS.TEST_LIST EPIC_Sounds_recognition_test_timestamps.pkl
```

If you are using [Slurm](https://slurm.schedmd.com/documentation.html) We have provided some example scripts in `slurm_scripts` folder where the following arguments must be filled in:

- `PATH_TO_EPIC_SOUNDS_CONDA_ENV`: Path to the epic_sounds conda environment you installed in the previous step.
- `PATH_TO_SPEC_AUGMENT`: Path to where you installed [SpecAugment](https://github.com/zcaceres/spec_augment)
- `PATH_TO_SLOWFAST_DIR`: Path this repositorys `slowfast` directory
- `OUTPUT_DIR: Where you wish to save the output from the scripts
- `ANNOTATIONS_DIR`: Path to the EPIC-SOUNDS annotation `.pkl` files in the parent directory of this `src` folder
- `PATH_TO_HDF5_FILE`: Path to the `EPIC_audio.hdf5` file.
- `PATH_TO_PRETRAINED_MODEL`: Path to pretrained model, if using.

You will also need to edit the `#SBATCH` parameters in the slurm script to suit your environment. You can submit training scripts for either model to the queue by `sbatch slurm_scripts/train_slowfast.sh` or `sbatch slurm_scripts/train_ssast.sh`, or you can run validation by submitting `sbatch slurm_scripts/validate_slowfast.sh` or `sbatch slurm_scripts/validate_ssast.sh`. Scores for the [Audio-Bassed Interaction Recognition Challenge](https://github.com/epic-kitchens/C9-epic-sounds) can be run by `sbatch slurm_scripts/test_slowfast.sh` or `sbatch slurm_scripts/test_ssast.sh`.
