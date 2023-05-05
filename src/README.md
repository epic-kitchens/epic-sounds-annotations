# Baseline models for EPIC-SOUNDS

The code in this repo is a condensed clone from [https://github.com/facebookresearch/SlowFast](https://github.com/facebookresearch/SlowFast). It contains code to train, validate and compute test scores on the [EPIC-SOUNDS Dataset](https://epic-kitchens.github.io/epic-sounds/) for both [Auditory Slow-Fast](https://github.com/ekazakos/auditory-slow-fast) and [Self-Supervised Audio Spectrogram Transformer (SSAST)](https://github.com/YuanGongND/ssast).

**NOTE:** The output predictions of these models are put into a `.pkl` format, which can be converted into a JSON submittable format for the EPIC-SOUNDS: Audio-based interaction recognition challenge [here](https://github.com/epic-kitchens/C9-epic-sounds).

## Pretrained Models

You can download our pretrained models on EPIC-SOUNDS:

- [SlowFast](https://www.dropbox.com/s/339zsc6kz6c3wz9/SLOWFAST_EPIC_SOUNDS.pyth?dl=0)
- [SSAST](https://www.dropbox.com/s/p0wgjl5akmshfha/SSAST_EPIC_SOUNDS.pyth?dl=0)

You can also download the pretrained models on VGG (SlowFast) and AudioSet+LibriSpeech (SSAST):
- [SlowFast (VGG)](https://www.dropbox.com/home/EPIC-SOUNDS%20Pretrained%20Models?preview=SLOWFAST_VGG.pyth)
- [SSAST (AudioSet+LibriSpeech)](https://www.dropbox.com/home/EPIC-SOUNDS%20Pretrained%20Models?preview=SSAST-Base-Patch-400.pth)

## Setup

Requirements:

If using conda, you can install the requirements with the following commands in you own conda environment. Rember to activate this environment with `conda activate epic-sounds`, or a similar name.

- [Python 3.9](https://www.python.org/) `conda install python=3.9`
- [PyTorch](https://pytorch.org/)
  - CUDA 11.6: `conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia`
- [librosa](https://librosa.org/) `conda install -c conda-forge librosa`
- [wandb](https://wandb.ai/site) `conda install -c conda-forge wandb`
  - [tensorboard](https://www.tensorflow.org/tensorboard) `conda install -c conda-forge tensorboard`
- [h5py](https://www.h5py.org/) `conda install -c anaconda h5py`
- [fvcore](https://github.com/facebookresearch/fvcore/) `conda install -c fvcore -c iopath -c conda-forge fvcore`
  - [iopath](https://github.com/facebookresearch/iopath) `conda install -c iopath iopath`
- [simplejson](https://simplejson.readthedocs.io/en/latest/) `conda install -c conda-forge simplejson`
- [psutil](https://psutil.readthedocs.io/en/latest/) `conda install -c conda-forge psutil`
- [pandas](https://pandas.pydata.org/pandas-docs/stable/index.html) `conda install pandas`
- [timm](https://huggingface.co/docs/timm/index) `conda install -c conda-forge timm`

You will also need to export the `slowfast` directory to your Python path with:

```(python)
export PYTHONPATH=<path-to-epic-sounds-annotations-directory>/slowfast:$PYTHONPATH
```

### Downloading EPIC-SOUNDS

The dataset in this codebase uses a HDF5 Audio Dataset containing all the raw audio samples from the [EPIC-KITCHENS-100](https://epic-kitchens.github.io/2022) videos. To install, complete the steps as follow:

- From the [annotation repository of EPIC-SOUNDS](https://github.com/epic-kitchens/epic-sounds-annotations) e.g. the parent directory of this `src` folder, download: `EPIC_Sounds_train.pkl`, `EPIC_Sounds_validation.pkl` and `EPIC_Sounds_recognition_test_timestamps.pkl`. `EPIC_Sounds_train.pkl`, `EPIC_Sounds_validation.pkl` may be used for training and validation, whilst `EPIC_Sounds_recognition_test_timestamps.pkl` can be used to generate an output that can be converted into a submission to the [Audio-Bassed Interaction Recognition Challenge](https://github.com/epic-kitchens/C9-epic-sounds).
- Extract the untrimmed audios from the videos, using the downloader [here](https://github.com/epic-kitchens/download-scripts-100). Instructions on how to extract and format the audio into a HDF5 dataset can be found on the [Auditory SlowFast](https://github.com/ekazakos/auditory-slow-fast) GitHub repo. Alternatively, you can email [uob-epic-kitchens@bristol.ac.uk](mailto:uob-epic-kitchens@bristol.ac.uk)  for access to an existing HDF5 file.

**NOTE:** For this directory, you should install and pass arugments for the `.pkl` annotation files only, rather than the `.csv` files.

## Training/validating on EPIC-SOUNDS

**NOTE:** By default, weights and biases is disabled for tracking training runs. If enabled, it requires an internet connection on your machine/node for an online run by default. If this is not possible, but you still wish to track the model, you can run an offline version which you can later sync to weights and biases by updating the environment variable before the python command e.g. `WANDB_MODE=offline python ...` and passing the argument `WANDB.ENABLE True`.

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

To train either model from scratch, remove the argument `TRAIN.CHECKPOINT_FILE_PATH`. To train a linear probe model, add the argument `MODEL.FREEZE_BACKBONE True`.

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
