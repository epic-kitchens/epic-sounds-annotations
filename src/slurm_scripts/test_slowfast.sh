#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1

# Set max wallclock time
#SBATCH --time=24:00:00

# Set name of job
#SBATCH --job-name=ASF_TEST

# Set number of GPUs
#SBATCH --gres=gpu:8

#SBATCH --mem=500GB

#SBATCH --cpus-per-task=40

# Set partition
#SBATCH --partition=big


module purge

# Run the application
nvidia-smi

# Initialise environment
PATH_TO_EPIC_SOUNDS_CONDA_ENV = ""
PATH_TO_SPEC_AUGMENT = ""
PATH_TO_SLOWFAST_DIR = ""

# Train args
OUTPUT_DIR = "SlowFast/EPIC-Sounds/baseline"
ANNOTATIONS_DIR = ""
PATH_TO_HDF5_FILE = ""
PATH_TO_PRETRAINED_MODEL = ""

source $PATH_TO_EPIC_SOUNDS_CONDA_ENV/activate epic_sounds
export PYTHONPATH=$PATH_TO_SPEC_AUGMENT:$PATH_TO_SLOWFAST_DIR:$PYTHONPATH

# Train the network
echo "Executing Code"
python -Wignore tools/run_net.py \
--cfg configs/EPIC-Sounds/slowfast/SLOWFASTAUDIO_8x8_R50.yaml \
TRAIN.ENABLE False \
TEST.ENABLE True \
NUM_GPUS 8 \
EPICSOUNDS.AUDIO_DATA_FILE $PATH_TO_HDF5_FILE \
EPICSOUNDS.ANNOTATIONS_DIR $ANNOTATIONS_DIR \
TEST.CHECKPOINT_FILE_PATH $PATH_TO_PRETRAINED_MODEL \
EPICSOUNDS.TEST_LIST EPIC_Sounds_recognition_test_timestamps.pkl
