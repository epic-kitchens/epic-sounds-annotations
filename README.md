# EPIC-SOUNDS Dataset

We introduce [EPIC-SOUNDS](https://epic-kitchens.github.io/epic-sounds/), a large scale dataset of audio annotations capturing temporal extents and class labels within the audio stream of the egocentric videos from EPIC-KITCHENS-100. EPIC-SOUNDS includes 79.2k segments of audible events and actions, distributed across 44 classes. In this repository, we provide labelled temporal timestamps for the train / val split, and just the timestamps for the recognition test split. We train and evaluate two state-of-the-art audio recognition models on our dataset, which we also provide the code and pretrained models.

A download script is provided for the videos [here](https://github.com/epic-kitchens/download-scripts-100). You will have to extract the untrimmed audios from these videos. Instructions on how to extract and format the audio into a HDF5 dataset can be found on the [Auditory SlowFast](https://github.com/ekazakos/auditory-slow-fast) GitHub repo. Alternatively, you can eamil [uob-epic-kitchens@bristol.ac.uk](mailto:uob-epic-kitchens@bristol.ac.uk) for access to an existing HDF5 file.

**Contact:** [uob-epic-kitchens@bristol.ac.uk](mailto:uob-epic-kitchens@bristol.ac.uk)

## Citing
When using the dataset, kindly reference:
```
@article{epicsounds2023,
   title={E{PIC}-{SOUNDS}: A LARGE-SCALE DATASET OF ACTIONS THAT SOUND},
   author={Huh, Jaesung and Chalk, Jacob and Kazakos, Evangelos and Damen, Dima and Zisserman, Andrew},
   journal={ArXiv},
   year={2023}
} 
```

Also cite the EPIC-KITCHENS-100 paper where the videos originate:
```
@article{Damen2022RESCALING,
           title={Rescaling Egocentric Vision: Collection, Pipeline and Challenges for EPIC-KITCHENS-100},
           author={Damen, Dima and Doughty, Hazel and Farinella, Giovanni Maria  and and Furnari, Antonino 
           and Ma, Jian and Kazakos, Evangelos and Moltisanti, Davide and Munro, Jonathan 
           and Perrett, Toby and Price, Will and Wray, Michael},
           journal   = {International Journal of Computer Vision (IJCV)},
           year      = {2022},
           volume = {130},
           pages = {33–55},
           Url       = {https://doi.org/10.1007/s11263-021-01531-2}
} 
```

## File Structure

#### EPIC_Sounds_train.csv

This CSV file contains the annotations for the training set and contains 8 columns:

| Column Name           | Type                       | Example        | Description                                                                   |
| --------------------- | -------------------------- | -------------- | ----------------------------------------------------------------------------- |
| `annotation_id`       | string                     | `P01_01_0`     | Unique ID for the annotation as a string with participant ID and video ID.    |
| `participant_id`      | int                        | `P01`          | ID of the participant (unique per participant).                               |
| `video_id`            | string                     | `P01_01`       | ID of the video where the segment originated from (unique per video).         |
| `start_timestamp`     | string                     | `00:00:02.466` | Start time in `HH:mm:ss.SSS` of the audio annotation.                         |
| `stop_timestamp`      | string                     | `00:00:05.315` | End time in `HH:mm:ss.SSS` of the audio annotation.                           |
| `description`         | string                     | `paper rustle` | Transcribed English description provided by the annotator.                    |
| `class`               | string                     | `rustle`       | Parsed class from the description.                                            |
| `class_id`            | int                        | `3`            | Numeric ID of the class.                                                      |

#### EPIC_Sounds_validation.csv

This CSV file contains the annotations for the validation set and contains 8 columns:

| Column Name           | Type                       | Example        | Description                                                                   |
| --------------------- | -------------------------- | -------------- | ----------------------------------------------------------------------------- |
| `annotation_id`       | string                     | `P01_01_0`     | Unique ID for the annotation as a string with participant ID and video ID.    |
| `participant_id`      | int                        | `P01`          | ID of the participant (unique per participant).                               |
| `video_id`            | string                     | `P01_01`       | ID of the video where the segment originated from (unique per video).         |
| `start_timestamp`     | string                     | `00:00:02.466` | Start time in `HH:mm:ss.SSS` of the audio annotation.                         |
| `stop_timestamp`      | string                     | `00:00:05.315` | End time in `HH:mm:ss.SSS` of the audio annotation.                           |
| `description`         | string                     | `paper rustle` | Transcribed English description provided by the annotator.                    |  
| `class`               | string                     | `rustle`       | Parsed class from the description.                                            |
| `class_id`            | int                        | `3`            | Numeric ID of the class.                                                      |

#### EPIC_Sounds_recognition_test_timestamps.csv

This CSV file contains the annotations for the testing set and contains 5 columns:

| Column Name           | Type                       | Example        | Description                                                                   |
| --------------------- | -------------------------- | -------------- | ----------------------------------------------------------------------------- |
| `annotation_id`       | string                     | `P01_01_0`     | Unique ID for the annotation as a string with participant ID and video ID.    |
| `participant_id`      | int                        | `P01`          | ID of the participant (unique per participant).                               |
| `video_id`            | string                     | `P01_01`       | ID of the video where the segment originated from (unique per video).         |
| `start_timestamp`     | string                     | `00:00:02.466` | Start time in `HH:mm:ss.SSS` of the audio annotation.                         |
| `stop_timestamp`      | string                     | `00:00:05.315` | End time in `HH:mm:ss.SSS` of the audio annotation.                           |

## License
All files in this dataset are copyright by us and published under the 
Creative Commons Attribution-NonCommerial 4.0 International License, found 
[here](https://creativecommons.org/licenses/by-nc/4.0/).
This means that you must give appropriate credit, provide a link to the license,
and indicate if changes were made. You may do so in any reasonable manner,
but not in any way that suggests the licensor endorses you or your use. You
may not use the material for commercial purposes.

## Disclaimer
EPIC-KITCHENS-55 and EPIC-KITCHENS-100 were collected as a tool for research in computer vision, however, it is worth noting that the dataset may have unintended biases (including those of a societal, gender or racial nature).
