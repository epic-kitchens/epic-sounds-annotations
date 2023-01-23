# EPIC-SOUNDS Dataset

We introduce [EPIC-SOUNDS](https://epic-kitchens.github.io/epic-sounds/), a large scale dataset of audio annotations capturing temporal extents and class labels within the audio stream of the egocentric videos from EPIC-KITCHENS-100. EPIC-SOUNDS includes 79.2k segments of audible events and actions, distributed across 44 classes. In this repository, we provide labelled temporal timestamps for the train / val split, and just the timestamps for the recognition test split. We train and evaluate two state-of-the-art audio recognition models on our dataset, which we also provide the code and pretrained models.

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
