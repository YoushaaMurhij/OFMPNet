# OFMPNet
Another Multi-modal Hierarchical Transformer for Occupancy Flow Field and Motion Prediction
 
![workflow](https://github.com/YoushaaMurhij/OFMPNet/actions/workflows/main.yml/badge.svg) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) ![Visitor count](https://shields-io-visitor-counter.herokuapp.com/badge?page=YoushaaMurhij/OFMPNet)

## Architecture:
![pipeline](./assets/OFMPNet-pipe.png)

## Abstract:
Motion prediction task is essential for autonomous driving systems and provides necessary information required to plan vehicles behaviour in the environment. 
Current motion prediction methods focus on predicting the future trajectory for each agent in the scene separately using its previous trajectory information.
In this work, we propose a end-to-end neural network method to predict all future behaviours for dynamic objects in the environment benefiting from the occupancy map and the motion flow of the scene. 
We are exploring various options for building a deep encoder-decoder model called OFMPNet, which takes as input a sequence of bird's-eye-view images with a road map, occupancy grid and previous motion flow. 
The model encoder can contain transformer, attention-based or convolutional units. The decoder considers the usage of both convolutional modules and recurrent blocks.
We also proposed a novel time weighted motion flow loss, the application of which demonstrated a significant reduction in end-point error.
On Waymo Occupancy and Flow Prediction benchmark, our approach achieved state-of-the-art results with 52.1\% Soft IoU and 76.75\% AUC on Flow-Grounded Occupancy.

## Main results:

## Use:
```bash
git clone OFMPNet
cd OFMPNet
bash docker/build.sh
bash docker/start.sh
bash docker/into.sh
```
### Data preprocessing:
Waymo Open Motion Dataset (WOD) is quite large dataset. Make Sure you have `+20TB` for the original and processed dataset.
Download and organize WOD dataset as follows:
```bash
.
└── waymo_open_dataset_motion_v_1_1_0
    └── uncompressed
        ├── occupancy_flow_challenge
        │   ├── testing_scenario_ids.txt
        │   ├── testing_scenario_ids.txt_.gstmp
        │   ├── validation_scenario_ids.txt
        │   └── validation_scenario_ids.txt_.gstmp
        ├── scenario
        │   ├── testing
        │   ├── testing_interactive
        │   ├── training
        │   ├── training_20s
        │   ├── validation
        │   └── validation_interactive
        └── tf_example
            ├── sample
            ├── testing
            ├── testing_interactive
            ├── training
            ├── validation
            └── validation_interactive
```
It is recommended to increase pooling number `--pool` in the arguments regarding your hardware specifications. 
```bash
python3 tools/data_preprocessing.py --pool 36
```
After running data preprocessing, the dataset should look like this:
```bash
.
└── waymo_open_dataset_motion_v_1_1_0
    └── uncompressed
        ├── occupancy_flow_challenge
        ├── preprocessed_data
        │   ├── test_numpy
        │   ├── train_numpy
        │   └── val_numpy
        ├── scenario
        └── tf_example

```
### Training:
```bash
python3 tools/train.py --title experinment_title 
```
### Inference:
```bash
python3 tools/inference.py --weight_path /path/to/weights
```

## Demo:
![M-Cross](./assets/sds_10.png)
![T-Cross](./assets/sds_0.png)
![Cross](./assets/sds_1.png)
## License:
OFMPNet is released under MIT license (see [LICENSE](./LICENSE)). It is developed based on a forked version of [STrajNet](https://github.com/georgeliu233/STrajNet). We also used code from [OFPNet](https://github.com/YoushaaMurhij/OFPNet), [Swin-Transformer](https://github.com/microsoft/Swin-Transformer) and [FMFNet](https://github.com/YoushaaMurhij/FMFNet).

## Contact:
Questions and suggestions are welcome! </br>
Youshaa Murhij: yosha.morheg at phystech.edu
