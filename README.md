# OFMPNet
Another Multi-modal Hierarchical Transformer for Occupancy Flow Field and Motion Prediction
 
![workflow](https://github.com/YoushaaMurhij/OFMPNet/actions/workflows/main.yml/badge.svg) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) ![Visitor count](https://shields-io-visitor-counter.herokuapp.com/badge?page=YoushaaMurhij/OFMPNet)
## Demo:

## Abstract:

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
## License:
OFMPNet is released under MIT license (see [LICENSE](./LICENSE)). It is developed based on a forked version of [STrajNet](https://github.com/georgeliu233/STrajNet). We also used code from [OFPNet](https://github.com/YoushaaMurhij/OFPNet), [Swin-Transformer](https://github.com/microsoft/Swin-Transformer) and [FMFNet](https://github.com/YoushaaMurhij/FMFNet).

## Contact:
Questions and suggestions are welcome! </br>
Youshaa Murhij: yosha.morheg at phystech.edu
