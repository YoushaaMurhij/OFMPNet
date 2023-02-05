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
```bash
python3 tools/data_preprocessing.py
```
### Training:
```bash
python3 tools/train.py --title experiment_title 
```
### Inference:
```bash
python3 tools/inference.py 
```
## License:
OFMPNet is released under MIT license (see [LICENSE](./LICENSE)). It is developed based on a forked version of STrajNet. We also used code from [OFPNet](https://github.com/YoushaaMurhij/OFPNet), [Swin-Transformer](https://github.com/microsoft/Swin-Transformer) and [FMFNet](https://github.com/YoushaaMurhij/FMFNet).

## Contact:
Questions and suggestions are welcome! </br>
Youshaa Murhij: yosha.morheg at phystech.edu
