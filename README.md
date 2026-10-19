# Semantic Segmentation of UAV Imagery in Urban and Rural Scenes Based on Improved PIDNet

This repository contains the code, data splits, and training configurations for the paper of 'Semantic Segmentation of UAV Imagery in Urban and Rural Scenes Based on Improved PIDNet'.

## Method

The improved PIDNet is built on PIDNet-M and changes it at two points.

- **CA-enhanced PAPPM.** Coordinate attention (CA) reweights the output of PAPPM, the context module of the I branch, along its rows and columns.
- **Lightweight learnable semantic head.** Depthwise separable convolution (DSC) replaces the dense 3 × 3 convolution of the head, and depthwise convolutional upsampling (DCU) replaces the fixed ×8 bilinear interpolation with a learned kernel for each output map. All three prediction heads use this design.

## Models

| Model | Model ID | Backbone |
|---|---|---|
| UNet | `unet` | None |
| DeepLab V3+ | `deeplabv3plus` | MobileNetV2 |
| HRNet | `hrnet` | HRNet-W18-Small-v2, classifier at 1/8 resolution |
| PSPNet | `pspnet` | MobileNetV2 |
| BiSeNet | `bisenet` | ResNet-18 |
| PIDNet | `pidnet_m` | PIDNet-M |
| PIDNet + CA | `pidnet_m_ca` | PIDNet-M |
| PIDNet + CA + DSC | `pidnet_m_ca_dsc` | PIDNet-M |
| PIDNet + CA + DCU | `pidnet_m_ca_dcu` | PIDNet-M |
| PIDNet + CA + DSC + DCU (ours) | `pidnet_m_ca_dsc_dcu` | PIDNet-M |

## Datasets

### Caihua Village

The dataset consists of 594 tiles of 512 × 512 pixels from UAV images of Caihua Village, Chongzhou City, Sichuan Province, China, acquired in May 2023 at a ground sampling distance of 0.1 m. The tiles are in `datasets/sources/caihua/` and are grouped into five blocks, four of 110 tiles and one of 154. Each fold of the five-fold cross-validation uses three blocks for training, one for validation, and one for testing. The lists of fold *k* are in `datasets/caihua/rs<k>/`.

The class IDs are 0 Background, 1 Farmland, 2 Bare soil, 3 Road, 4 Woodland, 5 Building, 6 Water, and 7 Other. Other does not occur in every evaluation partition, so mIoU averages IDs 0–6, each computed from the full 8 × 8 confusion matrix. OA counts all valid pixels, including Other.

### UDD

[UDD](https://github.com/MarcWong/UDD) contains UAV images of urban scenes. Its 160 images were cut into 9,216 tiles of 512 × 512 pixels, giving 6,912 training, 1,170 validation, and 1,134 test tiles. Images with overlapping ground coverage are kept in the same partition.

**The UDD tiles are not included in this repository**, as the authors of the UDD paper state that the dataset is only for non-commercial use, so the tiles cannot be redistributed under this repository's license. Download it from the official repository. The lists `datasets/udd/{train,val,test}.lst` define the partitions and give the path of each tile under `datasets/sources/udd/`. The folder names in these paths do not indicate the partitions.

The label gray values 108, 119, 90, 16, and 0 map to IDs 0–4, which are Building, Vegetation, Road, Vehicle, and Background.

## Training

The reference environment is Linux with Python 3.12.3, PyTorch 2.8.0, and CUDA 12.8. `environment/requirements_snapshot.txt` lists all packages. All runs use 512 × 512 crops and a batch size of 8, train for 200 epochs on Caihua Village and 80 on UDD, and use seeds 34, 304, and 3004. The configuration of every run in the paper is in `configs/effective/`.

```bash
python run_training.py --dataset caihua --model pidnet_m_ca_dsc_dcu --seed 304 --fold 1 --output new_runs/caihua_rs1
```

```bash
python run_training.py --dataset udd --model pidnet_m_ca_dsc_dcu --seed 304 --output new_runs/udd
```

`--output` must be a new directory, and `--gpu N` selects the GPU.

## Repository structure

```text
.
├── run_training.py      # Train one model
├── run_evaluation.py    # Evaluate trained checkpoints
├── run_profiling.py     # Measure speed, memory, and FLOPs
├── inference_runtime/   # Code shared by evaluation and profiling
├── code/
│   ├── caihua/          # Models and training code for Caihua Village
│   └── udd/             # Models and training code for UDD
├── configs/             # Training configurations
├── datasets/            # Caihua Village data, split lists, and class mappings
├── pretrained/          # ImageNet weights
└── environment/         # Reference package versions
```

## License

Copyright (C) 2026 Zezhong Zheng, Zixiao Yang, Zixuan Teng, Shuang Yu, and Qiang Liu. This repository is released under the [GNU General Public License v3.0](LICENSE). It includes code adapted from other open-source projects, which keep their original copyright notices and licenses, as listed in [LICENSE](LICENSE). The Caihua Village dataset is not covered and cannot be redistributed under this license. It is available from [IEEE DataPort](https://ieee-dataport.org/documents/dataset-paper-semantic-segmentation-uav-imagery-urban-and-rural-scenes-based-improved), which gives its license.