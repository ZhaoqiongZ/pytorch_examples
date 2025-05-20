# YOLOv5 Example of PyTorch* on Intel GPU

## Overview

This is a user guide for running [YOLOv5](https://github.com/ultralytics/yolov5) object detection with [PyTorch*](https://pytorch.org) on Intel GPU devices.

## Environment Setup

### PyTorch

Please follow [Getting Started on Intel GPU page in PyTorch document](https://pytorch.org/docs/stable/notes/get_start_xpu.html) to build up a PyTorch environment which can pass the sanity test.

```bash
python -c "import torch; import torchvision; print(torch.xpu.is_available());"
# 'True' is expected to be echoed
```

### YOLOv5

This step can be completed by cloning the YOLOv5 repository, applying the provided patch which includes necessary code changes to enable Intel GPU, then installing the required dependencies.

Download the provided `enable_xpu.patch` file on your local machine, then execute the following commands.

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
git checkout v7.0
git apply <PATH_OF_THE DOWNLOADED_PATCH_FILE>
pip install -r requirements.txt
```

## Running YOLOv5

### Training

```bash
python train.py
```

### Inference

```bash
python detect.py --source <PATH_OF_IMAGES_OR_VIDEO>
```

**Note:** These are simple example commands without specifying any arguments so that the default `yolov5s` model is used for training and inference and `coco128` is used as the default training dataset. Please read the `Usage` part at the top of the python files to understand how to config the arguments per your own needs.