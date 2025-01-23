# Introduction

This guide provides instructions for running classification and detection tasks using PyTorch on an Intel Client GPU (Windows Platform). It includes steps for installing necessary drivers and software, running ResNet classification with FP16 AMP, Faster R-CNN detection with FP16 AMP, and ResNet classification with INT8 PT2E.

### Step-by-Step Guide

#### Step 1: Install Intel Graphics Driver

1. **Download the driver**: Visit the [Intel Arc & Iris Xe Graphics Driver page](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html).
2. **Install the driver**: Follow the instructions on the page to download and install the driver on your system.

#### Step 2: Install PyTorch and other required packages

> [!NOTE]  
> We highly recommend installing an Anaconda environment. 

**Run the installation command**: Copy the command provided and run it in your terminal to install PyTorch on Intel GPU:
```bash
pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/test/xpu
pip install -r requirements.txt
```

#### Step 3: Run ResNet Classification with FP16 AMP

**Run the script**: Execute the script in your terminal:
   ```bash
   python resnet_classification.py
   ```

#### Step 4: Run Faster R-CNN Detection with FP16 AMP

**Run the script**: Execute the script in your terminal:
   ```bash
   python fasterrcnn_detection.py
   ```

#### Step 5: Run ResNet Classification with INT8 PT2E

> [!NOTE]  
> PT2E is actively being developed and optimized. Advanced users are recommended to install PyTorch nightly wheels or build from source to get the latest changes in the stock PyTorch master before using a PT2E example.

1. **Download the Dataset ImageNet**: We recommend calibration with a real dataset. Please Download ImageNet. If you don't want to use a dataset, you can also set dummy data for the following script.

2. **Run the script**: Execute the script in your terminal:
   
   If you have downloaded the ImageNet Dataset:

   ```bash
   python pt2e_resnet.py --data "path_to_imagenet"
   ```

   If you preferred to use Dummy Dataset:
   
   ```bash
   python pt2e_resnet.py --dummy
   ```