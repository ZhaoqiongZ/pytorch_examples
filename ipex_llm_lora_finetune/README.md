# Classification and detection with PyTorch on Intel Client GPU (Windows Platform)

### Step 1: Install Intel Graphics Driver

1. **Download the driver**: Visit the [Intel Arc & Iris Xe Graphics Driver page](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html).
2. **Install the driver**: Follow the instructions on the page to download and install the driver on your system.

### Step 2: Install PyTorch/Intel Extension for PyTorch

1. **Run the installation command**: Copy the command provided and run it in your terminal to install PyTorch on Intel GPU:

```bash
# For Intel® Core™ Ultra Series 2 with Intel® Arc™ Graphics, use the commands below:
conda install libuv
python -m pip install torch==2.5.1+cxx11.abi torchvision==0.20.1+cxx11.abi torchaudio==2.5.1+cxx11.abi intel-extension-for-pytorch==2.5.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/lnl/us/
```

The above installation is for Intel® Core™ Ultra Series 2 with Intel® Arc™ Graphics(Lunar Lake) with Windows OS, for other Intel GPU hardware, please refer to [IPEX Installation](https://pytorch-extension.intel.com/installation?platform=gpu&version=v2.5.10%2Bxpu)

### Step 3: Install required packages for lora-finetuning

```bash
pip install -r requirements.txt
```

### Step 4: Run Qwen/Qwen2-0.5B Lora Finetuning

**Run the script**: Execute the script in your terminal:

```bash
python qwen2_ft.py --model_name_or_path "Qwen/Qwen2-0.5B" --data_path "./dataset.json" --bf16 True --output_dir output_qwen --num_train_epochs 5 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 1 --evaluation_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 10 --learning_rate 3e-4 --weight_decay 0.01 --adam_beta2 0.95 --warmup_ratio 0.01 --lr_scheduler_type "cosine" --logging_steps 1 --report_to "none" --model_max_length 256 --use_lora
```
