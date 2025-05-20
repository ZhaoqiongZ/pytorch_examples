# Qwen LoRA Fine-tuning with PyTorch on Intel Client GPU

## LoRA

LoRA (Low-Rank Adaptation of Large Language Models) is a popular and lightweight training technique that significantly reduces the number of trainable parameters. It works by inserting a smaller number of new weights into the model and only these are trained.

### Step 1: Install Intel Graphics Driver

#### Windows

1. **Download the driver**: Visit the [Intel Arc & Iris Xe Graphics Driver page](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html).
2. **Install the driver**: Follow the instructions on the page to download and install the driver on your system.

#### Linux

Refer to [Installing Client GPUs](https://dgpu-docs.intel.com/driver/client/overview.html).

### Step 2: Install PyTorch/Intel Extension for PyTorch

1. **Run the installation command**: Copy the command provided and run it in your terminal to install PyTorch on Intel GPU:

```bash
python -m venv env_run
source env_run/bin/activate
pip install --upgrade pip

# For Intel® Core™ Ultra Series 2 with Intel® Arc™ Graphics, use the commands below:
python -m pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```

### Step 3: Install required packages for LoRA Fine-tuning

```bash
pip install -r requirements.txt
```

### Step 4: Huggingface

During the execution, you may need to log in your Hugging Face account to download model files from online mode. Refer to [HuggingFace Login](https://huggingface.co/docs/huggingface_hub/quick-start#login)

```bash
huggingface-cli login --token <your_token_here>
```

### Step 5: Run Qwen/Qwen2-0.5B LoRA Fine-tuning

**Run the script**: Execute the script in your terminal:

```bash
python qwen2_ft.py --model_name_or_path "Qwen/Qwen2-0.5B" --data_path "./dataset.json" --bf16 True --output_dir output_qwen --num_train_epochs 5 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 1 --evaluation_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 10 --learning_rate 3e-4 --weight_decay 0.01 --adam_beta2 0.95 --warmup_ratio 0.01 --lr_scheduler_type "cosine" --logging_steps 1 --report_to "none" --model_max_length 256 --use_lora
```
