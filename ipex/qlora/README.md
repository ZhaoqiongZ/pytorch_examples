# Qwen QLoRA Fine-tuning with PyTorch & Intel Extension for PyTorch* on Intel Client GPU

## QLoRA

QLoRA is a parameter-efficient fine-tuning technique for large language models (LLMs) that reduces memory usage and computational cost by combining quantization with LoRA (Low-Rank Adaptation).

It takes less memory usage and compute capability, but affects accuracy a little bit.

If you care the accuracy, please try with LoRA.

### Step 1: Install Intel Graphics Driver

#### Windows

1. **Download the driver**: Visit the [Intel Arc & Iris Xe Graphics Driver page](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html).
2. **Install the driver**: Follow the instructions on the page to download and install the driver on your system.

#### Linux

Refer to [Installing Client GPUs](https://dgpu-docs.intel.com/driver/client/overview.html).

### Step 2: Install PyTorch & Intel Extension for PyTorch*

1. **Run the installation command**: Copy the command provided and run it in your terminal to install PyTorch on Intel GPU:

```bash
python -m venv env_run
source env_run/bin/activate
pip install --upgrade pip

python -m pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
python -m pip install intel-extension-for-pytorch==2.7.10+xpu oneccl_bind_pt==2.7.0+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us
```

The above installation is for Windows OS, for Linux platform, please refer to [Intel Extension for PyTorch* Installation](https://pytorch-extension.intel.com/installation?platform=gpu&version=v2.7.10%2Bxpu)

### Step 3: Install required packages for QLoRA Fine-tuning

```bash
pip install -r requirements.txt
```

### Step 4: Huggingface Login

During the execution, you may need to log in your Hugging Face account to download model files from online mode. Refer to [HuggingFace Login](https://huggingface.co/docs/huggingface_hub/quick-start#login)

```bash
huggingface-cli login --token <your_token_here>
```

### Step 5: Run Qwen/Qwen2-1.5B QLora Fine-tuning

**Run the script**: Execute the script in your terminal:

```bash
bash run_qlora_client.sh
```

### Step 6: Run inference with quantized model

There will be a FileNotFoundError for libbitsandbytes_cpu.dll, but we don't need this package. Please ignore this and continue running. We are upstream to PyTorch to fix it in the following release

```bash
# set quant_type and max_new_tokens according to your needs
python bnb_inf_xpu.py --model_name "Qwen/Qwen2-1.5B" --quant_type nf4 --max_new_tokens 64 --device xpu
```
