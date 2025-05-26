# Installation


## Pre-requisite


torchtune requires PyTorch, so please install by following command. You should also install torchvision (for multimodal LLMs) and torchao (for quantization APIs). 

```
pip3 install --pre torch torchvision torchaudio torchao --index-url https://download.pytorch.org/whl/nightly/xpu

pip install -r requirements.txt
```

## Install torchtune


```
pip3 install --pre torchtune --index-url https://download.pytorch.org/whl/nightly/xpu
```

## Tune your LLM model

Follow the end to end flow to tune your LLM.

```
tune download Qwen/Qwen2.5-0.5B-Instruct
```

list the tune configurations

```
tune ls
```

tune the Qwen/Qwen2.5-0.5B-Instruct 

```
tune run lora_finetune_single_device --config qwen2_5/0.5B_lora_single_device
```