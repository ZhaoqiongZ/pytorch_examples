# Examples of Intel Extension for PyTorch* on Intel GPU

## Examples

1. [Classification by ResNet18](./classification_detection/README.md)

- Run inference with pre-trained ResNet18 with FP16 data type on Intel GPU.

- Quantize the model as INT8 by PyTorch 2 Export Post Training Quantization (PT2E) and compare the performance and accuracy with FP16 model.


2. [Object Detection by FasterRCNN-ResNet50](./classification_detection/README.md)

- Run inference with pre-trained FasterRCNN-ResNet50 with FP16 data type on Intel GPU.


3. [Fine-tuning & Object Detection by YOLOv5](./yolo/README.md)

- Run fine-tuning based on pre-trained YOLOv5 model on Intel GPU.

- Run inference on Intel GPU.


4. [LoRA Fine-tuning on Intel GPU](./lora/README.md)

- Run LoRA fine-tuning based on pre-trained Qwen/Qwen2-0.5B model with custom small dataset on Intel GPU.


5. [QLoRA Fine-tuning on Intel GPU](./qlora/README.md)

- Run QLoRA (Quantization and Low-Rank Adapters) fine-tuning based on pre-trained Qwen/Qwen2-0.5B model with custom small dataset on Intel GPU.

  QLoRA is a parameter-efficient fine-tuning technique for large language models (LLMs) that reduces memory usage and computational cost by combining quantization with LoRA (Low-Rank Adaptation).

  It takes less memory usage and compute capability, but affects accuracy a little bit.

  If you care the accuracy, please try with LoRA.
