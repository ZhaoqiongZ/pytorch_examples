import os

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import Image
import json
import argparse

print(f'Torch Version: {torch.__version__}')

parser = argparse.ArgumentParser(description='PyTorch ImageNet PT2E')
parser.add_argument('--data', metavar='DIR', nargs='?', default='imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--num-iterations', default=20, type=int)
parser.add_argument('-b', '--bs', default=32, type=int, metavar='N')
parser.add_argument('--dummy', action="store_true", help='use dummy data for '
                    'benchmark training or val')

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        pred = pred.cpu()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def pt2e_calib(model, val_loader_calib):
    import torch.ao.quantization.quantizer.xpu_inductor_quantizer as xpuiq
    import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
    from torch.ao.quantization.quantizer.xpu_inductor_quantizer import XPUInductorQuantizer
    from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
    from torch.export import export_for_training
    from torch.ao.quantization.quantize_pt2e import (
        _convert_to_reference_decomposed_fx,
        convert_pt2e,
        prepare_pt2e,
        prepare_qat_pt2e,
    )
    torch._inductor.config.freezing = True
    torch._inductor.config.force_disable_caches = True

    quantizer = XPUInductorQuantizer()
    quantizer.set_global(xpuiq.get_default_xpu_inductor_quantization_config())

    for i, (input, target) in enumerate(val_loader_calib):
        inputs = tuple([input.to("xpu")])
        output =  model(inputs[0])
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        break
    with torch.no_grad():
        export_model = export_for_training(
            model,
            inputs,
            strict=True
        ).module()
        prepare_model = (
            prepare_pt2e(export_model, quantizer)
        )
        for i, (input, target) in enumerate(val_loader_calib):
            calib = input.xpu()
            prepare_output = prepare_model(calib)
            acc1, acc5 = accuracy(prepare_output, target, topk=(1, 5))
            print("In Calibration, acc1 is: ", acc1, " acc5 is:", acc5)
            if i > 2:
                break
        torch.ao.quantization.move_exported_model_to_eval(prepare_model)
        convert_model = convert_pt2e(prepare_model)
        # torch.compile is not supported on Intel Client GPU for PyTorch v2.6, please comment the following line if you are running on Windows and uncomment for better performance if you are running on Linux
        # convert_model = torch.compile(convert_model)
        convert_model(inputs[0])
    print("PT2E compilation finished.")
    return convert_model

def main():
    args = parser.parse_args()
    # Load a pre-trained ResNet18 model
    model = models.resnet18(pretrained=True)
    model.eval()  # Set the model to evaluation mode

    # Move the model to XPU
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    model.to(device)

    # create dataset
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    if args.dummy:
        print("[info] dummy dataset is used")
        val_dataset_size = args.num_iterations * args.bs if (args.dummy and args.num_iterations) else 50000
        val_dataset = datasets.FakeData(val_dataset_size, (3, 224, 224), 1000, transforms.ToTensor())
    else:
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

    # calibration dataloader
    val_loader_calib = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.bs, shuffle=False,
        num_workers=args.workers, pin_memory=True, pin_memory_device="xpu")

    converted_model = pt2e_calib(model, val_loader_calib)

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load an image
    image = Image.open("test_image_classification.jpg")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)  # Move image to XPU

    # Perform inference with quantized model
    with torch.no_grad():
        output = converted_model(image)

    # Get the predicted class
    _, predicted = torch.max(output, 1)
    print("Predicted class index:", predicted.item())

    # Load ImageNet class index
    with open('ImageNet_class_index.json') as f:
        class_idx = json.load(f)

    # Get the class name
    class_name = class_idx[str(predicted.item())][1]
    print("Predicted class name:", class_name)

    # Get top 5 predictions
    topk = 5
    _, indices = torch.topk(output, topk)
    top5_classes = [class_idx[str(idx.item())][1] for idx in indices[0]]
    print("Top 5 predictions:", top5_classes)

if __name__ == '__main__':
    main()