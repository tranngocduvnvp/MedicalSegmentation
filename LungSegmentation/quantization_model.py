import torch
import torch.nn as nn
import time
import argparse
import yaml
from eval import validation, test
import random
import numpy as np
import os
import copy
from build import build
import warnings
warnings.filterwarnings('ignore')

def calibrate_model(model, loader, device=torch.device("cpu:0")):

    model.to(device)
    model.eval()

    for (inputs, labels, _) in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        _ = model(inputs)

def measure_inference_latency(model,
                              device,
                              input_size=(1, 3, 256, 256),
                              num_samples=100,
                              num_warmups=10):

    model.to(device)
    model.eval()

    x = torch.rand(size=input_size).to(device)

    with torch.no_grad():
        for _ in range(num_warmups):
            _ = model(x)
    torch.cuda.synchronize()

    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_samples):
            _ = model(x)
            torch.cuda.synchronize()
        end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_ave = elapsed_time / num_samples

    return elapsed_time_ave

def save_model(model, model_dir, model_filename):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)

def load_model(model, model_filepath, device):

    model.load_state_dict(torch.load(model_filepath, map_location=device))

    return model

def save_torchscript_model(model, model_dir, model_filename):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.jit.save(torch.jit.script(model), model_filepath)

def load_torchscript_model(model_filepath, device):

    model = torch.jit.load(model_filepath, map_location=device)

    return model

def model_equivalence(model_1, model_2, device, rtol=1e-03, atol=1e-04, num_tests=100, input_size=(1,3,256,256)):

    model_1.to(device)
    model_2.to(device)

    for _ in range(num_tests):
        x = torch.rand(size=input_size).to(device)
        y1 = model_1(x).detach().cpu().numpy()
        y2 = model_2(x).detach().cpu().numpy()
        if np.allclose(a=y1, b=y2, rtol=rtol, atol=atol, equal_nan=False) == False:
            print("Model equivalence test sample failed: ")
            print(y1)
            print(y2)
            print(_)
            return False

    return True

class QuantizedModel(nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedModel, self).__init__()
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = model_fp32

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.model_fp32(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x
    

if __name__ == "__main__":
    # Kiểm tra xem GPU có khả dụng không
    device = "cpu"
    print("Device:", device)

    # Đường dẫn đến tệp YAML
    yaml_file = "/home/bigdata/Documents/TND_Modeling/config.yaml"

    # Đọc tệp YAML
    with open(yaml_file, "r") as file:
        yaml_data = yaml.safe_load(file)

    # Chuyển đổi dữ liệu YAML thành đối tượng namespace
    args = argparse.Namespace(**yaml_data)

 

    #1. load model
    ( _, train_dataloader, val_dataloader, test_dataloader,
     perf, model_fp32, optimizer, checkpoint, scheduler, loss_fun) = build(args)

    #2. Load model to cpu
    model_fp32.to(device)
    model_fp32.eval()
    
    #3. Tao model fuse
    fused_model = copy.deepcopy(model_fp32)
    fused_model.eval()

    #4. Fuse model
    modules_to_fuse = [["encoder1", "ebn1"], ["encoder2", "ebn2"], ["encoder3", "ebn3"], ["encoder4", "ebn4"], ["encoder5", "ebn5"],
                        ["decoder1", "dbn1"], ["decoder2", "dbn2"], ["decoder3", "dbn3"], ["decoder4", "dbn4"]]
        
    fused_model = torch.quantization.fuse_modules(fused_model, modules_to_fuse, inplace=True)
    # print(model_fp32)
    # # Print fused model.
    # print(fused_model)

    #5. Check fusion
    assert model_equivalence(model_1=model_fp32, model_2=fused_model, device="cpu", rtol=1e-03, atol=1e-04, num_tests=100, input_size=(1,3,256,256))

    #6. Prepare model for quantization
    quantized_model = QuantizedModel(fused_model)
    quantization_config = torch.quantization.get_default_qconfig("fbgemm")
    quantized_model.qconfig = quantization_config
    print(quantized_model.qconfig)
    torch.quantization.prepare(quantized_model, inplace=True)


    #7. Use training data for calibration.
    calibrate_model(model=quantized_model, loader=train_dataloader, device="cpu")

    #8. Convert quantized model.
    quantized_model = torch.quantization.convert(quantized_model, inplace=True)
    quantized_model.eval()
    
    
    # print(quantized_model)
    #8. Save quantized model.
    save_torchscript_model(model=quantized_model, model_dir="./Trained models", model_filename=args.model_name["version"] + "_quantized" + ".pt")

    # #9. Load quantized model with torchscript.
    quantized_jit_model = load_torchscript_model(model_filepath="./Trained models/" + args.model_name["version"] + "_quantized" + ".pt", device=device)

    fp32_eval_accuracy, _ = test(
                model_fp32, device, test_dataloader, 300, perf,"Test"
            )
    fp32_eval_accuracy, _ = test(
                quantized_jit_model, device, test_dataloader, 300, perf,"Test"
            )

    fp32_cpu_inference_latency = measure_inference_latency(model=model_fp32, device="cpu", input_size=(1,3,256,256), num_samples=100)
    int8_cpu_inference_latency = measure_inference_latency(model=quantized_model, device="cpu", input_size=(1,3,256,256), num_samples=100)
    int8_jit_cpu_inference_latency = measure_inference_latency(model=quantized_jit_model, device="cpu", input_size=(1,3,256,256), num_samples=100)
    fp32_gpu_inference_latency = measure_inference_latency(model=model_fp32, device="cuda", input_size=(1,3,256,256), num_samples=100)
    
    print("FP32 CPU Inference Latency: {:.2f} ms / sample".format(fp32_cpu_inference_latency * 1000))
    print("FP32 CUDA Inference Latency: {:.2f} ms / sample".format(fp32_gpu_inference_latency * 1000))
    print("INT8 CPU Inference Latency: {:.2f} ms / sample".format(int8_cpu_inference_latency * 1000))
    print("INT8 JIT CPU Inference Latency: {:.2f} ms / sample".format(int8_jit_cpu_inference_latency * 1000))
