import torch
import torchvision
import time
import argparse
import yaml
import importlib
import random
import numpy as np



seed_value = 42
random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
np.random.seed(seed_value)


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


    # Chọn mô hình
    model_mapping = {
        "Attentionunet": "Models.Attentionunet",
        "Doubleunet": "Models.Doubleunet",
        "Fcn": "Models.Fcn",
        "Unext": "Models.Unext",
        "Unet": "Models.Unet",
        "Vapenet": "Models.Vapenet",
    }

    model_module = importlib.import_module(model_mapping[args.model_name["name"]])
    model = getattr(model_module, args.model_name["version"])()
    model.eval()

    gpu_inference_latency = measure_inference_latency(model=model, device="cuda", input_size=(1,3,256,256), num_samples=100)
    cpu_inference_latency = measure_inference_latency(model=model, device="cpu", input_size=(1,3,256,256), num_samples=100)
    
    print("CPU Inference Latency: {:.2f} ms / sample".format(cpu_inference_latency * 1000))
    print("CUDA Inference Latency: {:.2f} ms / sample".format(gpu_inference_latency * 1000))


# # Tạo dữ liệu ngẫu nhiên để đưa qua mô hình
# input_size = (1, 3, 256, 256)
# input_data = torch.randn(*input_size).to(device)



# # Đo FPS
# num_frames = 200  # Số lượng khung hình để đo FPS
# total_time = 0.0

# with torch.no_grad():
#     for i in range(num_frames):
#         start_time = time.time()   # Ghi lại thời điểm bắt đầu

#         # Chạy mô hình trên GPU
#         output = model(input_data)

#         end_time = time.time()  # Ghi lại thời điểm kết thúc
        
#         # Tính thời gian và cộng dồn để tính tổng thời gian thực hiện
#         elapsed_time = end_time - start_time
#         total_time += elapsed_time

# # Tính FPS
# average_time = total_time / num_frames
# fps = 1.0 / average_time  # Chuyển từ milliseconds sang seconds

# print("Average FPS: {:.2f}".format(fps))

# sum = 0
# for par in model.parameters():
#     sum += par.numel()
# print("Number of parameters: {:.2f}M".format(sum/(1e+6)))