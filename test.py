import argparse
import yaml

# Đường dẫn đến tệp YAML
yaml_file = "/home/bigdata/Documents/TND_Modeling/config.yaml"

# Đọc tệp YAML
with open(yaml_file, "r") as file:
    yaml_data = yaml.safe_load(file)

# Chuyển đổi dữ liệu YAML thành đối tượng namespace
args = argparse.Namespace(**yaml_data)

# Truy cập các thuộc tính như args.root
print(args.root)
print(args.epochs)
print(args.batch_size)
# và các thuộc tính khác