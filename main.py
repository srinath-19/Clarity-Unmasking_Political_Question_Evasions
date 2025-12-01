import torch
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.current_device())
    print("name:", torch.cuda.get_device_name(0))
