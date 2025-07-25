import torch
print(torch.cuda.is_available())   # Harus True
print(torch.cuda.get_device_name(0))  # Harus "NVIDIA GeForce RTX 2050"
