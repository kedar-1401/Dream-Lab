import torch

if torch.cuda.is_available():
    device = torch.cuda.current_device()
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"CUDA Cores: {torch.cuda.get_device_properties(device).multi_processor_count * 128}")
    print(f"Compute Capability: {torch.cuda.get_device_properties(device).major}.{torch.cuda.get_device_properties(device).minor}")
else:
    print("CUDA is not available.")
