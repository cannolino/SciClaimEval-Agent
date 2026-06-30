import torch

if torch.cuda.is_available():
   print("CUDA is available!")
   num_gpus = torch.cuda.device_count()
   print(f"Number of GPUs: {num_gpus}")
   for i in range(num_gpus):
       props = torch.cuda.get_device_properties(i)
       print(f"GPU {i}: {props.name}")
       print(f" - Total Memory: {props.total_memory / (1024**3):.2f} GB")
       print(f" - Compute Capability: {props.major}.{props.minor}")
else:
   print("CUDA is not available.")