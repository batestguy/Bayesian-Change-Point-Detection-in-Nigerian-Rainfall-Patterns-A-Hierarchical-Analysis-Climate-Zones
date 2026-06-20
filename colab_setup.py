"""Quick setup: create directories and verify GPU on Colab."""
import os
os.makedirs("/content/data/traces", exist_ok=True)
os.makedirs("/content/figures", exist_ok=True)
print("Directories created:")
for d in ["/content/data", "/content/data/traces", "/content/figures"]:
    print(f"  {d}")

import torch
if torch.cuda.is_available():
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("\nNo GPU detected")
