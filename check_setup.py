import torch
import platform

def check_env():
    print(f"--- Environment Check on {platform.system()} ---")
    print(f"Python Version: {platform.python_version()}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # 1. Check for NVIDIA GPU (CUDA)
    if torch.cuda.is_available():
        print(f"✅ NVIDIA GPU Detected (CUDA)")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        
    # 2. Check for AMD GPU (ROCm)
    elif hasattr(torch.version, 'hip') and torch.version.hip is not None:
        if torch.cuda.is_available():
            print(f"✅ AMD GPU Detected (ROCm)")
            print(f"Device: {torch.cuda.get_device_name(0)}")
        else:
            print("❌ AMD GPU Found but NOT active in PyTorch.")

    # 3. Check for Apple Silicon GPU (MPS) - for our Mac freinds
    elif torch.backends.mps.is_available():
        print("✅ Apple Silicon GPU Detected (MPS)")
    
    else:
        print("⚠️ No GPU detected. Running on CPU.")

if __name__ == "__main__":
    check_env()