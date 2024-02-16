import torch

def cuda_availability():
    """
    Check if CUDA is available and print some information about the CUDA devices.
    """
    if torch.cuda.is_available():
        print("CUDA is available")
        num_cuda_devices = torch.cuda.device_count()
        print("Number of CUDA devices:", num_cuda_devices)
        if num_cuda_devices > 0:
            print("CUDA device name:", torch.cuda.get_device_name(0))
            print("CUDA device capability:", torch.cuda.get_device_capability(0))
        else:
            print("No CUDA devices found despite CUDA being available")
    else:
        print("CUDA is not available")

