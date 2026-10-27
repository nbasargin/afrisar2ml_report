import os

import psutil
import torch


def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def is_cuda_available():
    return torch.cuda.is_available()


def print_ram_usage():
    ram_usage_byte = psutil.Process(os.getpid()).memory_info()[0]
    print(f"RAM usage:   {ram_usage_byte / (2**30):6.2f} GB")


def print_cuda_memory_usage():
    device = get_device()
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    msg1 = f"CUDA usage:  {allocated / (2**30):6.2f} GB allocated    {reserved / (2**30):6.2f} GB reserved"
    max_alloc = torch.cuda.max_memory_allocated(device)
    max_res = torch.cuda.max_memory_reserved(device)
    msg2 = f"    {max_alloc / (2**30):6.2f} GB max allocated    {max_res / (2**30):6.2f} GB max reserved"
    print(msg1 + msg2)
