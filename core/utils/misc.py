import os

import torch
import torch.distributed as dist


def print_once(
    *values: object,
    sep: str | None = " ",
    end: str | None = "\n",
) -> None:
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*values, sep=sep, end=end)
    else:
        print(*values, sep=sep, end=end)
        
def check_file_path_unused(file_path):
    # Check if the file path is None
    if file_path is None:
        print("Error: File path is empty.")
        exit()

    # Check if the file already exists
    if os.path.exists(file_path):
        print(f"Error: File {file_path} already exists. Please choose a different file path.")
        exit() 

