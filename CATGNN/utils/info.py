import os
import torch
import psutil
import math

"""
Automatically adjust the number of partitions based on the available system computational resources.

Args:
    dataset (str): Dataset name.
    path (str): Path of dataset directory.
    T (float): Memory needed for GNN computation. 
               By default, T = 2/3 * min(gpu_available_memory).

"""


def auto_number_of_partition(dataset, path, T):
    ## get graph info
    print("Data info")
    
    print("Dataset:", dataset)
    if dataset in ['ppi', 'flickr', 'reddit', 'yelp', 'amazon']:
        path = path + dataset
    elif dataset in ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M']:
        path = path + '_'.join(dataset.split('-')) + '/raw/'
                               
    # print(path)
    total_size = 0
    for file_path, dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(file_path, f)
            file_size = os.path.getsize(fp)
            # print("file name: ", f)
            # print("file size: ", file_size)
            total_size += file_size
    print(f"Toal data size: {total_size / 1024**3:.2f} GB")
    print("==========")

    ## get RAM info
    print("RAM info")
    try:
        ram_info = psutil.virtual_memory()
        print(f"Total RAM: {ram_info.total / 1024**3:.2f} GB")
        print(f"Available RAM: {ram_info.available / 1024**3:.2f} GB")
        print(f"Used RAM: {ram_info.used / 1024**3:.2f} GB")
        print(f"RAM percentage usage: {ram_info.percent}%")
    except FileNotFoundError:
        print("RAM info not available on this system")
    print("==========")

    ## get CPU info
    print("CPU info")
    try:
        cpu_percent = psutil.cpu_percent()
        print(f"CPU percentage usage: {cpu_percent}%")

        # CPU frequency
        cpu_info = psutil.cpu_freq()
        print(f"CPU current frequency: {cpu_info.current:.2f} Mhz")
        print(f"CPU minimum frequency: {cpu_info.min:.2f} Mhz")
        print(f"CPU maximum frequency: {cpu_info.max:.2f} Mhz")

    except FileNotFoundError:
        print("CPU info not available on this system")
    print("==========")

    ## get Disk info
    print("Disk info")
    try:
        disk_info = psutil.disk_usage("/")
        print(f"Total disk size: {disk_info.total / 1024**3:.2f} GB")
        print(f"Used disk size: {disk_info.used / 1024**3:.2f} GB")
        print(f"Free disk size: {disk_info.free / 1024**3:.2f} GB")
    except FileNotFoundError:
        print("Disk info not available on this system")
    print("==========")

    # ## get the GPU info
    print("GPU info")
    # get number of gpus
    n_gpus = torch.cuda.device_count()

    gpu_available_memory = []
    print("Number of GPUs: ", n_gpus)
    for i in range(n_gpus):
        device = torch.device("cuda:%i"%i)
        # get the properties
        device_properties = torch.cuda.get_device_properties(device)
        print("Device_properties: ", device_properties)

        # calculate the amount of available GPU memory
        available_memory = device_properties.total_memory - torch.cuda.max_memory_allocated()

        # print the result
        print(f"Available GPU memory: {available_memory / 1024**3:.2f} GB")
        gpu_available_memory.append(available_memory)
    #     # GraphSAGE model size
    #     model_size = 180000
    #
    #     print(available_memory - model_size)
    print("==========")

    
    a = 1.5
    # T = 2/3
    if T == None:
        T = 2/3 * min(gpu_available_memory)
    # total_size = 174*1024**3
    # print(f"Toal data size: {total_size / 1024**3:.2f} GB")
    # if a * total_size / n_gpus < min(gpu_available_memory):
    # if a * total_size / n_gpus < (1-T) * min(gpu_available_memory):
    if a * total_size / n_gpus < (min(gpu_available_memory) - T):
        number_of_partitions = n_gpus
        # print("Number of partitions: ", n_gpus)
    else:
        # number_of_partitions = n_gpus * math.ceil(a * total_size / n_gpus / min(gpu_available_memory))
        number_of_partitions = 2 ** math.ceil(math.log2(-22*(min(gpu_available_memory)/(total_size/(1-T/min(gpu_available_memory))))+22))
        # print("Number of partitions: ", n_gpus * math.ceil(a * total_size / n_gpus / min(gpu_available_memory)))
    print("Number of partitions: ", number_of_partitions)

    return number_of_partitions
