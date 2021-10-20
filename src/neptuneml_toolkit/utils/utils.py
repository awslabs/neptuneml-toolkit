import os
import shlex
import subprocess

def num_gpus():
    """Return the number of GPUs available in the current instance.
    Returns:
        int: Number of GPUs available in the current instance.
    """
    try:
        cmd = shlex.split("nvidia-smi --list-gpus")
        output = subprocess.check_output(cmd).decode("utf-8")
        return sum([1 for x in output.split("\n") if x.startswith("GPU ")])
    except (OSError, subprocess.CalledProcessError):
        print("No GPUs detected (normal if no gpus installed)")
        return 0



def get_available_devices():
    """Return the available devices in the current instance
    Returns:
        list[int]: list of available GPU device ids
        [-1] means GPU is not enabled on device and cpu is only available device
    """
    n_gpus = int(os.environ.get('SM_NUM_GPUS', num_gpus()))
    devices = [i for i in range(n_gpus)] if n_gpus > 0 else [-1]
    return devices

def get_device_type(devices):
    """Return the type of available device(s) in the current instance
    Returns:
        str: "cpu" or "gpu"
    """
    return "cpu" if devices[0] == -1 else "cuda"

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d
