import torch

def get_device():
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print("device_count: {}".format(device_count))
        for device_num in range(device_count):
            print("device {} capability {}".format(
                device_num,
                torch.cuda.get_device_capability(device_num)))
            print("device {} name {}".format(
                device_num, torch.cuda.get_device_name(device_num)))
    else:
        print("no cuda device")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")