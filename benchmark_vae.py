import torch
from sevae.inference.scripts.VAENetworkInterface import VAENetworkInterface
import os
import time

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    vae = VAENetworkInterface(device="cuda:0")
    imgs = torch.randn((1024, 270, 480), device="cuda:0")

    fwp_times = []
    for _ in range(2000):
        start = time.time()
        _ = vae.forward_torch(imgs)
        end = time.time()
        print(end - start)
        fwp_times.append(end - start)
