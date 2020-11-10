import argparse
import cv2
from models import hsm
import numpy as np
import os
import skimage.io
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import time
from models.submodule import *
from utils.eval import mkdir_p, save_pfm
from utils.preprocess import get_transform

from pathlib import Path

# Configuration
ROOT_DIR = Path()
WEIGHTS_PATH = ROOT_DIR / "weights" / "final_768px.tar"
PATH_TO_IMAGES = ROOT_DIR / "data"
LEFT_RECTIFIED = PATH_TO_IMAGES / "left_rect"
RIGHT_RECTIFIED = PATH_TO_IMAGES / "right_rect"
OUTPUT = ROOT_DIR / "output"

scale = 0.75 # 1.0 does not fit into Nvidia GPU memory

# cudnn.benchmark = True
cudnn.benchmark = False

test_left_img, test_right_img, _, _ = DA.dataloader(args.datapath)

# construct model
model = hsm(128, -1, level=1)
model = nn.DataParallel(model, device_ids=[0])

# Check if CUDA available
is_cuda_available = torch.cuda.is_available()
if is_cuda_available:
    model.cuda()

# Load model
if is_cuda_available:
    pretrained_dict = torch.load(WEIGHTS_PATH)
else:
    pretrained_dict = torch.load(WEIGHTS_PATH, map_location=torch.device("cpu"))

pretrained_dict["state_dict"] = {
    k: v for k, v in pretrained_dict["state_dict"].items() if "disp" not in k
}
model.load_state_dict(pretrained_dict["state_dict"], strict=False)


def main():

    processed = get_transform()
    model.eval()

    if is_cuda_available:
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
