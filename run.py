import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import skimage.io
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable

from models import hsm
from models.submodule import *
from utils.eval import mkdir_p, save_pfm
from utils.preprocess import get_transform

# Configuration
ROOT_DIR = Path()
WEIGHTS_PATH = ROOT_DIR / "weights" / "final_768px.tar"
PATH_TO_IMAGES = ROOT_DIR / "data"
LEFT_RECTIFIED = PATH_TO_IMAGES / "left_rect"
RIGHT_RECTIFIED = PATH_TO_IMAGES / "right_rect"
OUTPUT = ROOT_DIR / "output"

SCALE = 0.75  # 1.0 does not fit into Nvidia GPU memory
MAX_DISP = 768

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

    all_files = sorted(os.listdir(LEFT_RECTIFIED))

    for idx, filename in enumerate(all_files):

        path_to_left_img = LEFT_RECTIFIED / filename
        path_to_right_img = RIGHT_RECTIFIED / filename
        img1r = cv2.imread(str(path_to_left_img), 0)
        img2r = cv2.imread(str(path_to_right_img), 0)

        img1r = cv2.cvtColor(img1r, cv2.BGR2RGB).astype("float32")
        img2r = cv2.cvtColor(img2r, cv2.BGR2RGB).astype("float32")
        img_size = img1r.shape[:2]

        # change max disp
        tmpdisp = int(MAX_DISP * SCALE // 64 * 64)
        if (MAX_DISP * SCALE / 64 * 64) > tmpdisp:
            model.module.maxdisp = tmpdisp + 64
        else:
            model.module.maxdisp = tmpdisp
        if model.module.maxdisp == 64:
            model.module.maxdisp = 128

        if is_cuda_available:
            model.module.disp_reg8 = disparityregression(
                model.module.maxdisp, 16
            ).cuda()
            model.module.disp_reg16 = disparityregression(
                model.module.maxdisp, 16
            ).cuda()
            model.module.disp_reg32 = disparityregression(
                model.module.maxdisp, 32
            ).cuda()
            model.module.disp_reg64 = disparityregression(
                model.module.maxdisp, 64
            ).cuda()
        else:
            model.module.disp_reg8 = disparityregression(model.module.maxdisp, 16)
            model.module.disp_reg16 = disparityregression(model.module.maxdisp, 16)
            model.module.disp_reg32 = disparityregression(model.module.maxdisp, 32)
            model.module.disp_reg64 = disparityregression(model.module.maxdisp, 64)

        print(f"Maximum disparity: {model.module.maxdisp}")


    if is_cuda_available:
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
