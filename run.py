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
DISPARITY_DIR = OUTPUT / "disparity"
ENTROPY_DIR = OUTPUT / "entropy"

SCALE = 0.75  # 1.0 does not fit into Nvidia GPU memory
MAX_DISP = 768

# Create output directories
os.makedirs(DISPARITY_DIR, exist_ok=True)

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

    # resize
    img1_resized = cv2.resize(
        img1r,
        None,
        fx=SCALE,
        fy=SCALE,
        interpolation=cv2.INTER_CUBIC,
    )
    img2_resized = cv2.resize(
        img2r,
        None,
        fx=SCALE,
        fy=SCALE,
        interpolation=cv2.INTER_CUBIC,
    )
    img1_processed = processed(img1_resized).numpy()
    img2_processed = processed(img2_resized).numpy()

    img1_reshaped = np.reshape(img1_processed, [1, 3, img1_processed.shape[1], img1_processed.shape[2]])
    img2_reshaped = np.reshape(img2_processed, [1, 3, img2_processed.shape[1], img2_processed.shape[2]])

    # Padding
    max_h = int(img1_reshaped.shape[2] // 64 * 64)
    max_w = int(img1_reshaped.shape[3] // 64 * 64)
    if max_h < img1_reshaped.shape[2]:
        max_h += 64
    if max_w < img1_reshaped.shape[3]:
        max_w += 64

    top_pad = max_h - img1_reshaped.shape[2]
    left_pad = max_w - img1_reshaped.shape[3]
    img1_padded = np.lib.pad(
        img1_reshaped,
        ((0, 0), (0, 0), (top_pad, 0), (0, left_pad)),
        mode="constant",
        constant_values=0,
    )
    img2_padded = np.lib.pad(
        img2_reshaped,
        ((0, 0), (0, 0), (top_pad, 0), (0, left_pad)),
        mode="constant",
        constant_values=0,
    )

    # Evaluating the model
    if is_cuda_available:
        img1_torch = Variable(torch.FloatTensor(img1_padded).cuda())
        img2_torch = Variable(torch.FloatTensor(img2_padded).cuda())
    else:
        img1_torch = Variable(torch.FloatTensor(img1_padded))
        img2_torch = Variable(torch.FloatTensor(img2_padded))

    with torch.no_grad():

        if is_cuda_available:
            torch.cuda.synchronize()

        start_time = time.time()
        pred_disp, entropy = model(img1_torch, img2_torch)

        if is_cuda_available:
            torch.cuda.synchronize()

        lapsed_time = time.time() - start_time
        print(f"Lapsed time: {lapsed_time:.2f}")

    pred_disp = torch.squeeze(pred_disp).data.cpu().numpy()

    top_pad = max_h - img1_resized.shape[0]
    left_pad = max_w - img1_resized.shape[1]
    pred_disp = pred_disp[top_pad:, : pred_disp.shape[1] - left_pad]

    # save predictions
    # resize to highres
    pred_disp = cv2.resize(
        pred_disp / SCALE,
        (img_size[1], img_size[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    # clip while keep inf
    invalid = np.logical_or(pred_disp == np.inf, pred_disp != pred_disp)
    pred_disp[invalid] = np.inf
    disparity_u8 = (255 * pred_disp / MAX_DISP).astype(np.uint8)

    path_to_disp = DISPARITY_DIR / filename

    cv2.imwrite(path_to_disp, disparity_u8)

    if is_cuda_available:
        torch.cuda.empty_cache()
