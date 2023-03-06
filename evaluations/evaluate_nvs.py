import argparse
import cv2
import os
import numpy as np
import torch
import torch.nn.functional as F
import lpips as lpips_lib
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import pytorch_ssim


def mse2psnr(mse):
    """
    :param mse: scalar
    :return:    scalar np.float32
    """
    mse = np.maximum(mse, 1e-10)  # avoid -inf or nan when mse is very small.
    psnr = -10.0 * np.log10(mse)
    return psnr.astype(np.float32)


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    lpips_vgg_fn = lpips_lib.LPIPS(net='vgg').to(device)

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--seq_txt",
                             default="./evaluations/test_seqs.txt",
                             help="the .txt file listing the testing sequences")
    args_parser.add_argument("--gt_root",
                             default="./data",
                             help="the directory of the dataset")
    args_parser.add_argument("--skip", 
                             nargs="+",
                             help="sequences to skip")
    args_parser.add_argument("--method",
                             required=True,
                             help="name of the method to be evaluated")
    args = args_parser.parse_args()

    skip_seqs = args.skip if args.skip is not None else []
    with open(args.seq_txt, "r") as f:
        seqs = [l for l in f.read().split(",") if l not in skip_seqs]

    root_dir = f"./nvs/{args.method}"
    psnr_out = 0
    ssim_out = 0
    lpips_out = 0
    n_imgs = 0
    for seq in seqs:
        seq_dir = os.path.join(root_dir, seq)
        img_ids = sorted([f.split("_")[1].split(".")[0] for f in os.listdir(seq_dir) if f.startswith("gt") and f.endswith(".png")])
        for img_id in img_ids:
            rendered_img = cv2.imread(os.path.join(seq_dir, f"render_{img_id}.png"), -1)[...,::-1] / 255.
            gt_img = cv2.imread(os.path.join(seq_dir, f"gt_{img_id}.png"), -1)[...,::-1] / 255.
            rendered_img = torch.from_numpy(rendered_img).float().to(device)
            gt_img = torch.from_numpy(gt_img).float().to(device)

            # compute mse
            mse = F.mse_loss(rendered_img, gt_img).item()

            # compute psnr
            psnr = mse2psnr(mse)

            # compute ssim
            ssim = pytorch_ssim.ssim(rendered_img.permute(2, 0, 1).unsqueeze(0), gt_img.permute(2, 0, 1).unsqueeze(0)).item()

            # compute lpips
            lpips_loss = lpips_vgg_fn(rendered_img.permute(2, 0, 1).unsqueeze(0).contiguous(),
                                        gt_img.permute(2, 0, 1).unsqueeze(0).contiguous(), normalize=True).item()
            psnr_out += psnr
            ssim_out += ssim
            lpips_out += lpips_loss
            n_imgs += 1
    print(f"psnr: {psnr_out/n_imgs}")
    print(f"ssim_out: {ssim_out/n_imgs}")
    print(f"lpips: {lpips_out/n_imgs}")


if __name__ == "__main__":
    main()