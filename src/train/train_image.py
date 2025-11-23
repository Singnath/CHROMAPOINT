# src/train/train_image.py
import os, argparse
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from src.models.unet_color import UNetColor
from src.data.vimeo_frames import VimeoSeptupletFrames
from src.losses.base_losses import l1_ab, tv_smooth, sat_penalty
from src.losses.perceptual import VGGPerceptual
from src.utils.color import denorm_L, from_tensor_ab, compose_Lab, lab_to_rgb

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/vimeo_septuplet")
    ap.add_argument("--split", default="train")
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--save_dir", default="models/checkpoints_image")
    ap.add_argument("--log_dir", default="outputs/debug_image")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lambda_perc", type=float, default=0.2)
    ap.add_argument("--lambda_tv", type=float, default=0.0)
    ap.add_argument("--lambda_sat", type=float, default=0.0)
    ap.add_argument("--debug", action="store_true", help="Enable debug mode (limit sequences)")
    ap.add_argument("--max_sequences", type=int, default=0, help="Use first N sequences (0=all)")
    return ap.parse_args()

def to_device(batch, device):
    L, ab, _ = batch
    return L.to(device), ab.to(device)

def ab_to_rgb_grid(L_t, ab_t):
    """
    L_t:  [B,1,H,W] in [-1,1]
    ab_t: [B,2,H,W] in [-1,1] approx (scaled by /110 earlier)
    Returns: torch.Tensor [B,3,H,W] in [0,1]
    """
    import torch as _torch
    import numpy as _np
    import cv2
    from src.utils.color import denorm_L, from_tensor_ab, compose_Lab, lab_to_rgb

    B, _, H, W = L_t.shape
    L_u8_batch = denorm_L(L_t)          # [B,H,W] uint8
    ab_np_batch = from_tensor_ab(ab_t)  # [B,H',W',2] float32 (native scale)

    rgbs = []
    for b in range(B):
        L_u8 = L_u8_batch[b]                  # [H,W]
        ab_np = ab_np_batch[b]                # [H',W',2]
        if ab_np.shape[0] != H or ab_np.shape[1] != W:
            # resize each channel to (W,H)
            a = cv2.resize(ab_np[...,0], (W, H), interpolation=cv2.INTER_CUBIC)
            bch= cv2.resize(ab_np[...,1], (W, H), interpolation=cv2.INTER_CUBIC)
            ab_np = _np.stack([a, bch], axis=-1)

        lab = compose_Lab(L_u8, ab_np)        # [H,W,3]
        rgb = lab_to_rgb(lab)                 # uint8 [H,W,3]
        rgbs.append(_torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0)

    return _torch.stack(rgbs, dim=0)

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir,  exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)

    ds = VimeoSeptupletFrames(
        root=args.data_root,
        split=args.split,
        crop=args.crop,
        random_flip=True,
        max_sequences=(args.max_sequences if args.max_sequences > 0 else None),
        debug=args.debug
    )
    dl = DataLoader(
        ds, batch_size=args.bs, shuffle=True,
        num_workers=args.num_workers, pin_memory=False  # pin_memory False for Apple MPS
    )

    net = UNetColor(in_ch=1, base=64).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.5, 0.999))
    perc = VGGPerceptual().to(device).eval()

    steps_per_epoch = max(1, len(ds) // max(1, args.bs))
    global_step = 0

    for epoch in range(1, args.epochs+1):
        net.train()
        pbar = tqdm(dl, total=steps_per_epoch, desc=f"ep{epoch}", leave=False)
        for it, (L, ab_gt, _) in enumerate(pbar):
            L = L.to(device)         # [-1,1]
            ab_gt = ab_gt.to(device) # roughly [-1,1]

            ab_pred = net(L)         # native scale [-110,110]
            ab_pred_norm = ab_pred / 110.0

            with torch.no_grad():
                rgb_tgt = ab_to_rgb_grid(L, ab_gt).to(device)
            rgb_pred = ab_to_rgb_grid(L, ab_pred_norm).to(device)

            loss_l1  = l1_ab(ab_pred_norm, ab_gt)
            loss_perc= perc(rgb_pred, rgb_tgt) * args.lambda_perc
            loss_tv  = tv_smooth(ab_pred_norm, weight=args.lambda_tv)
            loss_sat = sat_penalty(ab_pred, weight=args.lambda_sat)
            loss = loss_l1 + loss_perc + loss_tv + loss_sat

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if (it % 50) == 0:
                pbar.set_postfix({
                    "l1": f"{loss_l1.item():.4f}",
                    "perc": f"{(loss_perc.item() if torch.is_tensor(loss_perc) else loss_perc):.4f}",
                    "total": f"{loss.item():.4f}"
                })
                with torch.no_grad():
                    grid = torch.cat([
                        (L[:4].repeat(1,3,1,1) + 1)/2,   # grayscale preview in [0,1]
                        rgb_pred[:4],
                        rgb_tgt[:4],
                    ], dim=0)
                    save_image(grid, os.path.join(args.log_dir, f"ep{epoch}_step{global_step}.png"), nrow=4)

            global_step += 1
            if it+1 >= steps_per_epoch:  # hard cap tqdm loop for stable ETA
                break

        ckpt_path = os.path.join(args.save_dir, f"unet_ep{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model": net.state_dict(),
            "opt": opt.state_dict(),
            "args": vars(args),
        }, ckpt_path)
        print("Saved:", ckpt_path)

if __name__ == "__main__":
    main()