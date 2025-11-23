# src/train/validate_image.py
import os, re, glob, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from src.models.unet_color import UNetColor
from src.data.vimeo_frames import VimeoSeptupletFrames
from src.utils.color import denorm_L, from_tensor_ab, compose_Lab, lab_to_rgb
from src.eval.metrics import psnr as psnr_np, ssim as ssim_np, LPIPSWrapper

def ab_to_rgb_batch(L_t, ab_t):
    """Convert batches of L[-1..1] + ab[-1..1] to uint8 RGB (B,H,W,3). CPU numpy."""
    import cv2
    B, _, H, W = L_t.shape
    L_u8_batch = denorm_L(L_t)           # [B,H,W] uint8
    ab_np_batch = from_tensor_ab(ab_t)   # [B,H,W,2] float32 (native scale)
    rgbs = []
    for b in range(B):
        L_u8 = L_u8_batch[b]
        ab_np = ab_np_batch[b]
        if ab_np.shape[0] != H or ab_np.shape[1] != W:
            a = cv2.resize(ab_np[...,0], (W, H), interpolation=cv2.INTER_CUBIC)
            bb= cv2.resize(ab_np[...,1], (W, H), interpolation=cv2.INTER_CUBIC)
            ab_np = np.stack([a, bb], axis=-1)
        lab = compose_Lab(L_u8, ab_np)
        rgb = lab_to_rgb(lab)  # uint8
        rgbs.append(rgb)
    return np.stack(rgbs, axis=0)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/vimeo_septuplet")
    ap.add_argument("--split", default="test")  # validate on test list (held-out)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--ckpt_dir", default="models/checkpoints_image")
    ap.add_argument("--best_out", default="models/best_image.pt")
    ap.add_argument("--log_dir", default="outputs/val_image")
    ap.add_argument("--max_sequences", type=int, default=1000, help="limit val for speed")
    ap.add_argument("--debug", action="store_true", help="smaller, fast validation (first 200 sequences)")
    ap.add_argument("--save_previews", type=int, default=12, help="save N sample predictions")

    ap.add_argument("--use_lpips", action="store_true", help="compute LPIPS if available (slower)")
    return ap.parse_args()

def list_checkpoints(ckpt_dir):
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "unet_ep*.pt")))
    # natural sort by epoch number
    def _key(p):
        m = re.search(r"unet_ep(\d+)\.pt$", p)
        return int(m.group(1)) if m else 1e9
    return sorted(paths, key=_key)

def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    net = UNetColor(in_ch=1, base=64).to(device)
    net.load_state_dict(ck["model"], strict=True)
    net.eval()
    meta = ck.get("args", {})
    return net, meta

def evaluate_one(net, batch, device, want_lpips=False, lpips_model=None):
    L, ab_gt, _ = batch
    L = L.to(device)
    with torch.no_grad():
        ab_pred = net(L)        # [-110,110]
        ab_pred = (ab_pred / 110.0).clamp(-1, 1)
    # build RGB on CPU numpy
    rgb_pred = ab_to_rgb_batch(L.cpu(), ab_pred.cpu())
    rgb_gt   = ab_to_rgb_batch(L.cpu(), ab_gt.cpu())

    # metrics per image
    psnrs, ssims, lp = [], [], []
    for i in range(rgb_pred.shape[0]):
        p = psnr_np(rgb_pred[i], rgb_gt[i])
        s = ssim_np(rgb_pred[i], rgb_gt[i])
        psnrs.append(p); ssims.append(s)
        if want_lpips and lpips_model:
            lp.append(lpips_model(rgb_pred[i], rgb_gt[i]))
    result = {
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
    }
    if want_lpips and lp:
        result["lpips"] = float(np.mean(lp))
    return result, rgb_pred, rgb_gt

def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    # device
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print("[val] Device:", device)

    # dataset
    max_seq = 200 if args.debug else args.max_sequences
    ds = VimeoSeptupletFrames(
        root=args.data_root, split=args.split, crop=args.crop,
        random_flip=False, max_sequences=max_seq, debug=bool(args.debug)
    )
    dl = DataLoader(ds, batch_size=args.bs, shuffle=False,
                    num_workers=args.num_workers, pin_memory=False)

    # LPIPS (optional)
    lpips_model = None
    if args.use_lpips:
        try:
            lpips_model = LPIPSWrapper(net="alex", device=("cuda" if device=="cuda" else "cpu"))
            if not lpips_model:
                print("[val] lpips not available, skipping.")
                lpips_model = None
        except Exception as e:
            print("[val] lpips init failed:", repr(e))
            lpips_model = None

    # evaluate all checkpoints, keep the best (by highest SSIM; if LPIPS available, use lowest LPIPS)
    ckpts = list_checkpoints(args.ckpt_dir)
    if not ckpts:
        raise SystemExit(f"No checkpoints found in {args.ckpt_dir}")
    print("[val] Found checkpoints:", ckpts)

    best = None  # (score_key_value, path, metrics_dict)
    for ck in ckpts:
        print("[val] Evaluating:", ck)
        net, meta = load_model(ck, device)

        agg = {"psnr": [], "ssim": [], "lpips": []}
        save_count = 0

        for batch in tqdm(dl, desc=os.path.basename(ck), dynamic_ncols=True):
            metrics, rgb_pred, rgb_gt = evaluate_one(net, batch, device, want_lpips=bool(lpips_model), lpips_model=lpips_model)
            agg["psnr"].append(metrics["psnr"])
            agg["ssim"].append(metrics["ssim"])
            if "lpips" in metrics: agg["lpips"].append(metrics["lpips"])

            # save a few previews
            if save_count < args.save_previews:
                # build a 3-row grid: L (as 3ch), pred, gt
                L, _, _ = batch
                L3 = (L[:4].repeat(1,3,1,1)+1)/2  # [0,1]
                # torchify rgb arrays for saving
                import torch as _T
                p = _T.from_numpy((rgb_pred[:4].astype(np.float32)/255.0)).permute(0,3,1,2)
                g = _T.from_numpy((rgb_gt[:4].astype(np.float32)/255.0)).permute(0,3,1,2)
                grid = torch.cat([L3, p, g], dim=0)
                save_image(grid, os.path.join(args.log_dir, f"{os.path.basename(ck)}_sample{save_count}.png"), nrow=4)
                save_count += 1

        # aggregate
        mean_psnr = float(np.mean(agg["psnr"])) if agg["psnr"] else 0.0
        mean_ssim = float(np.mean(agg["ssim"])) if agg["ssim"] else 0.0
        mean_lp   = float(np.mean(agg["lpips"])) if agg["lpips"] else None

        summary = {"ckpt": ck, "psnr": mean_psnr, "ssim": mean_ssim}
        if mean_lp is not None: summary["lpips"] = mean_lp
        print("[val] Summary:", summary)

        # choose best: prefer LPIPS if present (lower is better), else SSIM (higher is better)
        if mean_lp is not None:
            key = (-mean_lp, mean_ssim)  # lower lpips, then higher ssim
        else:
            key = (mean_ssim, mean_psnr) # higher ssim, then higher psnr

        if (best is None) or (key > best[0]):
            best = (key, ck, summary)

    # save results json
    results_path = os.path.join(args.log_dir, "val_summary.json")
    with open(results_path, "w") as f:
        json.dump(best[2], f, indent=2)
    print("[val] Best checkpoint:", best[1])
    print("[val] Metrics:", best[2])
    print("[val] Wrote:", results_path)

    # copy best to a canonical path
    import shutil
    shutil.copy2(best[1], args.best_out)
    print("[val] Saved best copy to:", args.best_out)

if __name__ == "__main__":
    main()