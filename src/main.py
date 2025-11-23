# src/main.py
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def step2_train_image(
    data_root="vimeo_septuplet",
    split="train",
    crop=256,
    bs=12,
    epochs=2,
    lr=2e-4,
    save_dir="models/checkpoints_image",
    log_dir="outputs/debug_image",
    num_workers=4,
    lambda_perc=0.2,
    lambda_tv=0.0,
    lambda_sat=0.0,
):
    from src.train.train_image import main as train_image_main

    debug_on = os.environ.get("DEBUG", "0").strip().lower() in ("1", "true", "yes", "on")
    max_seqs = os.environ.get("VIMEO_DEBUG_N")

    argv = [
        "train_image.py",
        "--data_root", data_root,
        "--split", split,
        "--crop", str(crop),
        "--bs", str(bs),
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--save_dir", save_dir,
        "--log_dir", log_dir,
        "--num_workers", str(num_workers),
        "--lambda_perc", str(lambda_perc),
        "--lambda_tv", str(lambda_tv),
        "--lambda_sat", str(lambda_sat),
    ]
    if debug_on:
        argv += ["--debug"]
        if max_seqs and max_seqs.isdigit():
            argv += ["--max_sequences", max_seqs]

    prev = sys.argv[:]
    try:
        sys.argv = argv
        print("[main] Launching Step 2: image training")
        print("[main] Args:", " ".join(argv[1:]))
        train_image_main()
    finally:
        sys.argv = prev

def step3_validate(
    data_root="vimeo_septuplet",
    split="test",
    crop=256,
    bs=8,
    num_workers=4,
    ckpt_dir="models/checkpoints_image",
    best_out="models/best_image.pt",
    log_dir="outputs/val_image",
    max_sequences=1000,
    debug=False,
    use_lpips=False,
):
    from src.train.validate_image import main as validate_main
    argv = [
        "validate_image.py",
        "--data_root", data_root,
        "--split", split,
        "--crop", str(crop),
        "--bs", str(bs),
        "--num_workers", str(num_workers),
        "--ckpt_dir", ckpt_dir,
        "--best_out", best_out,
        "--log_dir", log_dir,
        "--max_sequences", str(max_sequences),
    ]
    if debug:
        argv += ["--debug"]
    if use_lpips:
        argv += ["--use_lpips"]

    prev = sys.argv[:]
    try:
        sys.argv = argv
        print("[main] Launching Step 3: validation + metrics")
        print("[main] Args:", " ".join(argv[1:]))
        validate_main()
    finally:
        sys.argv = prev

def main():
    step = os.environ.get("CHROMAPOINT_STEP", "2").strip()
    if step == "2":
        step2_train_image()
    elif step == "3":
        # toggle fast validation with DEBUG=1, and choose LPIPS if you installed it
        debug_on = os.environ.get("DEBUG", "0").strip().lower() in ("1", "true", "yes", "on")
        use_lpips = os.environ.get("USE_LPIPS", "0").strip().lower() in ("1","true","yes","on")
        step3_validate(debug=debug_on, use_lpips=use_lpips)
    else:
        print(f"[main] Unknown or unsupported step: {step}")
        print("Set CHROMAPOINT_STEP=2 for training, CHROMAPOINT_STEP=3 for validation.")

if __name__ == "__main__":
    main()


#