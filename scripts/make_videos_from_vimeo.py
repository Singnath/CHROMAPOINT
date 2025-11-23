import argparse, os, cv2
from tqdm import tqdm

def make_video(seq_dir, out_path, fps=14):
    # seq_dir: .../sequences/<clip>/<seq>/ with im1.png ... im7.png
    frames = []
    for i in range(1, 8):
        p = os.path.join(seq_dir, f"im{i}.png")
        if not os.path.isfile(p):
            return False
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            return False
        frames.append(bgr)  # keep BGR for VideoWriter
    if not frames:
        return False

    h, w = frames[0].shape[:2]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    return True

def to_gray(in_path, out_path):
    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 14
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok: break
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        frames.append(gray3)
    cap.release()
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    return True

def main(args):
    seq_root = args.sequences
    out_gt = args.out_gt
    out_gray = args.out_gray
    os.makedirs(out_gt, exist_ok=True)
    os.makedirs(out_gray, exist_ok=True)

    # load official test list (e.g., lines like "00001/0001")
    with open(args.testlist, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]

    picked = lines[:args.num]  # small subset to start
    for key in tqdm(picked, desc="Building videos"):
        seq_dir = os.path.join(seq_root, key)
        # Create a filename like 00001_0001.mp4
        base = key.replace('/', '_') + ".mp4"
        color_path = os.path.join(out_gt, base)
        gray_path  = os.path.join(out_gray, base)

        ok = make_video(seq_dir, color_path, fps=args.fps)
        if ok:
            to_gray(color_path, gray_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", required=True, help="path to .../vimeo_septuplet/sequences")
    ap.add_argument("--testlist", required=True, help="path to .../vimeo_septuplet/sep_testlist.txt")
    ap.add_argument("--out_gt", default="data/gt_videos")
    ap.add_argument("--out_gray", default="data/gray_videos")
    ap.add_argument("--num", type=int, default=50, help="how many clips to convert")
    ap.add_argument("--fps", type=int, default=14)
    args = ap.parse_args()
    main(args)