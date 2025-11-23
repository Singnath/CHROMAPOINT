# src/data/vimeo_frames.py
import os, random, cv2
import numpy as np
from torch.utils.data import Dataset
from src.utils.color import rgb_to_lab, L_from_rgb, to_tensor_L, to_tensor_ab

class VimeoSeptupletFrames(Dataset):
    """
    Treat each of the 7-frame sequences as 7 independent images.
    Expected layout:
      data/vimeo_septuplet/
        sequences/<clip>/<seq>/im1.png ... im7.png
        sep_trainlist.txt
        sep_testlist.txt
    """
    def __init__(self, root="data/vimeo_septuplet", split="train",
                 crop=256, random_flip=True,
                 max_sequences=None, debug=False):
        super().__init__()
        self.seq_root = os.path.join(root, "sequences")
        self.list_path = os.path.join(root, f"sep_{split}list.txt")
        with open(self.list_path, "r") as f:
            items = [l.strip() for l in f if l.strip()]

        # DEBUG limiting
        if debug:
            max_env = os.environ.get("VIMEO_DEBUG_N")
            try:
                max_env = int(max_env) if max_env else None
            except ValueError:
                max_env = None
            max_sequences = max_env or max_sequences or 2000

        if max_sequences is not None and max_sequences > 0:
            items = items[:max_sequences]

        self.items = items
        self.crop = crop
        self.random_flip = random_flip

        # expand to (key, frame_index)
        expanded = []
        for key in self.items:
            for i in range(1, 8):
                expanded.append((key, i))
        self.samples = expanded

    def __len__(self): return len(self.samples)

    def _random_crop(self, img):
        h,w = img.shape[:2]
        ch = min(self.crop, h); cw = min(self.crop, w)
        if h==ch and w==cw: return img
        y = random.randint(0, h-ch)
        x = random.randint(0, w-cw)
        return img[y:y+ch, x:x+cw]

    def __getitem__(self, idx):
        key, i = self.samples[idx]
        path = os.path.join(self.seq_root, key, f"im{i}.png")
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if self.crop is not None:
            rgb = self._random_crop(rgb)

        if self.random_flip and random.random() < 0.5:
            rgb = np.ascontiguousarray(np.fliplr(rgb))

        L = L_from_rgb(rgb)       # uint8 HxW
        lab = rgb_to_lab(rgb)     # float HxWx3
        ab  = lab[...,1:]         # float HxWx2

        L_t  = to_tensor_L(L)     # 1xHxW in [-1,1]
        ab_t = to_tensor_ab(ab)   # 2xHxW roughly [-1,1]
        return L_t, ab_t, path
#
