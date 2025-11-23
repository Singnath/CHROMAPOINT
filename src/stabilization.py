import numpy as np

class RollingLabStabilizer:
    """Match a/b mean & std to a rolling reference to kill flicker."""
    def __init__(self, win=15, momentum=0.05):
        self.win = win
        self.momentum = momentum
        self.hist = []

    def _stats(self, ab):
        a = ab[...,0].reshape(-1); b = ab[...,1].reshape(-1)
        return (np.array([a.mean(), b.mean()]),
                np.array([a.std()+1e-6, b.std()+1e-6]))

    def apply(self, L, ab):
        mu_t, sig_t = self._stats(ab)
        self.hist.append((mu_t, sig_t))
        if len(self.hist) > self.win: self.hist.pop(0)
        mu_ref = np.mean([m for m,_ in self.hist], axis=0)
        sig_ref = np.mean([s for _,s in self.hist], axis=0)

        a = (ab[...,0] - mu_t[0]) * (sig_ref[0]/sig_t[0]) + mu_ref[0]
        b = (ab[...,1] - mu_t[1]) * (sig_ref[1]/sig_t[1]) + mu_ref[1]
        return L, np.stack([a,b], axis=-1).astype(np.float32)

    def reset(self): self.hist = []

#