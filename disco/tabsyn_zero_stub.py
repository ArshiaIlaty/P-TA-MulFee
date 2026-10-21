"""Minimal stub of Gorishniy's `zero` utility library (PyPI `libzero`), which
TabSyn's src/ imports transitively. The TabSyn VAE/diffusion/sample code paths
we use do not call these functions, but the import must resolve. Installing the
real libzero would downgrade torch to 1.13 and break the rest of the env, so we
provide the small API surface actually referenced (random, iter_batches,
hardware, improve_reproducibility)."""
import numpy as np
import torch


def iter_batches(data, size):
    n = len(data[0]) if isinstance(data, (tuple, list)) else len(data)
    for i in range(0, n, size):
        if isinstance(data, (tuple, list)):
            yield tuple(d[i:i + size] for d in data)
        else:
            yield data[i:i + size]


class _Random:
    def get_state(self):
        return {"torch": torch.random.get_rng_state(), "numpy": np.random.get_state()}

    def set_state(self, s):
        if s:
            torch.random.set_rng_state(s["torch"])
            np.random.set_state(s["numpy"])


class _Hardware:
    def get_gpus_info(self):
        return {"devices": []}


random = _Random()
hardware = _Hardware()


def improve_reproducibility(*a, **k):
    pass
