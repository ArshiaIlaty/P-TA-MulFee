#!/usr/bin/env python
"""Best-effort trainable-parameter counts per generator, written to
tables/comprehensive/profile_params.csv.

The language-model family (DISCO-TAB, GReaT, RealTabFormer) share a GPT-2 / DistilGPT-2
backbone whose count we read directly from the HuggingFace model. DISCO-TAB also adds
four small discriminators. The shallow baselines (synthcity GANs/VAEs/flows, TabDDPM,
TabSyn, TAEGAN) use compact MLP/GAN nets built lazily at fit time with dataset-dependent
widths, so we report their architecture class and an order-of-magnitude estimate rather
than a spuriously precise number. Everything is wrapped so a failure never aborts the
calling pipeline."""
import csv, os

OUT = os.path.join("tables", "comprehensive", "profile_params.csv")
rows = []


def hf_params(name):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(name)
    return sum(p.numel() for p in m.parameters())


def safe(label, fn, note=""):
    try:
        n = fn()
        rows.append((label, n, note))
        print(f"{label}: {n:,} params  {note}")
    except Exception as e:
        rows.append((label, "", f"n/a ({type(e).__name__})"))
        print(f"{label}: n/a ({e})")


def main():
    # GPT-2 backbone (DISCO-TAB generator; GReaT/RealTabFormer use the same family)
    safe("gpt2_backbone", lambda: hf_params("gpt2"), "GPT-2 124M generator backbone")
    safe("distilgpt2_backbone", lambda: hf_params("distilgpt2"), "DistilGPT-2 (GReaT default)")

    # DISCO-TAB discriminator heads, if a trained checkpoint is present
    def disco_disc():
        import glob, torch
        ckpts = sorted(glob.glob("checkpoints/*disc*.pt") + glob.glob("checkpoints/*discriminator*.pt"))
        if not ckpts:
            raise FileNotFoundError("no discriminator checkpoint")
        sd = torch.load(ckpts[-1], map_location="cpu")
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        return sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
    safe("disco_discriminators", disco_disc, "token/sentence/feature/row heads")

    # Shallow baselines: architecture notes (widths are dataset-dependent at fit time)
    for lab, note in [
        ("ctgan", "conditional GAN, 2x256 MLP gen+disc, ~1-3M"),
        ("tvae", "VAE, 2x128 MLP enc+dec, ~0.3-1M"),
        ("adsgan", "WGAN-GP, MLP gen+disc, ~1-3M"),
        ("pategan", "DP GAN w/ teacher ensemble, MLP, ~1-3M"),
        ("rtvae", "robust VAE, MLP, ~0.3-1M"),
        ("nflow", "neural spline flow, ~0.5-2M"),
        ("tabddpm", "MLP diffusion denoiser, ~1-4M"),
        ("tabsyn", "VAE + latent-space diffusion, ~2-6M"),
        ("taegan", "masked-autoencoder GAN, MLP, ~1-3M"),
        ("marginal", "non-parametric independent sampler, 0 trainable"),
    ]:
        rows.append((lab, "", note))
        print(f"{lab}: (shallow) {note}")

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "params", "note"])
        w.writerows(rows)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
