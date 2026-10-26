"""
Measure real peak GPU memory (per-process) + wall-clock for each generator's
fit+sample, so the paper's computational-cost table uses measured numbers
instead of placeholders. Run per family in its own conda env, e.g.:
  conda run -n discobase python profile_cost.py --family ctgan   --dataset obesity
  conda run -n discobase python profile_cost.py --family tvae    --dataset obesity
  conda run -n discobase python profile_cost.py --family taegan  --dataset obesity
  conda run -n discodiff python profile_cost.py --family tabddpm --dataset obesity
Appends {family: {dataset: {vram_gb, time_s}}} to results/cost_profile.json.
"""
import os, sys, time, json, argparse, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch

OUT = os.path.join(HERE, "results", "cost_profile.json")


def measure(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    fn()
    torch.cuda.synchronize()
    return time.time() - t0, torch.cuda.max_memory_allocated() / 1e9


def run_family(family, dataset):
    import baselines as B
    from data import load_dataset
    b = load_dataset(dataset, seed=0)
    n = min(len(b.df_train), 2000)
    if family in ("ctgan", "tvae", "gaussian_copula"):
        def go(): B.generate(dataset, family, seed=0, epochs=100)
    elif family == "tabddpm":
        import baselines_diffusion as D
        def go(): D.generate_tabddpm(dataset, seed=0) if hasattr(D, "generate_tabddpm") else D.generate(dataset, seed=0)
    elif family == "taegan":
        import run_taegan as T
        def go(): T.run_one(dataset, "taegan_prof", 0)
    else:
        raise SystemExit(f"unknown family {family}")
    dt, vram = measure(go)
    rec = {}
    if os.path.exists(OUT):
        try: rec = json.load(open(OUT))
        except Exception: rec = {}
    rec.setdefault(family, {})[dataset] = {"vram_gb": round(vram, 3), "time_s": round(dt, 1)}
    json.dump(rec, open(OUT, "w"), indent=2)
    print(f"[{family} {dataset}] peak VRAM {vram:.2f} GB | time {dt:.1f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--dataset", default="obesity")
    a = ap.parse_args()
    run_family(a.family, a.dataset)
