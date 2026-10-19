"""
DISCO-TAB master runner: launches run_experiment as a subprocess per
(dataset, seed, config) so each run gets a clean GPU allocation. Resumable
(skips a run whose results/*.json already exists). Logs unbuffered to logs/.

Usage:
  python run_all.py --mode main   --datasets heart_failure ... --seeds 0 1 2 3 4
  python run_all.py --mode ablation --datasets heart_failure obesity --seeds 0
"""
import os, sys, json, time, argparse, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# Ablation configs expressed as CLI-overridable cfg via a tiny JSON passed through env.
ABLATIONS = {
    "disco":    {},                                   # full system
    "no_token": {"use_token": False},
    "no_sent":  {"use_sent": False},
    "no_feat":  {"use_feat": False},
    "no_row":   {"use_row": False},
    "no_ifrs":  {"use_ifrs": False},
    "row_only": {"use_token": False, "use_sent": False, "use_feat": False},
    "no_rl":    {"ppo_steps": 0, "disc_pretrain": 0},  # SFT only (no RL)
}


def result_exists(ds, tag, seed):
    p = os.path.join(HERE, "results", f"{ds}__{tag}__seed{seed}.json")
    if not os.path.exists(p):
        return False
    try:
        import json
        return json.load(open(p)).get("n_syn", 0) >= 10
    except Exception:
        return False


def launch(ds, tag, seed, cfg_over, steps, batch, sft_epochs):
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    log = os.path.join(HERE, "logs", f"{ds}__{tag}__seed{seed}.log")
    env = dict(os.environ)
    env["HF_HOME"] = os.path.join(HERE, ".hf_cache")
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["DISCO_CFG"] = json.dumps(cfg_over)
    cmd = [PY, "-u", os.path.join(HERE, "run_experiment.py"),
           "--dataset", ds, "--seed", str(seed), "--tag", tag]
    if steps is not None:
        cmd += ["--ppo_steps", str(steps)]
    if batch is not None:
        cmd += ["--ppo_batch", str(batch)]
    if sft_epochs is not None:
        cmd += ["--sft_epochs", str(sft_epochs)]
    t0 = time.time()
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    return r.returncode, round(time.time() - t0, 1), log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["main", "ablation"], default="main")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--tags", nargs="+", default=None,
                    help="ablation tags (ablation mode); default all")
    ap.add_argument("--ppo_steps", type=int, default=None)
    ap.add_argument("--ppo_batch", type=int, default=None)
    ap.add_argument("--sft_epochs", type=int, default=None)
    args = ap.parse_args()

    if args.mode == "main":
        tags = {"disco": {}}
    else:
        chosen = args.tags or list(ABLATIONS.keys())
        tags = {t: ABLATIONS[t] for t in chosen}

    plan = [(ds, t, s) for ds in args.datasets for t in tags for s in args.seeds]
    print(f"planned {len(plan)} runs: datasets={args.datasets} tags={list(tags)} "
          f"seeds={args.seeds}")
    for ds, tag, seed in plan:
        if result_exists(ds, tag, seed):
            print(f"  skip (done): {ds} {tag} seed{seed}")
            continue
        cfg_over = dict(tags[tag])
        steps = cfg_over.pop("ppo_steps", args.ppo_steps)
        print(f"  >>> {ds} {tag} seed{seed} ...", flush=True)
        # Retry transient failures (e.g. CUDA OOM when the shared GPU is busy);
        # a wait lets the co-tenant free VRAM before the next attempt.
        for attempt in range(1, 4):
            rc, secs, log = launch(ds, tag, seed, cfg_over, steps, args.ppo_batch,
                                   args.sft_epochs)
            if rc == 0 and result_exists(ds, tag, seed):
                print(f"      ok in {secs}s  log={log}", flush=True)
                break
            print(f"      FAIL(rc={rc}) attempt {attempt}/3 in {secs}s  log={log}",
                  flush=True)
            if attempt < 3:
                time.sleep(120)


if __name__ == "__main__":
    main()
