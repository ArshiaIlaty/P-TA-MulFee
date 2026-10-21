"""
TAGAL (Blanchard et al., "TAGAL: Tabular Data Generation using Agentic LLM
Methods", ECML-PKDD 2025 SynDAiTE workshop) baseline, label "tagal".

Backend note: the chosen DeepSeek-R1-Distill-8B could not be used -- the disk is
full (7 GB free < ~16 GB download) and R1's <think> reasoning traces break
TAGAL's CSV deserializer. We instead use the already-cached, non-reasoning
Qwen2.5-7B-Instruct, loaded in 4-bit NF4 on the T4 (see the llm_handler patch),
registered as "qwen2.5-7b" in llms_infos.

TAGAL is an *agentic augmentation* method (iterative LLM generate + LLM-critic
feedback loop), designed to produce a modest number of rows, not a full
synthetic replica. To use it as a TSTR generator we cap it at N_GEN rows with
MAX_ITER refinement iterations; this is a deliberate, noted caveat.

Run in `discoRTF` (transformers 4.40 + bitsandbytes). Mirrors baselines_llm.py:
writes synthetic/baselines/<ds>__tagal__seed<seed>.csv, resumable, one isolated
subprocess per (dataset,seed) with a timeout (also guards the agentic loop from
hanging if the LLM stops producing parseable rows).
"""
import os, sys, argparse, subprocess, tempfile
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data_raw")
OUTDIR = os.path.join(HERE, "synthetic", "baselines")
TAGAL_ROOT = "/home/ec2-user/disco_llm_models/tagal"
TAGAL_SL = os.path.join(TAGAL_ROOT, "synthloop")
os.makedirs(OUTDIR, exist_ok=True)

DEF_DATASETS = ["heart_failure", "obesity", "diabetes", "surv_whas500",
                "surv_pbc3", "tab_maternal_risk"]
TARGET = "target"           # our raw CSVs; avoid importing disco/data.py (name
                            # clash with TAGAL's own `data` package)
LLM = "qwen2.5-7b"
N_GEN = 150                 # capped synthetic rows (augmentation method)
MAX_ITER = 2
N_SHOTS = 5
MAX_NEW_TOKENS = 2048


def align(syn, real_cols, target_classes):
    syn = syn.copy()
    for c in real_cols:
        if c not in syn.columns:
            syn[c] = np.nan
    syn = syn[real_cols]
    mode = pd.Series(list(target_classes)).mode()[0]
    syn[TARGET] = syn[TARGET].astype(str)
    syn.loc[~syn[TARGET].isin(target_classes), TARGET] = mode
    return syn


def run_one(ds, seed):
    # Drop this script's own dir from sys.path: disco/data.py would otherwise
    # shadow TAGAL's `data` package (`from data.datasets_infos import ...`).
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
    sys.path.insert(0, TAGAL_SL)     # dataloader, serializer, models, feedback_maker, llms_infos
    sys.path.insert(0, TAGAL_ROOT)   # data.datasets_infos (now a real package)
    from dataloader import Dataloader
    from serializer import CSVSerializer
    from models import SynthLoop
    from feedback_maker import LLMFeedback

    full = pd.read_csv(os.path.join(RAW, f"{ds}.csv"))
    full[TARGET] = full[TARGET].astype(str)
    target_classes = set(full[TARGET])
    n_gen = int(min(N_GEN, len(full) * 0.8))

    # TAGAL writes intermediate per-iteration CSVs to the cwd -> isolate them.
    scratch = tempfile.mkdtemp(prefix=f"tagal_{ds}_s{seed}_")
    cwd0 = os.getcwd()
    os.chdir(scratch)
    try:
        dl = Dataloader(ds, parent_directory=RAW, test_size=0.2)
        serializer = CSVSerializer(dl, feat_order="original")
        gen = SynthLoop(LLM, dl, serializer, temperature=0.7,
                        max_new_tokens=MAX_NEW_TOKENS, do_sample=True, random_seed=seed)
        critic = LLMFeedback(LLM, dl, serializer, True, False, 0.7, 1024, True, 42)
        df = gen.generate(critic, n_examples=n_gen, n_iter_max=MAX_ITER,
                          n_shots=N_SHOTS, give_dataset_info=True, verbose=False)
    finally:
        os.chdir(cwd0)
        import shutil
        shutil.rmtree(scratch, ignore_errors=True)

    df = align(df, list(full.columns), target_classes)
    out = os.path.join(OUTDIR, f"{ds}__tagal__seed{seed}.csv")
    df.to_csv(out, index=False)
    print(f"[OK] {ds} tagal seed{seed} -> {df.shape}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DEF_DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--one", nargs=2, default=None, help="internal: DATASET SEED")
    args = ap.parse_args()

    if args.one:
        run_one(args.one[0], int(args.one[1]))
        return

    for ds in args.datasets:
        for seed in args.seeds:
            out = os.path.join(OUTDIR, f"{ds}__tagal__seed{seed}.csv")
            if os.path.exists(out):
                print(f"skip (exists): {ds} tagal seed{seed}", flush=True)
                continue
            base = [sys.executable, __file__, "--one", ds, str(seed)]
            env = {**os.environ, "TOKENIZERS_PARALLELISM": "false",
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
            print(f">>> {ds} tagal seed{seed}", flush=True)
            try:
                r = subprocess.run(base, env=env, timeout=7200)
                if r.returncode != 0 or not os.path.exists(out):
                    print(f"[FAIL] {ds} tagal seed{seed} rc={r.returncode}", flush=True)
            except subprocess.TimeoutExpired:
                print(f"[TIMEOUT] {ds} tagal seed{seed}", flush=True)


if __name__ == "__main__":
    main()
