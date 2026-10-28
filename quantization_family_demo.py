"""Model families and quantization levels: a hands-on comparison.

Companion to QuantizationGuide.md. Uses the Qwen2.5-Instruct family from the
Hugging Face Hub, which Qwen publishes in many sizes (0.5B to 72B parameters),
each one in up to nine GGUF quantization levels (q2_k to fp16). The script:

  1. Discovers the family on the Hub and prints a size x quantization matrix of
     file sizes and licenses, without downloading anything.
  2. Downloads a chosen grid of (size, quantization) files.
  3. Measures each one the same way: load time, RAM, file size, bits per weight,
     perplexity on a fixed passage, p95 latency, generation speed, and macro-F1
     on a 10-intent slice of banking77 (the dataset from the main walkthrough).
  4. Saves results (JSON + CSV) and a trade-off chart under artifacts/results/.

Setup (in the course's model-demo env, Python 3.10+):
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

Usage:
    python quantization_family_demo.py --list                      # Hub discovery only
    python quantization_family_demo.py                             # default grid (~7 GB download)
    python quantization_family_demo.py --sizes 0.5B --quants q4_k_m q8_0 fp16
    python quantization_family_demo.py --quick                     # rehearsal: tiny eval, 2 quants
"""

import argparse
import csv
import gc
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import psutil
from huggingface_hub import HfApi, snapshot_download

FAMILY_AUTHOR = "Qwen"
REPO_PATTERN = re.compile(r"^Qwen/Qwen2\.5-(\d+(?:\.\d+)?B)-Instruct-GGUF$")
REPO_TEMPLATE = "Qwen/Qwen2.5-{size}-Instruct-GGUF"

# Quantization levels, lowest precision first. The names are llama.cpp's.
QUANTS = ["q2_k", "q3_k_m", "q4_0", "q4_k_m", "q5_0", "q5_k_m", "q6_k", "q8_0", "fp16"]

QUANT_NOTES = {
    "q2_k":   ("smallest file; runs on almost anything", "clear quality loss, worst on small models"),
    "q3_k_m": ("small; usable on larger models", "noticeable loss; small models degrade fast"),
    "q4_0":   ("small, simple, fast on many CPUs", "older scheme; less accurate than q4_k_m at similar size"),
    "q4_k_m": ("the usual default: ~70% smaller than fp16, small loss", "measurable loss on hard reasoning"),
    "q5_0":   ("near-lossless for most tasks", "bigger than q4_k_m; older scheme"),
    "q5_k_m": ("near-lossless; good if RAM allows", "~15% bigger than q4_k_m"),
    "q6_k":   ("practically indistinguishable from fp16", "saves less memory"),
    "q8_0":   ("effectively lossless reference", "half the savings of 4-bit; slower on CPU than q4"),
    "fp16":   ("the original weights; the quality ceiling", "largest and slowest; 2 bytes per weight"),
}

SIZE_NOTES = {
    "0.5B": ("tiny, fast, fits on a phone or edge device", "weak instruction following; easily confused"),
    "1.5B": ("fast on CPU; decent for narrow tasks", "limited knowledge and reasoning"),
    "3B":   ("good quality per GB on a laptop", "non-commercial Qwen Research license"),
    "7B":   ("strong general model; runs on one consumer GPU", "~4.7 GB even at q4_k_m; slow on CPU"),
    "14B":  ("much better reasoning", "needs a 12-16 GB GPU at 4-bit"),
    "32B":  ("near-frontier for open models of its era", "needs a 24 GB GPU at 4-bit"),
    "72B":  ("best in the family", "multi-GPU or very large RAM; Qwen license with usage limits"),
}

# A 10-intent slice of banking77: distinct enough for a zero-shot LLM to have a fair chance.
INTENTS = [
    "card_arrival", "lost_or_stolen_card", "exchange_rate", "top_up_failed", "pin_blocked",
    "cash_withdrawal_charge", "declined_card_payment", "change_pin", "terminate_account",
    "wrong_amount_of_cash_received",
]

SYSTEM_PROMPT = (
    "You route customer messages for a bank. Reply with exactly one intent label "
    "from this list and nothing else:\n" + "\n".join(f"- {i}" for i in INTENTS)
)

# Fixed passage for perplexity. Written for this demo so it is not in any training set verbatim.
PPL_TEXT = (
    "When a customer contacts the bank, the first job is to understand what they need. "
    "Some people write a single word, such as 'card', while others describe a long story "
    "about a trip abroad, a declined payment at a restaurant, and a cash machine that "
    "kept their card. A routing system reads each message and sends it to the team best "
    "placed to help. If the message is about a lost card, the security team blocks the "
    "card and orders a replacement. If it is about an exchange rate, the payments team "
    "explains how the rate was calculated and whether a fee applied. Messages that the "
    "system cannot place with confidence go to a person, who reads them and decides. "
    "Over time the bank reviews these hand-routed messages to find new kinds of request. "
    "A new product, a change in fees, or a holiday season can each shift what customers "
    "ask about, and the routing model must be checked again when that happens. Good "
    "routing is quick, accurate, and honest about its own uncertainty."
)

FREE_TEXT_PROMPT = "I was charged twice for the same coffee this morning. What should I do?"

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts" / "results" / "quant_family"


def size_key(size):
    return float(size.rstrip("B"))


# ----------------------------------------------------------------- 1. discovery

def discover_family():
    """Find every size in the family and the GGUF files each one offers. Metadata only."""
    api = HfApi()
    repos = sorted(
        (m.id for m in api.list_models(author=FAMILY_AUTHOR, search="Instruct-GGUF")
         if REPO_PATTERN.match(m.id)),
        key=lambda r: size_key(REPO_PATTERN.match(r).group(1)),
    )
    family = {}
    for repo in repos:
        info = api.model_info(repo, files_metadata=True)
        size = REPO_PATTERN.match(repo).group(1)
        files = {}
        for q in QUANTS:
            # Big models split one quantization across shards: name-q4_k_m-00001-of-00002.gguf
            parts = [s for s in info.siblings
                     if re.search(rf"-{q}(-\d{{5}}-of-\d{{5}})?\.gguf$", s.rfilename)]
            if parts:
                files[q] = sum(s.size or 0 for s in parts)
        license_ = (info.card_data or {}).get("license") if info.card_data else None
        if license_ == "other" and info.card_data:
            license_ = info.card_data.get("license_name") or "other"
        family[size] = {"repo": repo, "license": license_, "downloads": info.downloads, "files": files}
    return family


def print_family(family):
    print(f"\nQwen2.5-Instruct GGUF family on the Hub ({len(family)} sizes)\n")
    print(f"{'parameters':>10}  {'license':<16} {'downloads':>10}  " + " ".join(f"{q + ' (GB)':>11}" for q in QUANTS))
    for size, f in family.items():
        cells = " ".join(f"{f['files'][q] / 1e9:11.2f}" if q in f["files"] else f"{'-':>11}" for q in QUANTS)
        print(f"{size:>10}  {str(f['license']):<16} {f['downloads'] or 0:>10,}  {cells}")
    print()

    print("Family members: advantages / disadvantages")
    for size in family:
        pro, con = SIZE_NOTES.get(size, ("", ""))
        print(f"  {size:>5}  + {pro}\n         - {con}")
    print("\nQuantization levels: advantages / disadvantages")
    for q in QUANTS:
        pro, con = QUANT_NOTES[q]
        print(f"  {q:>7}  + {pro}\n           - {con}")
    print()


# ----------------------------------------------------------------- 2. download

def download(size, quant):
    """Fetch one quantization (all shards) into the HF cache; return the path llama.cpp loads."""
    repo = REPO_TEMPLATE.format(size=size)
    local = snapshot_download(repo, allow_patterns=[f"*-{quant}.gguf", f"*-{quant}-*-of-*.gguf"])
    paths = sorted(Path(local).glob(f"*-{quant}*.gguf"))
    if not paths:
        raise FileNotFoundError(f"{repo} has no {quant} file")
    return paths[0], sum(p.stat().st_size for p in paths)   # llama.cpp finds later shards itself


# ----------------------------------------------------------------- 3. measure

def load_eval_set(n_per_intent, seed=42):
    import datasets
    test = datasets.load_dataset("mteb/banking77", split="test").to_pandas()
    test = test[test["label_text"].isin(INTENTS)]
    test = test.groupby("label_text").sample(n=n_per_intent, random_state=seed)
    return test["text"].tolist(), test["label_text"].tolist()


def bootstrap_ci(y, p, n=1000, seed=42):
    """95% interval for macro-F1 by resampling the eval set, as in the main walkthrough."""
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        scores.append(f1_score(y[idx], p[idx], average="macro", labels=INTENTS, zero_division=0))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def perplexity(model_path, n_threads):
    """exp(mean negative log-likelihood) of PPL_TEXT. Lower is better; compare against fp16."""
    from llama_cpp import Llama
    llm = Llama(str(model_path), n_ctx=512, logits_all=True, n_threads=n_threads, verbose=False)
    tokens = llm.tokenize(PPL_TEXT.encode(), add_bos=False)
    llm.reset()
    llm.eval(tokens)
    logits = np.asarray(llm.scores[: len(tokens) - 1], dtype=np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    nll = -log_probs[np.arange(len(tokens) - 1), tokens[1:]].mean()
    del llm
    gc.collect()
    return float(math.exp(nll))


def measure(size, quant, texts, labels, n_threads):
    from llama_cpp import Llama, LlamaGrammar
    from sklearn.metrics import f1_score

    path, file_bytes = download(size, quant)
    proc = psutil.Process()
    gc.collect()
    rss0 = proc.memory_info().rss
    t0 = time.perf_counter()
    # use_mmap=False reads the weights into RAM, so the RSS change is the model's real footprint.
    llm = Llama(str(path), n_ctx=1024, n_threads=n_threads, use_mmap=False, verbose=False)
    load_s = time.perf_counter() - t0
    ram_mb = (proc.memory_info().rss - rss0) / 2**20

    import llama_cpp
    n_params = llama_cpp.llama_model_n_params(llm.model)

    # The grammar forces the output to be exactly one label, so every answer is scorable.
    grammar = LlamaGrammar.from_string("root ::= " + " | ".join(f'"{i}"' for i in INTENTS), verbose=False)

    def classify(text):
        out = llm.create_chat_completion(
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}],
            grammar=grammar, temperature=0.0, max_tokens=24)
        return out["choices"][0]["message"]["content"].strip()

    classify(texts[0])   # warm-up: caches the system prompt's KV and pages in code
    preds, times = [], []
    for t in texts:
        t1 = time.perf_counter()
        preds.append(classify(t))
        times.append((time.perf_counter() - t1) * 1000)

    t1 = time.perf_counter()
    reply = llm.create_chat_completion(
        messages=[{"role": "user", "content": FREE_TEXT_PROMPT}], temperature=0.0, max_tokens=96)
    gen_s = time.perf_counter() - t1
    gen_tokens = reply["usage"]["completion_tokens"]

    del llm
    gc.collect()
    ppl = perplexity(path, n_threads)
    lo, hi = bootstrap_ci(labels, preds)

    return {
        "size": size,
        "quant": quant,
        "file_mb": round(file_bytes / 2**20, 1),
        "params_m": round(n_params / 1e6, 1),
        "bits_per_weight": round(file_bytes * 8 / n_params, 2),
        "ram_mb": round(ram_mb, 1),
        "load_s": round(load_s, 2),
        "perplexity": round(ppl, 3),
        "macro_f1": round(f1_score(labels, preds, average="macro", labels=INTENTS, zero_division=0), 4),
        "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
        "accuracy": round(float(np.mean([p == y for p, y in zip(preds, labels)])), 4),
        "p95_ms": round(float(np.percentile(times, 95)), 1),
        "gen_tok_per_s": round(gen_tokens / gen_s, 1),
        "sample_reply": reply["choices"][0]["message"]["content"].strip(),
        "n_eval": len(labels),
    }


# ----------------------------------------------------------------- 4. report

def print_results(rows):
    ref = {r["size"]: r["perplexity"] for r in rows if r["quant"] == "fp16"}
    print(f"\n{'size':>5} {'quant':>7} {'file MB':>8} {'bpw':>5} {'RAM MB':>7} {'load s':>6} "
          f"{'ppl':>7} {'vs fp16':>8} {'F1':>6} {'F1 95% CI':>13} {'p95 ms':>7} {'tok/s':>6}")
    for r in rows:
        delta = f"{(r['perplexity'] / ref[r['size']] - 1) * 100:+7.1f}%" if r["size"] in ref else f"{'':>8}"
        print(f"{r['size']:>5} {r['quant']:>7} {r['file_mb']:8.0f} {r['bits_per_weight']:5.2f} "
              f"{r['ram_mb']:7.0f} {r['load_s']:6.2f} {r['perplexity']:7.2f} {delta} "
              f"{r['macro_f1']:6.3f} [{r['ci95_low']:.2f}, {r['ci95_high']:.2f}] {r['p95_ms']:7.0f} {r['gen_tok_per_s']:6.1f}")
    print("\nSample replies to:", FREE_TEXT_PROMPT)
    for r in rows:
        print(f"\n[{r['size']} {r['quant']}] {r['sample_reply'][:300]}")


def plot(rows, path):
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]   # fixed order, one per family size
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    sizes = sorted({r["size"] for r in rows}, key=size_key)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor="#fcfcfb")
    panels = [("perplexity", "Perplexity on fixed passage (lower is better)"),
              ("macro_f1", "Macro-F1 with 95% CI, 10 banking77 intents (higher is better)")]
    for ax, (metric, title) in zip(axes, panels):
        ax.set_facecolor("#fcfcfb")
        for i, size in enumerate(sizes):
            pts = sorted((r for r in rows if r["size"] == size), key=lambda r: r["file_mb"])
            x = [r["file_mb"] for r in pts]
            y = [r[metric] for r in pts]
            if metric == "macro_f1":   # small eval set: show the bootstrap interval
                ax.fill_between(x, [r["ci95_low"] for r in pts], [r["ci95_high"] for r in pts],
                                color=series[i % len(series)], alpha=0.12, lw=0)
            ax.plot(x, y, color=series[i % len(series)], lw=2, marker="o", ms=8,
                    markeredgecolor="#fcfcfb", markeredgewidth=2, label=f"Qwen2.5-{size}")
            for r in (pts[0], pts[-1]):   # label only the extremes to avoid clutter
                ax.annotate(r["quant"], (r["file_mb"], r[metric]), textcoords="offset points",
                            xytext=(0, 9), ha="center", fontsize=8, color=muted)
        ax.set_xscale("log")
        ax.set_xticks([400, 700, 1000, 2000, 3500])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.minorticks_off()
        ax.set_xlabel("File size (MB, log scale)", color=muted)
        ax.set_title(title, color=ink, fontsize=11, loc="left")
        ax.grid(True, color=grid, lw=0.8)
        ax.tick_params(colors=muted)
        for s in ax.spines.values():
            s.set_color(grid)
    axes[0].legend(frameon=False, labelcolor=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="only show the family on the Hub; download nothing")
    ap.add_argument("--sizes", nargs="+", default=["0.5B", "1.5B"])
    ap.add_argument("--quants", nargs="+", default=["q2_k", "q3_k_m", "q4_k_m", "q5_k_m", "q6_k", "q8_0", "fp16"],
                    choices=QUANTS)
    ap.add_argument("--per-intent", type=int, default=15, help="test messages per intent")
    ap.add_argument("--threads", type=int, default=psutil.cpu_count(logical=False))
    ap.add_argument("--quick", action="store_true", help="rehearsal: 3 messages per intent, two quants")
    args = ap.parse_args()

    family = discover_family()
    print_family(family)
    if args.list:
        return
    if args.quick:
        args.per_intent, args.quants = 3, ["q4_k_m", "fp16"]

    texts, labels = load_eval_set(args.per_intent)
    print(f"Eval set: {len(texts)} banking77 test messages across {len(INTENTS)} intents; "
          f"{args.threads} CPU threads\n")

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for size in sorted(args.sizes, key=size_key):
        for quant in [q for q in QUANTS if q in args.quants]:   # always lowest precision first
            if quant not in family.get(size, {}).get("files", {}):
                print(f"skip {size} {quant}: not published")
                continue
            print(f"--- Qwen2.5-{size}-Instruct {quant} ...", flush=True)
            r = measure(size, quant, texts, labels, args.threads)
            print(f"    F1={r['macro_f1']:.3f}  ppl={r['perplexity']:.2f}  p95={r['p95_ms']:.0f} ms  "
                  f"RAM={r['ram_mb']:.0f} MB")
            rows.append(r)

    print_results(rows)
    (OUT / "results.json").write_text(json.dumps({"family": family, "results": rows}, indent=2))
    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot(rows, OUT / "quantization_tradeoffs.png")
    print(f"\nSaved {OUT.relative_to(ROOT)}/results.json, results.csv, quantization_tradeoffs.png")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
