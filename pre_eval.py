"""
pre_eval.py — Evaluate the UNEDITED model to get "Pre-edited" numbers for Table 1.

This script mirrors the case-file format produced by evaluate.py so that
the standard experiments/summarize.py can be used unchanged.

Usage (run from the AlphaEdit repo root):

  # CounterFact (MCF)
  python pre_eval.py \
      --model_name=meta-llama/Meta-Llama-3-8B-Instruct \
      --ds_name=mcf \
      --dataset_size_limit=2000 \
      --out_dir=results/pre_edited/Llama3-8B-mcf

  # ZsRE
  python pre_eval.py \
      --model_name=meta-llama/Meta-Llama-3-8B-Instruct \
      --ds_name=zsre \
      --dataset_size_limit=2000 \
      --out_dir=results/pre_edited/Llama3-8B-zsre

  # Summarise afterwards (works with existing summarize.py):
  python experiments/summarize.py \
      --dir_name=pre_edited \
      --runs=Llama3-8B-mcf

Repeat for gpt2-xl, EleutherAI/gpt-j-6b, and any new models.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------------------------------------------------------------- #
# These imports come from the AlphaEdit repo itself.
# Run this script from the repo root so the imports resolve correctly.
# --------------------------------------------------------------------------- #
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    MultiCounterFactDataset,
    get_tfidf_vectorizer,
)
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from util.globals import DATA_DIR


DS_DICT = {
    "mcf":  (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "cf":   (CounterFactDataset,      compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset,           compute_rewrite_quality_zsre),
}


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto" if args.device == "auto" else None,
    )
    if args.device != "auto":
        model = model.to(args.device)
    model.eval()

    tok = AutoTokenizer.from_pretrained(args.model_name)
    tok.pad_token = tok.eos_token  # required for GPT-style models

    # ── Load dataset & generation helpers ───────────────────────────────────
    print("Loading dataset …")
    ds_class, ds_eval = DS_DICT[args.ds_name]
    ds = ds_class(DATA_DIR, tok=tok, size=args.dataset_size_limit)

    # snips & vec are required for MCF generation tests (reference_score, ngram_entropy).
    # For ZsRE, pass None — that eval function ignores them.
    if args.ds_name in ("mcf", "cf"):
        print("Loading AttributeSnippets and TF-IDF vectoriser …")
        snips = AttributeSnippets(DATA_DIR)
        vec   = get_tfidf_vectorizer(DATA_DIR)
    else:
        snips, vec = None, None

    # ── Evaluate every record ────────────────────────────────────────────────
    print(f"Evaluating {len(ds)} records on UNEDITED model → {out_dir}")
    for i, record in enumerate(ds):
        case_id  = record["case_id"]
        out_file = out_dir / f"pre_case_{case_id}.json"

        if out_file.exists():
            print(f"  Skipping case {case_id} (already done)")
            continue

        with torch.no_grad():
            # Pass snips/vec for MCF (computes reference_score + ngram_entropy).
            # For ZsRE, snips/vec are ignored internally.
            metrics_post = ds_eval(model, tok, record, snips, vec)

        # Save in the SAME format as evaluate.py so summarize.py works unchanged.
        # The "post" key is what summarize.py reads for `prefix == "post"`.
        result = {
            "case_id":           case_id,
            "grouped_case_ids":  [case_id],
            "num_edits":         0,       # 0 edits = pre-edited baseline
            "requested_rewrite": record["requested_rewrite"],
            "time":              0.0,
            "post":              metrics_post,
        }

        with open(out_file, "w") as f:
            json.dump(result, f, indent=1)

        if (i + 1) % 100 == 0:
            print(f"  Done {i+1}/{len(ds)}")

    print("Pre-evaluation complete.")
    print(f"\nTo summarise, run:\n"
          f"  python experiments/summarize.py "
          f"--dir_name=pre_edited --runs={out_dir.name}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pre-edited baseline evaluation for AlphaEdit Table 1")
    p.add_argument("--model_name",         required=True,  type=str,
                   help="HuggingFace model id, e.g. meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--ds_name",            required=True,  choices=["mcf","cf","zsre"],
                   help="Dataset: mcf | cf | zsre")
    p.add_argument("--dataset_size_limit", default=2000,   type=int,
                   help="Number of records to evaluate (default 2000, matching the paper)")
    p.add_argument("--out_dir",            required=True,  type=str,
                   help="Directory to write case JSON files into")
    p.add_argument("--device",             default="cuda", type=str,
                   help="'cuda', 'cuda:0', 'cpu', or 'auto' for device_map=auto")
    p.add_argument("--fp16",               action="store_true",
                   help="Load model in float16 to save VRAM")
    args = p.parse_args()
    main(args)