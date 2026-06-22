"""
diagnose_phi_gemma.py
=====================
Minimal diagnostic to identify exactly why Phi3-3.8B and Gemma2-2B produce
0% accuracy and ~50% success on CounterFact / ZsRE.

Run from the AlphaEdit repo root:

  python diagnose_phi_gemma.py --model microsoft/Phi-3-mini-4k-instruct
  python diagnose_phi_gemma.py --model google/gemma-2-2b-it

Each test is self-contained and prints PASS / FAIL with a clear explanation.
The tests are ordered from most-likely to least-likely root cause.
"""

import argparse
import json
import sys
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── helpers ──────────────────────────────────────────────────────────────────

SEP = "─" * 70

def section(title):
    print(f"\n{SEP}\nTEST: {title}\n{SEP}")

def ok(msg):
    print(f"  ✅  PASS — {msg}")

def fail(msg):
    print(f"  ❌  FAIL — {msg}")

def info(msg):
    print(f"  ℹ   {msg}")

# ── load model ────────────────────────────────────────────────────────────────

def load_model(model_name: str):
    print(f"\nLoading {model_name} …")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    tok.pad_token = tok.eos_token
    print(f"  Loaded. dtype={next(model.parameters()).dtype}, "
          f"device={next(model.parameters()).device}")
    return model, tok


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1 — BOS TOKEN CONTAMINATION (most likely cause of acc=0%)
# This is exactly what eval_utils_counterfact.py does when it tokenises a
# target string such as " Paris".  If the tokeniser prepends a BOS token,
# token_ids[0] == bos_id, and the code then checks whether the model predicts
# BOS as the next token — which it never does → acc always 0%.
# ══════════════════════════════════════════════════════════════════════════════

def test_bos_contamination(model, tok):
    section("BOS token in target tokenisation  →  causes acc = 0%")

    sample_targets = ["Paris", "London", "Apple", "football", "English"]
    bos_id = tok.bos_token_id

    info(f"tokenizer.bos_token_id = {bos_id}  ({tok.bos_token})")
    info(f"tokenizer class: {type(tok).__name__}")

    contaminated = []
    for t in sample_targets:
        ids_with_space    = tok(f" {t}")["input_ids"]
        ids_without_space = tok(t)["input_ids"]
        for label, ids in [("' "+t+"'", ids_with_space), ("'"+t+"'", ids_without_space)]:
            if len(ids) > 1 and ids[0] == bos_id:
                contaminated.append((label, ids))
                info(f"  tok({label}) = {ids}  ← BOS={bos_id} at index 0  ← PROBLEM")
            else:
                info(f"  tok({label}) = {ids}  ← OK, no BOS prefix")

    if contaminated:
        fail(
            f"{len(contaminated)} of {len(sample_targets)*2} tokenisations include BOS at index 0. "
            f"eval_utils_counterfact.py will look for BOS as the predicted next token → acc=0% always."
        )
        print("\n  FIX: in eval_utils_counterfact.py, add BOS stripping for this tokeniser:")
        print("       if ids[0] == tok.bos_token_id: ids = ids[1:]")
        print("  The same fix is needed in THREE places:")
        print("    1. target_id computation for *_prompts_correct")
        print("    2. suffix token ids for log-prob computation")
        print("    3. prefix_tok_len adjustment (subtract 1 if BOS in prefix)")
        return True   # bug confirmed
    else:
        ok("No BOS contamination in target tokenisation.")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 — PREFIX LENGTH OFF-BY-ONE  →  causes ~50% success rate
# The log-prob computation indexes into logits using prefix_tok_len.
# If prefix_tok_len is wrong by 1 (because BOS is counted in the prefix
# but not stripped), the wrong logit position is used → random probabilities.
# ══════════════════════════════════════════════════════════════════════════════

def test_prefix_length(model, tok):
    section("Prefix length off-by-one  →  causes success ≈ 50%")

    prompt = "The capital of France is"
    target = " Paris"

    # Simulate exactly what eval_utils_counterfact.py does
    target_ids_raw  = tok(target)["input_ids"]
    target_ids_stripped = target_ids_raw[1:] if (
        len(target_ids_raw) > 1 and target_ids_raw[0] == tok.bos_token_id
    ) else target_ids_raw

    prompt_ids = tok(prompt)["input_ids"]

    # prefix_tok_len WITHOUT any correction (what the code does for non-llama)
    prefix_tok_len_wrong   = len(tok(prompt)["input_ids"])
    # prefix_tok_len WITH bos correction (what the code does for llama)
    prefix_tok_len_correct = prefix_tok_len_wrong - (
        1 if (tok.bos_token_id is not None and
              len(prompt_ids) > 0 and prompt_ids[0] == tok.bos_token_id)
        else 0
    )

    info(f"prompt: '{prompt}'")
    info(f"target: '{target}'")
    info(f"target_ids (raw): {target_ids_raw}")
    info(f"target_ids (stripped): {target_ids_stripped}")
    info(f"prefix_tok_len (uncorrected): {prefix_tok_len_wrong}")
    info(f"prefix_tok_len (bos-corrected): {prefix_tok_len_correct}")

    # Run one forward pass to compare logit at wrong vs correct position
    full_input = tok(prompt + target, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**full_input).logits[0]  # (seq_len, vocab)

    if tok.bos_token_id is not None and full_input["input_ids"][0][0] == tok.bos_token_id:
        logits = logits[1:]  # strip BOS position from logits too

    seq_len = logits.shape[0]
    info(f"logits shape (after any BOS strip): {logits.shape}")

    # What token does the model actually predict at the position just before target?
    correct_pos = prefix_tok_len_correct - 1
    wrong_pos   = prefix_tok_len_wrong - 1

    if correct_pos < seq_len and len(target_ids_stripped) > 0:
        target_tok = target_ids_stripped[0]
        prob_at_correct = torch.softmax(logits[correct_pos], dim=-1)[target_tok].item()
        prob_at_wrong   = torch.softmax(logits[wrong_pos],   dim=-1)[target_tok].item() \
                          if wrong_pos < seq_len else float("nan")

        info(f"P('{tok.decode([target_tok])}') at CORRECT position [{correct_pos}]: {prob_at_correct:.4f}")
        info(f"P('{tok.decode([target_tok])}') at WRONG   position [{wrong_pos}]:   {prob_at_wrong:.4f}")

        if abs(prefix_tok_len_correct - prefix_tok_len_wrong) > 0:
            fail(
                f"prefix_tok_len differs by {prefix_tok_len_wrong - prefix_tok_len_correct} "
                f"without BOS correction.  Code indexes wrong logit position → "
                f"log-probs are for the wrong token → success≈50%."
            )
            return True
        else:
            ok("No prefix length off-by-one detected.")
            return False
    else:
        info("Could not verify logit position (sequence too short).")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3 — LOGIT SHIFT (separate from prefix length)
# The code does:  logits = model(**prompt_tok).logits
# Then for llama: logits = logits[:, 1:, :]
# Without this shift, position i gives probability conditioned on tokens 0..i-1
# but the code uses position prefix_tok_len+j-1 to get P(target[j]).
# If the BOS-shift is missing, every position is off by 1 in the sequence dim.
# ══════════════════════════════════════════════════════════════════════════════

def test_logit_shift(model, tok):
    section("Logit sequence-dimension shift  →  secondary cause of wrong probs")

    prompt = "The Eiffel Tower is located in"
    target = " Paris"

    prompt_ids  = tok(prompt)["input_ids"]
    full_ids    = tok(prompt + target, return_tensors="pt").to(model.device)

    with torch.no_grad():
        logits_raw = model(**full_ids).logits[0]   # (seq, vocab) — NOT shifted

    target_ids = tok(target)["input_ids"]
    if target_ids[0] == tok.bos_token_id:
        target_ids = target_ids[1:]

    prefix_len = len(prompt_ids)
    bos_in_prefix = (len(prompt_ids) > 0 and prompt_ids[0] == tok.bos_token_id)

    info(f"BOS in prefix: {bos_in_prefix}")
    info(f"logits_raw shape: {logits_raw.shape}")

    # With no shift, logits[i] predicts token i+1
    # Position to look at for first target token = prefix_len - 1 (0-indexed, no BOS strip)
    # OR prefix_len - 2 if BOS is counted in prefix but logits aren't shifted

    if bos_in_prefix:
        pos_no_shift   = prefix_len - 1     # wrong: includes BOS in prefix count
        pos_with_shift = prefix_len - 2     # correct: logit[i] after BOS strip
        logits_shifted = logits_raw[1:]     # simulate the llama-style shift
    else:
        pos_no_shift   = prefix_len - 1
        pos_with_shift = prefix_len - 1
        logits_shifted = logits_raw

    if len(target_ids) > 0 and pos_with_shift >= 0:
        t = target_ids[0]
        p_no_shift   = torch.softmax(logits_raw[pos_no_shift],   dim=-1)[t].item() \
                       if pos_no_shift < logits_raw.shape[0] else float("nan")
        p_with_shift = torch.softmax(logits_shifted[pos_with_shift], dim=-1)[t].item() \
                       if pos_with_shift < logits_shifted.shape[0] else float("nan")

        info(f"P('{tok.decode([t])}') WITHOUT logit shift: {p_no_shift:.4f}")
        info(f"P('{tok.decode([t])}') WITH    logit shift: {p_with_shift:.4f}")

        top_no_shift   = tok.decode([logits_raw[pos_no_shift].argmax().item()])
        top_with_shift = tok.decode([logits_shifted[pos_with_shift].argmax().item()])
        info(f"Top-1 token WITHOUT shift: '{top_no_shift}'")
        info(f"Top-1 token WITH    shift: '{top_with_shift}'")

        if bos_in_prefix and abs(p_with_shift - p_no_shift) > 0.01:
            fail("Logit shift matters for this model. Without the "
                 "`logits = logits[:, 1:, :]` correction, probabilities are "
                 "computed at the wrong sequence position.")
            return True
        else:
            ok("Logit shift does not significantly affect results (or model has no BOS).")
            return False
    else:
        info("Skipped — sequence too short.")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4 — GENERATION QUALITY  →  explains consistency ≈ 0
# Does the model produce meaningful text when given a bare subject prompt?
# (This is exactly what eval_utils_counterfact uses for Flu/Consis.)
# ══════════════════════════════════════════════════════════════════════════════

def test_generation_quality(model, tok):
    section("Generation quality from bare subject prompt  →  explains Flu/Consis drop")

    prompts = [
        "The Eiffel Tower",
        "Steve Jobs",
        "The capital of Germany is",
    ]

    for prompt in prompts:
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                ids["input_ids"],
                max_new_tokens=40,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        generated = tok.decode(out[0], skip_special_tokens=True)
        new_tokens = generated[len(prompt):]
        token_count = len(tok(new_tokens)["input_ids"])
        info(f"Prompt: '{prompt}'")
        info(f"  Generated ({token_count} new tokens): '{new_tokens[:120]}'")

        if token_count <= 3:
            fail(f"Model generates only {token_count} token(s) before stopping. "
                 "This produces near-empty text → TF-IDF similarity ≈ 0 → Consis ≈ 0.")
        else:
            ok(f"Model generates {token_count} tokens — generation is working.")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5 — ARCHITECTURE CHECK
# Confirm module paths match the hparams JSON exactly.
# A single typo here means get_cov() hooks the wrong module and the null space
# projection is computed over garbage activations.
# ══════════════════════════════════════════════════════════════════════════════

def test_architecture(model, tok):
    section("Module path verification  →  confirms hparams JSON is correct")

    # Expected paths from our hparam files
    model_name_lower = model.config._name_or_path.lower()
    if "phi" in model_name_lower:
        expected = {
            "rewrite_module_tmp": "model.layers.{}.mlp.down_proj",
            "layer_module_tmp":   "model.layers.{}",
            "mlp_module_tmp":     "model.layers.{}.mlp",
            "attn_module_tmp":    "model.layers.{}.self_attn",
            "ln_f_module":        "model.norm",
            "lm_head_module":     "lm_head",
        }
    elif "gemma" in model_name_lower:
        expected = {
            "rewrite_module_tmp": "model.layers.{}.mlp.down_proj",
            "layer_module_tmp":   "model.layers.{}",
            "mlp_module_tmp":     "model.layers.{}.mlp",
            "attn_module_tmp":    "model.layers.{}.self_attn",
            "ln_f_module":        "model.norm",
            "lm_head_module":     "lm_head",
        }
    else:
        info("Unknown model family — skipping architecture check.")
        return

    layer_idx = 0
    all_module_names = {name for name, _ in model.named_modules()}

    for hparam_key, template in expected.items():
        resolved = template.replace("{}", str(layer_idx))
        if resolved in all_module_names:
            ok(f"{hparam_key}: '{resolved}' exists in model ✓")
        else:
            fail(f"{hparam_key}: '{resolved}' NOT found in model — hparams JSON has wrong path!")

    # Extra: check for Gemma2's post_feedforward_layernorm
    if "gemma" in model_name_lower:
        post_ln = f"model.layers.{layer_idx}.post_feedforward_layernorm"
        pre_ln  = f"model.layers.{layer_idx}.pre_feedforward_layernorm"
        if post_ln in all_module_names:
            fail(
                f"Gemma2 has '{post_ln}' — this normalises the MLP output BEFORE "
                f"the residual add. AlphaEdit's v* optimisation does not account for "
                f"this layer, causing the effective weight update to be wrong in scale."
            )
        if pre_ln in all_module_names:
            info(f"Note: '{pre_ln}' also present — changes input distribution to MLP.")

    # Extra: check for Phi3's fused QKV
    if "phi" in model_name_lower:
        fused_qkv = f"model.layers.{layer_idx}.self_attn.qkv_proj"
        sep_q     = f"model.layers.{layer_idx}.self_attn.q_proj"
        if fused_qkv in all_module_names and sep_q not in all_module_names:
            info(
                f"Phi3 uses fused qkv_proj (not separate q/k/v). "
                f"attn_module_tmp is only used for identification, not editing, "
                f"so this should NOT cause failures — but note it for your paper."
            )


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6 — DIRECT PROBABILITY SANITY CHECK
# Compute P("Paris") given a prompt the model should know.
# This tells us whether the probability machinery itself is working.
# ══════════════════════════════════════════════════════════════════════════════

def test_probability_sanity(model, tok):
    section("Direct probability sanity check  →  confirms log-prob machinery works")

    # A fact the model should know well
    prompt  = "The capital of France is"
    correct = " Paris"
    wrong   = " Berlin"

    def compute_logprob(prompt_str, target_str):
        """Replicate exactly what eval_utils_counterfact does."""
        target_ids_raw = tok(target_str)["input_ids"]

        # Strip BOS — this is the fix we propose
        bos = tok.bos_token_id
        if bos is not None and len(target_ids_raw) > 1 and target_ids_raw[0] == bos:
            target_ids = target_ids_raw[1:]
            bos_stripped = True
        else:
            target_ids = target_ids_raw
            bos_stripped = False

        prompt_ids = tok(prompt_str)["input_ids"]
        bos_in_prefix = (bos is not None and len(prompt_ids) > 0 and prompt_ids[0] == bos)

        full_tok = tok([f"{prompt_str}{target_str}"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**full_tok).logits

        # Apply the BOS-shift if the model adds BOS
        if bos_in_prefix:
            logits = logits[:, 1:, :]

        prefix_tok_len = len(prompt_ids)
        if bos_in_prefix:
            prefix_tok_len -= 1

        total_lp = 0.0
        for j, cur_tok in enumerate(target_ids):
            lp = -torch.nn.functional.log_softmax(
                logits[0, prefix_tok_len + j - 1, :], dim=0
            )[cur_tok].item()
            total_lp += lp
        avg_lp = total_lp / len(target_ids)

        return np.exp(-avg_lp), bos_stripped

    p_correct, bos_stripped = compute_logprob(prompt, correct)
    p_wrong,   _            = compute_logprob(prompt, wrong)

    info(f"Prompt: '{prompt}'")
    info(f"BOS was stripped from targets: {bos_stripped}")
    info(f"P('{correct}') = {p_correct:.6f}")
    info(f"P('{wrong}')   = {p_wrong:.6f}")

    if p_correct > p_wrong:
        ok(f"Model correctly assigns P('{correct}') > P('{wrong}') — "
           f"probability machinery is working WITH BOS fix applied.")
    else:
        fail(f"Model assigns P('{correct}') <= P('{wrong}') even with BOS fix. "
             f"Deeper architectural incompatibility suspected.")

    # Now test WITHOUT fix to show the damage
    info("\n  --- Repeating WITHOUT BOS fix (to show what the current code does) ---")
    def compute_logprob_broken(prompt_str, target_str):
        target_ids = tok(target_str)["input_ids"]   # NO stripping
        prompt_ids = tok(prompt_str)["input_ids"]
        prefix_tok_len = len(prompt_ids)            # NO correction

        full_tok = tok([f"{prompt_str}{target_str}"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**full_tok).logits       # NO shift

        total_lp = 0.0
        for j, cur_tok in enumerate(target_ids):
            pos = prefix_tok_len + j - 1
            if pos >= logits.shape[1]:
                break
            lp = -torch.nn.functional.log_softmax(
                logits[0, pos, :], dim=0
            )[cur_tok].item()
            total_lp += lp
        avg_lp = total_lp / max(len(target_ids), 1)
        return np.exp(-avg_lp)

    p_correct_broken = compute_logprob_broken(prompt, correct)
    p_wrong_broken   = compute_logprob_broken(prompt, wrong)
    info(f"BROKEN — P('{correct}') = {p_correct_broken:.6f}")
    info(f"BROKEN — P('{wrong}')   = {p_wrong_broken:.6f}")
    if abs(p_correct_broken - p_wrong_broken) < 0.01:
        fail("Without BOS fix, both targets get nearly equal probability → "
             "success metric ≈ 50%, confirming this is the root cause.")
    else:
        info("Interestingly, broken version still separates probabilities — "
             "investigate further.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="HuggingFace model ID, e.g. microsoft/Phi-3-mini-4k-instruct")
    args = parser.parse_args()

    print(f"\n{'═'*70}")
    print(f"  AlphaEdit Evaluation Diagnostic")
    print(f"  Model: {args.model}")
    print(f"{'═'*70}")

    model, tok = load_model(args.model)

    findings = {}
    findings["bos_contamination"]  = test_bos_contamination(model, tok)
    findings["prefix_length"]      = test_prefix_length(model, tok)
    findings["logit_shift"]        = test_logit_shift(model, tok)
    test_generation_quality(model, tok)
    test_architecture(model, tok)
    test_probability_sanity(model, tok)

    print(f"\n{'═'*70}")
    print("  SUMMARY")
    print(f"{'═'*70}")
    n_bugs = sum(findings.values())
    if n_bugs == 0:
        print("  No evaluation bugs detected. Issue may be algorithmic (see Test 4/5).")
    else:
        print(f"  {n_bugs} evaluation bug(s) confirmed. The fix is in eval_utils_counterfact.py:")
        print("  → Add BOS stripping to _target_token_ids() for all target tokenisations.")
        print("  → Add the `logits = logits[:, 1:, :]` shift for BOS-adding tokenisers.")
        print("  → Subtract 1 from prefix_tok_len for BOS-adding tokenisers.")
        print("\n  After the fix, re-run pre_eval.py and then the full AlphaEdit evaluation.")

    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()