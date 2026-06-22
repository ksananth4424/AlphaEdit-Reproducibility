"""
Step 1b: Verify that AlphaEdit's prefix/suffix-length subtraction trick
(rome/repr_tools.py: get_words_idxs_in_templates) correctly finds the
subject's last token under the new tokenizer.

This is a pure tokenizer test -- no model forward pass needed -- so it's
cheap to run across many (prompt, subject) pairs.
"""
from transformers import AutoTokenizer

def alphaedit_predicted_idx(tok, prefix, word, suffix):
    """Reimplements the exact arithmetic in rome/repr_tools.py lines 56-65."""
    prefix_len = len(tok.encode(prefix))
    prompt_len = len(tok.encode(prefix + word))
    word_len = prompt_len - prefix_len
    # subtoken == "last"
    return prefix_len + word_len - 1

def ground_truth_idx(tok, prefix, word, suffix):
    """Tokenize the FULL string once and find where the word's last token
    actually lands by checking decoded spans -- the 'honest' way to do it."""
    full = prefix + word + suffix
    ids = tok.encode(full)
    # Find the token span whose decoded text ends exactly at the end of `word`
    # by incrementally decoding and comparing character offsets.
    target_char_end = len(prefix + word)
    running = ""
    for i, tid in enumerate(ids):
        running = tok.decode(ids[: i + 1])
        if len(running) >= target_char_end:
            return i
    return None

def audit(model_name, cases):
    print(f"=== {model_name} ===")
    tok = AutoTokenizer.from_pretrained(model_name)
    mismatches = 0
    for prefix, word, suffix in cases:
        pred = alphaedit_predicted_idx(tok, prefix, word, suffix)
        truth = ground_truth_idx(tok, prefix, word, suffix)
        full_ids = tok.encode(prefix + word + suffix)
        full_toks = tok.convert_ids_to_tokens(full_ids)
        pred_tok = full_toks[pred] if 0 <= pred < len(full_toks) else "OUT_OF_RANGE"
        truth_tok = full_toks[truth] if truth is not None and 0 <= truth < len(full_toks) else "?"
        status = "OK" if pred == truth else "MISMATCH"
        if status == "MISMATCH":
            mismatches += 1
        print(f"  [{status}] word='{word}' | predicted_idx={pred} ('{pred_tok}') "
              f"| ground_truth_idx={truth} ('{truth_tok}')")
    print(f"  -> {mismatches}/{len(cases)} mismatches\n")
    return mismatches

# Use a sample of real CounterFact-style (prefix, subject, suffix) triples.
# Swap these for actual rows pulled from your CounterFact/zsRE run for the
# real experiment -- this is just to get the diagnostic running quickly.
test_cases = [
    ("The mother tongue of ", "Danielle Darrieux", " is"),
    ("", "Eiffel Tower", " is located in"),
    ("I really like ", "Cristiano Ronaldo", "'s playing style"),
    ("The capital of ", "Burkina Faso", " is"),
    ("", "Schadenfreude", " describes a feeling of joy"),
]

if __name__ == "__main__":
    for name in ["google/gemma-2-2b", "microsoft/Phi-3-mini-4k-instruct"]:
        audit(name, test_cases)
