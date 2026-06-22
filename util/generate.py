import unicodedata
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util.logit_lens import LogitLens
from util.device import get_model_input_device


def generate_interactive(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    top_k: int = 5,
    max_out_len: int = 200,
    compare_against: Optional[AutoModelForCausalLM] = None,
    use_logit_lens: bool = False,
    layer_module_tmp: str = "transformer.h.{}",
    ln_f_module: str = "transformer.ln_f",
    lm_head_module: str = "lm_head",
):
    """
    Puts generation in a loop. Allows users to repeatedly provide inputs
    with which text is generated.
    """

    if use_logit_lens:
        llens_gen = LogitLens(
            model,
            tok,
            layer_module_tmp,
            ln_f_module,
            lm_head_module,
            disabled=not use_logit_lens,
        )
        if compare_against:
            llens_vanilla = LogitLens(
                compare_against,
                tok,
                layer_module_tmp,
                ln_f_module,
                lm_head_module,
                disabled=not use_logit_lens,
            )

    while True:
        prompt = input("Enter a prompt: ").strip(" \r\t\n")

        print(
            f"Argument Model: "
            f"{generate_fast(model, tok, [prompt], n_gen_per_prompt=1, top_k=top_k, max_out_len=max_out_len)}"
        )
        if compare_against:
            print(
                f"Baseline Model: "
                f"{generate_fast(compare_against, tok, [prompt], n_gen_per_prompt=1, top_k=top_k, max_out_len=max_out_len)}"
            )

        if use_logit_lens:
            inp_prompt = tok([prompt], padding=True, return_tensors="pt").to(
                get_model_input_device(model)
            )

            with llens_gen:
                model(**inp_prompt)
            print("\n--- Argument Model Logit Lens ---")
            llens_gen.pprint()

            if compare_against:
                with llens_vanilla:
                    compare_against(**inp_prompt.to(get_model_input_device(compare_against)))
                print("--- Baseline Model Logit Lens ---")
                llens_vanilla.pprint()

        print()


def generate_fast(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    n_gen_per_prompt: int = 1,
    top_k: int = 5,
    max_out_len: int = 200,
):
    """Wrapper around HuggingFace model.generate using top-k sampling.

    This avoids the custom KV-cache handling that was causing misalignment
    when prompts had different lengths. It keeps the same signature and
    return type as the original generate_fast.
    """

    device = get_model_input_device(model)

    # Unroll prompts (n_gen_per_prompt times) to match previous behavior
    inp = [prompt for prompt in prompts for _ in range(n_gen_per_prompt)]

    # Tokenize prompts once
    enc = tok(inp, padding=True, return_tensors="pt").to(device)

    # Use HF generation API with top-k sampling and no temperature change
    with torch.no_grad():
        gen_ids = model.generate(
            **enc,
            max_new_tokens=max_out_len,
            do_sample=True,
            top_k=top_k,
            pad_token_id=tok.pad_token_id,
        )


    txt = [tok.decode(x, skip_special_tokens=True) for x in gen_ids.detach().cpu().numpy().tolist()]
    txt = [unicodedata.normalize("NFKD", x).replace("\n\n", " ") for x in txt]

    return txt
